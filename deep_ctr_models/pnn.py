"""
    @Reference  Product-based neural networks for user response prediction[C].
"""
import tensorflow as tf


def pnn_model_fn(features, labels, mode, params):
    deep_columns     = params['deep_columns']
    deep_fields_size = params['deep_fields_size']
    org_emb_size     = params['embedding_dim']
    wide_columns     = params['wide_columns']
    wide_fields_size = params['wide_fields_size']
    deep_input_layer = tf.feature_column.input_layer(features=features, feature_columns=deep_columns)
    deep_feat_emb = tf.reshape(deep_input_layer, [-1, deep_fields_size, org_emb_size])

    with tf.name_scope('wide'):
        wide_input_layer = tf.feature_column.input_layer(features=features, feature_columns=wide_columns)
        wide_output_layer = tf.layers.dense(inputs=wide_input_layer, units=1, activation=None, use_bias=True)

    with tf.name_scope('deep'):
        d_layer_1 = tf.layers.dense(inputs=deep_input_layer, units=50, activation=tf.nn.relu, use_bias=True)
        bn_layer_1 = tf.layers.batch_normalization(inputs=d_layer_1, axis=-1, momentum=0.99, epsilon=0.001, center=True, scale=True)
        deep_output_layer = tf.layers.dense(inputs=bn_layer_1, units=40, activation=tf.nn.relu, use_bias=True)

    with tf.name_scope('inner_product'):
        inner_product = tf.matmul(deep_feat_emb, tf.transpose(deep_feat_emb, perm=[0,2,1]))      # batch * feature_size * feature_size
        inner_product = tf.reshape(inner_product, [-1, deep_fields_size * deep_fields_size])

    with tf.name_scope('outer_product'):
        outer_collection = []
        for i in range(deep_fields_size):
            for j in range(i+1, deep_fields_size):
                vi = tf.gather(deep_feat_emb, indices=i, axis=1, batch_dims=0, name='vi')        # batch * embedding_size
                vj = tf.gather(deep_feat_emb, indices=j, axis=1, batch_dims=0, name='vj')        # batch * embedding_size
                outer_collection.append(tf.reshape(tf.einsum('ai,aj->aij',vi,vj), [-1, org_emb_size * org_emb_size])) # batch * (emb * emb)
            outer_product = tf.concat(outer_collection, axis=1)

    with tf.name_scope('concat'):
        m_layer = tf.concat([wide_output_layer, deep_output_layer, inner_product, outer_product], axis=-1, name='concat')
        o_layer = tf.layers.dense(inputs=m_layer, units=1, activation=None, use_bias=True)

    with tf.name_scope('logit'):
        o_prob = tf.nn.sigmoid(o_layer)
        predictions = tf.cast((o_prob > 0.5), tf.float32)

    if mode == tf.estimator.ModeKeys.PREDICT:
        predictions = {
            'probabilities':    o_prob,
            'label':            predictions
        }
        return tf.estimator.EstimatorSpec(mode, predictions=predictions)

    loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=labels, logits=o_layer))

    if mode == tf.estimator.ModeKeys.TRAIN:
        if   params['optimizer'] == 'adam':
            optimizer = tf.train.AdamOptimizer(learning_rate=params['learning_rate'], beta1=0.9, beta2=0.999, epsilon=1e-8)
        elif params['optimizer'] == 'adagrad':
            optimizer = tf.train.AdagradOptimizer(learning_rate=params['learning_rate'], initial_accumulator_value=1e-8)
        elif params['optimizer'] == 'momentum':
            optimizer = tf.train.MomentumOptimizer(learning_rate=params['learning_rate'], momentum=0.95)
        elif params['optimizer'] == 'ftrl':
            optimizer = tf.train.FtrlOptimizer(learning_rate=params['learning_rate'])
        else:
            optimizer = tf.train.GradientDescentOptimizer(learning_rate=params['learning_rate'])
        train_op = optimizer.minimize(loss=loss, global_step=tf.train.get_global_step())
        return tf.estimator.EstimatorSpec(mode, loss=loss, train_op=train_op)

    if mode == tf.estimator.ModeKeys.EVAL:
        accuracy = tf.metrics.accuracy(labels, predictions)
        auc = tf.metrics.auc(labels, predictions)
        my_metrics = {
            'accuracy': tf.metrics.accuracy(labels, predictions),
            'auc':      tf.metrics.auc(labels,predictions)
        }
        tf.summary.scalar('accuracy', accuracy[1])
        tf.summary.scalar('auc', auc[1])
        return tf.estimator.EstimatorSpec(mode, loss=loss, eval_metric_ops=my_metrics)
