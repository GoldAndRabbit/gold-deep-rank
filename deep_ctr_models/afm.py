"""
    @Date       2020/06/11
    @Author     AA Gold
    @Reference  Attentional factorization machines: Learning the weight of feature interactions via attention networks[J].
"""
import numpy as np
import tensorflow as tf


def afm_model_fn(features, labels, mode, params):
    deep_columns = params['deep_columns']
    deep_fields_size = params['deep_fields_size']
    org_emb_size = params['embedding_dim']
    wide_columns = params['wide_columns']
    wide_fields_size = params['wide_fields_size']
    hidden_factor = [16, 16]
    deep_input_layer = tf.feature_column.input_layer(features=features, feature_columns=deep_columns)

    with tf.name_scope('wide'):
        wide_input_layer = tf.feature_column.input_layer(features=features, feature_columns=wide_columns)
        wide_output_layer = tf.layers.dense(inputs=wide_input_layer, units=1, activation=None, use_bias=True)

    with tf.name_scope('mlp'):
        d_layer_1 = tf.layers.dense(inputs=deep_input_layer, units=50, activation=tf.nn.relu, use_bias=True)
        bn_layer_1 = tf.layers.batch_normalization(inputs=d_layer_1, axis=-1, momentum=0.99, epsilon=0.001, center=True, scale=True)
        mlp_output_layer = tf.layers.dense(inputs=bn_layer_1, units=40, activation=tf.nn.relu, use_bias=True)

    with tf.name_scope('afm_attention'):
        def _get_attention_weights(element_wise_product):
            att_weights = {}
            glorot = np.sqrt(2.0 / (hidden_factor[0] + hidden_factor[1]))
            att_weights['att_p'] = tf.Variable(np.random.normal(loc=0, scale=1, size=(hidden_factor[0])), dtype=np.float32, name='att_p')
            att_weights['att_b'] = tf.Variable(np.random.normal(loc=0, scale=glorot, size=(1, hidden_factor[0])), dtype=np.float32, name='att_b')
            att_mul = tf.layers.dense(element_wise_product,units=hidden_factor[0], kernel_initializer=tf.glorot_uniform_initializer())  # (b, f*(f-1)/2, hidden_factor[0])
            att_relu = tf.reduce_sum(tf.multiply(att_weights['att_p'],tf.nn.relu(att_mul + att_weights['att_b'])),2, keep_dims=True)
            att_out = tf.nn.softmax(att_relu)                                                                                           # (batch_size, f*(f-1)/2, 1)
            return att_out
        feature_embeddings = tf.reshape(deep_input_layer, (-1, deep_fields_size, org_emb_size))                 # (batch_size, deep_fields_size, org_emb_size)
        element_wise_product_list = []
        for i in range(0, deep_fields_size):
            for j in range(i + 1, deep_fields_size):
                element_wise_product_list.append(tf.multiply(feature_embeddings[:, i, :], feature_embeddings[:, j, :]))
        element_wise_product = tf.stack(element_wise_product_list)                                              # (f*(f-1)/2, batch_size, org_emb_size)
        element_wise_product = tf.transpose(element_wise_product, perm=[1, 0, 2], name='element_wise_product')  # (batch_size, f*(f-1)/2, org_emb_size)
        att_out = _get_attention_weights(element_wise_product)
        afm_output = tf.reduce_sum(tf.multiply(att_out, element_wise_product), 1, name='afm')                   # (batch_size, k)

    with tf.name_scope('concat'):
        m_layer = tf.concat([wide_output_layer, mlp_output_layer, afm_output], axis=-1, name='concat')
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
        if params['optimizer'] == 'adam':
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
            'auc': tf.metrics.auc(labels,predictions)
        }
        tf.summary.scalar('accuracy', accuracy[1])
        tf.summary.scalar('auc', auc[1])
        return tf.estimator.EstimatorSpec(mode, loss=loss, eval_metric_ops=my_metrics)