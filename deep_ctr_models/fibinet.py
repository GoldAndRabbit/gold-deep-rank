"""
    @Date       2020/06/13
    @Author     AA Gold
    @Reference  FiBiNET: Combining Feature Importance and Bilinear feature Interaction for Click-Through Rate Prediction[J].
"""
import random
import tensorflow as tf


def fibinet_model_fn(features, labels, mode, params):
    deep_columns = params['deep_columns']
    deep_fields_size = params['deep_fields_size']
    org_emb_size = params['embedding_dim']
    wide_columns = params['wide_columns']
    wide_fields_size = params['wide_fields_size']
    deep_input_layer = tf.feature_column.input_layer(features=features, feature_columns=deep_columns)

    def _build_bilinear_layers(net, params):
        feat_emb = tf.reshape(net, (-1, deep_fields_size, org_emb_size))
        cnt = 0
        element_wise_product_list = []
        for i in range(0, deep_fields_size):
            for j in range(i+1, deep_fields_size):
                with tf.variable_scope('weight_', reuse=tf.AUTO_REUSE):
                    weight = tf.get_variable(name='weight_' + str(cnt), shape=[org_emb_size, org_emb_size], initializer=tf.glorot_normal_initializer(seed=random.randint(0, 1024)), dtype=tf.float32)
                element_wise_product_list.append(tf.multiply(tf.matmul(feat_emb[:, i, :], weight), feat_emb[:, j, :]))
                cnt += 1
        element_wise_product = tf.stack(element_wise_product_list)
        element_wise_product = tf.transpose(element_wise_product, perm=[1, 0, 2], name="element_wise_product")
        bilinear_output = tf.layers.flatten(element_wise_product)
        return bilinear_output

    def _build_SENET_layers(net, params):
        reduction_ratio = params['fibinet']['reduction_ratio']
        feat_emb = tf.reshape(net, (-1, deep_fields_size, org_emb_size))
        original_feature = feat_emb
        if params['fibinet']['pooling'] == "max":
            feat_emb = tf.reduce_max(feat_emb, axis=2)
        else:
            feat_emb = tf.reduce_mean(feat_emb, axis=2)
        reduction_num = int(max(deep_fields_size / reduction_ratio, 1))
        att_layer = tf.layers.dense(feat_emb, units=reduction_num, activation=tf.nn.relu, kernel_initializer=tf.glorot_uniform_initializer())   # (b, f/r)
        att_layer = tf.layers.dense(att_layer, units=deep_fields_size,activation=tf.nn.relu, kernel_initializer=tf.glorot_uniform_initializer())          # (b, f)
        senet_layer = original_feature * tf.expand_dims(att_layer, axis=-1)
        senet_output = tf.layers.flatten(senet_layer)
        return senet_output

    with tf.name_scope('wide'):
        wide_input_layer = tf.feature_column.input_layer(features=features, feature_columns=wide_columns)
        wide_output_layer = tf.layers.dense(inputs=wide_input_layer, units=1, activation=None, use_bias=True)

    with tf.name_scope('deep'):
        d_layer_1 = tf.layers.dense(inputs=deep_input_layer, units=128, activation=tf.nn.relu, use_bias=True)
        d_layer_2 = tf.layers.dense(inputs=d_layer_1, units=64, activation=tf.nn.relu, use_bias=True)
        deep_output_layer = tf.layers.dense(inputs=d_layer_2, units=40, activation=tf.nn.relu, use_bias=True)

    with tf.name_scope('fibi_net'):
        senet_layer = _build_SENET_layers(deep_input_layer, params)
        combination_layer = tf.concat([_build_bilinear_layers(deep_input_layer, params),
                                       _build_bilinear_layers(senet_layer, params)], axis=1)

    with tf.name_scope('concat'):
        m_layer = tf.concat([combination_layer, deep_output_layer], axis=-1, name='concat')
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
            'auc':      tf.metrics.auc(labels,      predictions)
        }
        tf.summary.scalar('accuracy', accuracy[1])
        tf.summary.scalar('auc', auc[1])
        return tf.estimator.EstimatorSpec(mode, loss=loss, eval_metric_ops=my_metrics)
