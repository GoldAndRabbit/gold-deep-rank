# import tensorflow as tf
import tensorflow.compat.v1 as tf

def dcn_model_fn(features, labels, mode, params):
    deep_columns     = params['deep_columns']
    deep_fields_size = params['deep_fields_size']
    org_emb_size     = params['embedding_dim']
    wide_columns     = params['wide_columns']
    wide_fields_size = params['wide_fields_size']
    input_layer = tf.feature_column.input_layer(features=features, feature_columns=deep_columns)

    def _cross_variable_create(column_num):
        w = tf.Variable(
            tf.random_normal((column_num, 1), mean=0.0, stddev=0.5), dtype=tf.float32)
        b = tf.Variable(
            tf.random_normal((column_num, 1), mean=0.0, stddev=0.5), dtype=tf.float32)
        return w, b

    def _cross_op(x0, x, w, b):
        x0 = tf.expand_dims(x0, axis=2)                                             # mxdx1
        x = tf.expand_dims(x, axis=2)                                               # mxdx1
        multiple = w.get_shape().as_list()[0]
        x0_broad_horizon = tf.tile(x0, [1, 1, multiple])                            # mxdx1 -> mxdxd
        x_broad_vertical = tf.transpose(tf.tile(x, [1, 1, multiple]), [0, 2, 1])    # mxdx1 -> mxdxd #
        w_broad_horizon = tf.tile(w, [1, multiple])                                 # dx1 -> dxd #
        mid_res = tf.multiply(tf.multiply(x0_broad_horizon, x_broad_vertical), w)   # mxdxd # here use broadcast compute #
        res = tf.reduce_sum(mid_res, axis=2)                                        # mxd #
        # mxd + 1xd # here also use broadcast compute #a
        res = res + tf.transpose(b)
        return res

    with tf.name_scope('cross'):
        column_num = input_layer.get_shape().as_list()[1]
        c_w_1, c_b_1 = _cross_variable_create(column_num)
        c_w_2, c_b_2 = _cross_variable_create(column_num)
        c_layer_1 = _cross_op(input_layer, input_layer, c_w_1, c_b_1) + input_layer
        c_layer_2 = _cross_op(input_layer, c_layer_1, c_w_2, c_b_2) + c_layer_1

    with tf.name_scope('deep'):
        d_layer_1 = tf.layers.dense(inputs=input_layer, units=128, activation=tf.nn.relu, use_bias=True)
        # bn_layer_1 = tf.layers.batch_normalization(inputs=d_layer_1, axis=-1, momentum=0.99, epsilon=0.001, center=True, scale=True)
        d_layer_2 = tf.layers.dense(inputs=d_layer_1, units=64, activation=tf.nn.relu, use_bias=True)

    with tf.name_scope('concat'):
        m_layer = tf.concat([d_layer_2, c_layer_2], 1)
        m_layer_1 = tf.layers.dense(inputs=m_layer, units=32, activation=tf.nn.relu, use_bias=True)
        o_layer = tf.layers.dense(inputs=m_layer_1, units=1, activation=None, use_bias=True)

    with tf.name_scope('predicted_label'):
        o_prob = tf.nn.sigmoid(o_layer)
        predictions = tf.cast((o_prob > 0.5), tf.float32)

    if mode == tf.estimator.ModeKeys.PREDICT:
        predictions = {
            'probabilities': o_prob,
            'label': predictions
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

