import tensorflow as tf


def xdeepfm_model_fn(features, labels, mode, params):
    deep_columns = params['deep_columns']
    deep_fields_size = params['deep_fields_size']
    wide_columns = params['wide_columns']
    wide_fields_size = params['wide_fields_size']
    emb_input_layer = tf.feature_column.input_layer(features=features, feature_columns=deep_columns)

    with tf.name_scope('wide'):
        wide_input_layer = tf.feature_column.input_layer(features=features, feature_columns=wide_columns)
        wide_output_layer = tf.layers.dense(inputs=wide_input_layer, units=1, activation=None, use_bias=True)

    with tf.name_scope('deep'):
        d_layer_1 = tf.layers.dense(inputs=emb_input_layer, units=50, activation=tf.nn.relu, use_bias=True)
        bn_layer_1 = tf.layers.batch_normalization(inputs=d_layer_1, axis=-1, momentum=0.99, epsilon=0.001, center=True, scale=True)
        deep_output_layer = tf.layers.dense(inputs=bn_layer_1, units=40, activation=tf.nn.relu, use_bias=True)

    with tf.name_scope('cin'):
        ori_emb_input = tf.reshape(emb_input_layer, [-1, deep_fields_size, 32])
        cross_layer_sizes = [64, 64, 32]
        dim = 32 
        final_len = 0
        field_nums = [int(deep_fields_size)]
        cin_layers = [ori_emb_input]
        final_result = []
        split_tensor0 = tf.split(cin_layers[0], dim * [1], 2)
        for idx, layer_size in enumerate(cross_layer_sizes):
            split_tensor = tf.split(cin_layers[-1], dim * [1], 2)
            dot_result_m = tf.matmul(split_tensor0, split_tensor, transpose_b=True)
            dot_result_o = tf.reshape(dot_result_m, shape=[dim, -1, field_nums[0] * field_nums[-1]])
            dot_result = tf.transpose(dot_result_o, perm=[1, 0, 2])
            
            conv_filters = tf.get_variable(name='f_'+str(idx),
                                           shape=[1, field_nums[0] * field_nums[-1], layer_size],
                                           dtype=tf.float32)
            curr_out = tf.nn.conv1d(dot_result, filters=conv_filters, stride=1, padding='VALID')
            curr_out = tf.nn.relu(curr_out)
            curr_out = tf.transpose(curr_out, perm=[0, 2, 1])
            
            dirrect_connect = curr_out
            next_hidden = curr_out
            final_len += int(layer_size)
            field_nums.append(int(layer_size))
            
            final_result.append(dirrect_connect)            
            cin_layers.append(next_hidden)
        result = tf.concat(final_result, axis=1)
        result = tf.reduce_sum(result, -1)

    with tf.name_scope('concat'):
        m_layer = tf.concat([wide_output_layer, deep_output_layer, result], 1)
        o_layer = tf.layers.dense(inputs=m_layer, units=1, activation=None, use_bias=True)

    with tf.name_scope('logit'):
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
        optimizer = tf.train.AdamOptimizer(learning_rate=0.0004, beta1=0.9, beta2=0.999)
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
