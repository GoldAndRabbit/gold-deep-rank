"""
    @Reference  AutoInt: Automatic Feature Interaction Learning via Self-Attentive Neural Networks[J]
"""
import tensorflow as tf


def autoint_model_fn(features, labels, mode, params):
    def _normalize(inputs, epsilon=1e-8):
        inputs_shape = inputs.get_shape()
        params_shape = inputs_shape[-1:]
        mean, variance = tf.nn.moments(inputs, [-1], keep_dims=True)
        beta = tf.Variable(tf.zeros(params_shape))
        gamma = tf.Variable(tf.ones(params_shape))
        normalized = (inputs - mean) / ((variance + epsilon) ** 0.5)
        outputs = gamma * normalized + beta
        return outputs

    deep_columns = params['deep_columns']
    deep_fields_size = params['deep_fields_size']
    wide_columns = params['wide_columns']
    wide_fields_size = params['wide_fields_size']
    org_emb_size = params['embedding_dim']
    input_layer = tf.feature_column.input_layer(features=features,feature_columns=deep_columns)
    total_feat = tf.reshape(input_layer, [-1, deep_fields_size, org_emb_size])
    att_emb_size = 64
    num_heads = 2
    has_residual = True

    # By linear projection, generate Q, K, V
    Q = tf.layers.dense(total_feat, att_emb_size, activation=tf.nn.relu)
    print('Q.shape', Q.get_shape().as_list())
    K = tf.layers.dense(total_feat, att_emb_size, activation=tf.nn.relu)
    V = tf.layers.dense(total_feat, att_emb_size, activation=tf.nn.relu)

    if has_residual:
        V_res = tf.layers.dense(total_feat, att_emb_size, activation=tf.nn.relu)

    # Split and concat
    Q_ = tf.concat(tf.split(Q, num_heads, axis=2), axis=0)
    K_ = tf.concat(tf.split(K, num_heads, axis=2), axis=0)
    V_ = tf.concat(tf.split(V, num_heads, axis=2), axis=0)
    print('Q_.shape', Q_.get_shape().as_list())

    # Multiplication
    weights = tf.matmul(Q_, tf.transpose(K_, [0, 2, 1]))
    print('multiplication weights:', weights.get_shape().as_list())

    # Scale
    weights = weights / (K_.get_shape().as_list()[-1] ** 0.5)
    print('Scale weights:', weights.get_shape().as_list())

    # Activation
    weights = tf.nn.softmax(weights)
    print('Activation weights:', weights.get_shape().as_list())

    # Dropouts
    # weights = tf.layers.dropout(weights, rate=1-dropout_keep_prob, training=tf.convert_to_tensor(is_training))

    # Weighted sum
    outputs = tf.matmul(weights, V_)
    print('weighted sum outputs :', outputs.get_shape().as_list())

    # Restore shape
    outputs = tf.concat(tf.split(outputs, num_heads, axis=0), axis=2)
    print('restore shape outputs :', outputs.get_shape().as_list())

    # Residual connection
    if has_residual:
        outputs += V_res

    outputs = tf.nn.relu(outputs)
    print('outputs:', outputs.get_shape().as_list())

    # Normalize
    outputs = _normalize(outputs)

    output_size = outputs.get_shape().as_list()[1] * outputs.get_shape().as_list()[2]
    outputs = tf.reshape(outputs, shape=[-1, output_size])
    print('outputs:', outputs.get_shape().as_list())

    o_layer = tf.layers.dense(inputs=outputs, units=1, activation=None, use_bias=True)
    print('o_layer:', o_layer.get_shape().as_list())

    with tf.name_scope('predicted_label'):
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


