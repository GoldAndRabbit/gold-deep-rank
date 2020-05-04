import tensorflow as tf
import tensorflow.feature_column as fc
from utils.census_feat_config import build_census_emb_columns


def autoint_model_fn(features, labels, mode, params):
    columns, feat_field_size = build_census_emb_columns()
    input_layer = tf.feature_column.input_layer(features=features, feature_columns=columns)

    def normalize(inputs, epsilon=1e-8):
        inputs_shape = inputs.get_shape()
        params_shape = inputs_shape[-1:]
        mean, variance = tf.nn.moments(inputs, [-1], keep_dims=True)
        beta = tf.Variable(tf.zeros(params_shape))
        gamma = tf.Variable(tf.ones(params_shape))
        normalized = (inputs - mean) / ((variance + epsilon) ** (.5))
        outputs = gamma * normalized + beta
        return outputs

    total_feat = tf.reshape(input_layer, [-1, feat_field_size, 32])
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
    outputs = normalize(outputs)

    output_size = outputs.get_shape().as_list()[1] * outputs.get_shape().as_list()[2]
    outputs = tf.reshape(outputs, shape=[-1, output_size])
    print('outputs:', outputs.get_shape().as_list())

    o_layer = tf.layers.dense(inputs=outputs, units=1, activation=None, use_bias=True)
    print('o_layer:', o_layer.get_shape().as_list())

    with tf.name_scope('predicted_label'):
        o_prob = tf.nn.sigmoid(o_layer)
        predictions = tf.cast((o_prob > 0.5), tf.float32)

    labels = tf.cast(labels, tf.float32, name='true_label')
    # define loss
    loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=labels, logits=o_layer))
    # evaluation
    accuracy = tf.metrics.accuracy(labels, predictions)
    auc = tf.metrics.auc(labels, predictions)
    my_metrics = {
        'accuracy': tf.metrics.accuracy(labels, predictions),
        'auc': tf.metrics.auc(labels, predictions)
    }
    tf.summary.scalar('accuracy', accuracy[1])
    tf.summary.scalar('auc', auc[1])

    # define train_op
    optimizer = tf.train.AdamOptimizer(learning_rate=0.0004, beta1=0.9, beta2=0.999)
    train_op = optimizer.minimize(loss=loss, global_step=tf.train.get_global_step())

    if mode == tf.estimator.ModeKeys.TRAIN:
        return tf.estimator.EstimatorSpec(mode, loss=loss, train_op=train_op)
    elif mode == tf.estimator.ModeKeys.EVAL:
        return tf.estimator.EstimatorSpec(mode, loss=loss, eval_metric_ops=my_metrics)
    elif mode == tf.estimator.ModeKeys.PREDICT:
        return tf.estimator.EstimatorSpec(mode, predictions=predictions)
    else:
        print('ERROR')