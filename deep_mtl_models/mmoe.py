import tensorflow as tf
from feat_config.census_mtl_feat_config import build_mtl_census_emb_columns


def build_deep_layers(net):
    for num_hidden_units in [32, 16]:
        net = tf.layers.dense(
            net,
            units=num_hidden_units,
            activation=tf.nn.relu,
            kernel_initializer=tf.glorot_uniform_initializer())
    return net


def mmoe_model_fn(features, labels, mode, params):
    columns, feat_field_size = build_mtl_census_emb_columns()
    net = tf.feature_column.input_layer(features, columns)

    expert_one = build_deep_layers(net)
    expert_one = tf.layers.dense(expert_one, units=1, activation=tf.nn.relu, kernel_initializer=tf.glorot_uniform_initializer())
    expert_two = build_deep_layers(net)
    expert_two = tf.layers.dense(expert_two, units=1, activation=tf.nn.relu, kernel_initializer=tf.glorot_uniform_initializer())
    expert_three = build_deep_layers(net)
    expert_three = tf.layers.dense(expert_three, units=1, activation=tf.nn.relu, kernel_initializer=tf.glorot_uniform_initializer())
    expert_out = tf.concat([expert_one, expert_two, expert_three], axis=-1)

    weight_1 = tf.layers.dense(net, units=3, activation=tf.nn.relu, kernel_initializer=tf.glorot_uniform_initializer())
    gate_1 = tf.nn.softmax(weight_1)

    weight_2 = tf.layers.dense(net, units=3, activation=tf.nn.relu, kernel_initializer=tf.glorot_uniform_initializer())
    gate_2 = tf.nn.softmax(weight_2)

    tower_1 = tf.multiply(expert_out, gate_1)

    tower_2 = tf.multiply(expert_out, gate_2)

    ctr_logits = tf.layers.dense(tower_1, units=1, kernel_initializer=tf.glorot_uniform_initializer())
    cvr_logits = tf.layers.dense(tower_2, units=1, kernel_initializer=tf.glorot_uniform_initializer())
    ctr_preds = tf.sigmoid(ctr_logits)
    cvr_preds = tf.sigmoid(cvr_logits)
    ctcvr_preds = tf.multiply(ctr_preds, cvr_preds)

    optimizer = tf.train.AdamOptimizer(learning_rate=0.0004, beta1=0.9, beta2=0.999)
    ctr_label = labels['ctr']
    cvr_label = labels['cvr']

    ctr_loss = tf.reduce_sum(tf.nn.sigmoid_cross_entropy_with_logits(labels=ctr_label, logits=ctr_logits))
    ctcvr_loss = tf.reduce_sum(tf.losses.log_loss(labels=cvr_label, predictions=ctcvr_preds))
    loss = ctr_loss + ctcvr_loss

    if mode == tf.estimator.ModeKeys.TRAIN:
        train_op = optimizer.minimize(loss, global_step=tf.train.get_global_step())
        return tf.estimator.EstimatorSpec(mode, loss=loss, train_op=train_op)
    elif mode == tf.estimator.ModeKeys.EVAL:
        ctr_accuracy = tf.metrics.accuracy(labels=ctr_label, predictions=tf.to_float(tf.greater_equal(ctr_preds, 0.5)))
        cvr_accuracy = tf.metrics.accuracy(labels=cvr_label, predictions=tf.to_float(tf.greater_equal(ctcvr_preds, 0.5)))
        ctr_auc = tf.metrics.auc(ctr_label, ctr_preds)
        cvr_auc = tf.metrics.auc(cvr_label, ctcvr_preds)
        my_metrics = {
            'cvr_accuracy': cvr_accuracy,
            'ctr_accuracy': ctr_accuracy,
            'ctr_auc': ctr_auc,
            'cvr_auc': cvr_auc
        }
        tf.summary.scalar('ctr_accuracy', ctr_accuracy[1])
        tf.summary.scalar('cvr_accuracy', cvr_accuracy[1])
        tf.summary.scalar('ctr_auc', ctr_auc[1])
        tf.summary.scalar('cvr_auc', cvr_auc[1])
        return tf.estimator.EstimatorSpec(mode, loss=loss, eval_metric_ops=my_metrics)
    elif mode == tf.estimator.ModeKeys.PREDICT:
        predictions = {
            'probabilities': ctcvr_preds,
            'ctr_probabilities': ctr_preds,
            'cvr_probabilities': cvr_preds
        }
        export_outputs = {
            'prediction': tf.estimator.export.PredictOutput(predictions)
        }
        return tf.estimator.EstimatorSpec(mode, predictions=predictions, export_outputs=export_outputs)
    else:
        print('ERROR')