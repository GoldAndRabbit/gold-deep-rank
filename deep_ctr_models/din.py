import tensorflow as tf
from utils.census_ctr_feat_config import build_ama_ele_columns


def attention_layer(querys, keys, keys_id):
    keys_length = tf.shape(keys)[1]
    embedding_size = querys.get_shape().as_list()[-1]
    keys = tf.reshape(keys, shape=[-1, keys_length, embedding_size])
    querys = tf.reshape(tf.tile(querys, [1, keys_length]), shape=[-1, keys_length, embedding_size])

    net = tf.concat([keys, keys - querys, querys, keys * querys], axis=-1)
    for units in [32, 16]:
        net = tf.layers.dense(net, units=units, activation=tf.nn.relu)
    att_wgt = tf.layers.dense(net, units=1, activation=tf.sigmoid)
    outputs = tf.reshape(att_wgt,shape=[-1, 1, keys_length], name="weight")

    scores = outputs
    key_masks = tf.expand_dims(tf.not_equal(keys_id, '0'),axis=1)
    paddings = tf.ones_like(scores) * (-2 ** 32 + 1)
    scores = tf.where(key_masks,scores,paddings)
    scores = scores / (embedding_size ** 0.5)
    scores = tf.nn.softmax(scores)
    outputs = tf.matmul(scores, keys)
    outputs = tf.reduce_sum(outputs, 1, name="attention_embedding")
    return outputs


def din_model_fn(features, labels, mode, params):
    emb_feat_columns, emb_field_size = build_ama_ele_columns()
    emb_input_layer = tf.feature_column.input_layer(features=features, feature_columns=emb_feat_columns)

    with tf.name_scope('dnn_layers'):
        d_layer_1 = tf.layers.dense(inputs=emb_input_layer, units=32, activation=tf.nn.relu, use_bias=True)
        bn_layer_1 = tf.layers.batch_normalization(inputs=d_layer_1, axis=-1, momentum=0.99, epsilon=0.001, center=True, scale=True)
        d_layer_2 = tf.layers.dense(inputs=bn_layer_1, units=16, activation=tf.nn.relu, use_bias=True)
        dnn_output_layer = tf.layers.dense(inputs=d_layer_2, units=8, activation=tf.nn.relu, use_bias=True)

    # with tf.name_scope('din_attention_layers'):
    #     seq_item_hash = tf.string_to_hash_bucket_fast(features["seq"], 200000)                      # seq item
    #     target_item_hash = tf.string_to_hash_bucket_fast(features["item_id"], 200000)               # target item emb
    #     item_lookup_mat = tf.get_variable(name='item_lookup_mat', dtype=tf.float32, shape=[200000, 20])
    #     seq_item_emb = tf.nn.embedding_lookup(item_lookup_mat, seq_item_hash)
    #     target_item_emb = tf.nn.embedding_lookup(item_lookup_mat, target_item_hash)
    #     seq_item_att = attention_layer(target_item_emb, seq_item_emb, features["seq"])
    #
    #     seq_cate_hash = tf.string_to_hash_bucket_fast(features["seq_cate"], 200000)                 # seq cate
    #     target_cate_hash = tf.string_to_hash_bucket_fast(features["item_cate"], 200000)             # target cate emb
    #     cate_lookup_mat = tf.get_variable(name='cate_lookup_mat', dtype=tf.float32, shape=[200000, 20])
    #     seq_cate_emb = tf.nn.embedding_lookup(cate_lookup_mat, seq_cate_hash)
    #     target_cate_emb = tf.nn.embedding_lookup(cate_lookup_mat, target_cate_hash)
    #     seq_cate_att = attention_layer(target_cate_emb, seq_cate_emb, features["seq_cate"])

    with tf.name_scope('concat_layers'):
        m_layer = tf.concat([dnn_output_layer], axis=1)
        # m_layer = tf.concat([dnn_output_layer, seq_item_att, seq_cate_att], axis=1)
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