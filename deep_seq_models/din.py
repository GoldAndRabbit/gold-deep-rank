import math
import tensorflow as tf
from utils.ama_ele_feat_config import build_ama_ele_columns


def din_model_fn(features, labels, mode, params):
    def _attention_layer(queries,keys):
        """
        queries: (batch_size * emb_size)
        keys:    (batch_size * seq_size * emb_size)
        """
        emb_size = queries.get_shape().as_list()[-1]                                # get emb_size
        seq_size = keys.get_shape().as_list()[1]                                    # get seq_size
        queries = tf.reshape(tf.tile(queries,[1,seq_size]),
                             [-1,seq_size,emb_size])                                # tile queries as same as keys shape, then reshape
        att_unit = tf.concat([queries,keys,queries - keys,queries * keys], axis=-1) # batch_size * seq_size * [4 * emb_size]
        unit_layer_1 = tf.layers.dense(att_unit,80,activation=tf.nn.sigmoid)
        unit_layer_2 = tf.layers.dense(unit_layer_1,40,activation=tf.nn.sigmoid)
        unit_layer_3 = tf.layers.dense(unit_layer_2,1,activation=None)              # batch_size * seq_size * 1
        outputs = tf.reshape(unit_layer_3,[-1,1,seq_size])                          # batch_size * 1 * seq_size
        outputs = outputs / (math.sqrt(emb_size))                                   # scale
        outputs = tf.nn.softmax(outputs)                                            # softmax
        outputs = tf.matmul(outputs, keys)                                          # weighted sum: batch_size * ((1 * seq_size) * (seq_size * emb_size)) => batch_size * 1 * emb_size
        outputs = tf.reshape(outputs,[-1, emb_size])
        return outputs

    emb_feat_columns, emb_field_size = build_ama_ele_columns()
    emb_input_layer = tf.feature_column.input_layer(features=features, feature_columns=emb_feat_columns)

    with tf.name_scope('deep'):
        d_layer_1 = tf.layers.dense(inputs=emb_input_layer, units=128, activation=tf.nn.relu, use_bias=True)
        bn_layer_1 = tf.layers.batch_normalization(inputs=d_layer_1, axis=-1, momentum=0.99, epsilon=0.001, center=True, scale=True)
        d_layer_2 = tf.layers.dense(inputs=bn_layer_1, units=64, activation=tf.nn.relu, use_bias=True)
        deep_output_layer = tf.layers.dense(inputs=d_layer_2, units=32, activation=tf.nn.relu, use_bias=True)

    with tf.name_scope('attention'):
        seq_items = tf.string_to_hash_bucket_fast(features['seq'], 200000)
        target_item = tf.string_to_hash_bucket_fast(features['item_id'], 200000)
        item_emb_matrix = tf.get_variable(name='item_emb_table', dtype=tf.float32, shape=[200000, 8])
        seq_items_emb = tf.nn.embedding_lookup(item_emb_matrix, seq_items)      # keys:    (batch_size * seq_size * emb_size)
        target_item_emb = tf.nn.embedding_lookup(item_emb_matrix, target_item)  # queries: (batch_size * emb_size)

        seq_cates = tf.string_to_hash_bucket_fast(features['seq_cate'], 200000)
        target_cate = tf.string_to_hash_bucket_fast(features['item_cate'], 200000)
        cate_emb_matrix = tf.get_variable(name='cate_emb_table', dtype=tf.float32, shape=[200000, 8])
        seq_cates_emb = tf.nn.embedding_lookup(cate_emb_matrix, seq_cates)
        target_cate_emb = tf.nn.embedding_lookup(cate_emb_matrix, target_cate)

        seq_items_attention = _attention_layer(target_item_emb, seq_items_emb)
        seq_cates_attention = _attention_layer(target_cate_emb, seq_cates_emb)

    with tf.name_scope('concat'):
        m_layer = tf.concat([deep_output_layer, seq_items_attention, seq_cates_attention], axis=1)
        o_layer = tf.layers.dense(inputs=m_layer, units=1, activation=None, use_bias=True)

    with tf.name_scope('logit'):
        o_prob = tf.nn.sigmoid(o_layer)
        predictions = tf.cast((o_prob > 0.5), tf.float32)

    labels = tf.cast(labels, tf.float32, name='true_label')
    loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=labels, logits=o_layer))
    accuracy = tf.metrics.accuracy(labels, predictions)
    auc = tf.metrics.auc(labels, predictions)
    my_metrics = {
        'accuracy': tf.metrics.accuracy(labels, predictions),
        'auc':      tf.metrics.auc(labels, predictions)
    }
    tf.summary.scalar('accuracy', accuracy[1])
    tf.summary.scalar('auc', auc[1])

    optimizer = tf.train.AdamOptimizer(learning_rate=0.001, beta1=0.9, beta2=0.999)
    train_op = optimizer.minimize(loss=loss, global_step=tf.train.get_global_step())

    if mode == tf.estimator.ModeKeys.TRAIN:
        return tf.estimator.EstimatorSpec(mode, loss=loss, train_op=train_op)
    elif mode == tf.estimator.ModeKeys.EVAL:
        return tf.estimator.EstimatorSpec(mode, loss=loss, eval_metric_ops=my_metrics)
    elif mode == tf.estimator.ModeKeys.PREDICT:
        return tf.estimator.EstimatorSpec(mode, predictions=predictions)
    else:
        print('ERROR')