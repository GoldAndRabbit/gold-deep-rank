import tensorflow as tf
from utils.ama_ele_feat_config import build_ama_ele_columns


def attention_layer(querys,keys,keys_id):
    """
        queries:     [Batchsize, 1, embedding_size]
        keys:        [Batchsize, max_seq_len, embedding_size]  max_seq_len is the number of keys(e.g. number of clicked creativeid for each sample)
        keys_id:     [Batchsize, max_seq_len]
    """
    keys_length = tf.shape(keys)[1]  # padded_dim
    embedding_size = querys.get_shape().as_list()[-1]
    keys = tf.reshape(keys,shape=[-1,keys_length,embedding_size])
    querys = tf.reshape(tf.tile(querys,[1,keys_length,1]),shape=[-1,keys_length,embedding_size])

    net = tf.concat([keys,keys - querys,querys,keys * querys],axis=-1)
    for units in [32,16]:
        net = tf.layers.dense(net,units=units,activation=tf.nn.relu)
    att_wgt = tf.layers.dense(net,units=1,activation=tf.sigmoid)  # shape(batch_size, max_seq_len, 1)
    outputs = tf.reshape(att_wgt,shape=[-1,1,keys_length],name="weight")  # shape(batch_size, 1, max_seq_len)

    scores = outputs
    # key_masks = tf.expand_dims(tf.cast(keys_id > 0, tf.bool), axis=1)  # shape(batch_size, 1, max_seq_len) we add 0 as padding
    # tf.not_equal(keys_id, '0')  如果改成str
    key_masks = tf.expand_dims(tf.not_equal(keys_id,'0'),axis=1)
    paddings = tf.ones_like(scores) * (-2 ** 32 + 1)
    scores = tf.where(key_masks,scores,paddings)
    scores = scores / (embedding_size ** 0.5)  # scale
    scores = tf.nn.softmax(scores)
    outputs = tf.matmul(scores,keys)  # (batch_size, 1, embedding_size)
    outputs = tf.reduce_sum(outputs,1,name="attention_embedding")  # (batch_size, embedding_size)
    return outputs

def din_model_fn(features, labels, mode, params):
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
        seq_items_emb = tf.nn.embedding_lookup(item_emb_matrix, seq_items)
        target_item_emb = tf.nn.embedding_lookup(item_emb_matrix, target_item)
        print('seq_items_emb')
        print(seq_items_emb.get_shape())
        print('target_item_emb')
        print(target_item_emb.get_shape())

        seq_cates = tf.string_to_hash_bucket_fast(features['seq_cate'], 200000)
        target_cate = tf.string_to_hash_bucket_fast(features['item_cate'], 200000)
        cate_emb_matrix = tf.get_variable(name='cate_emb_table', dtype=tf.float32, shape=[200000, 8])
        seq_cates_emb = tf.nn.embedding_lookup(cate_emb_matrix, seq_cates)
        target_cate_emb = tf.nn.embedding_lookup(cate_emb_matrix, target_cate)
        print('seq_cates_emb')
        print(seq_cates_emb.get_shape())
        print('target_cate_emb')
        print(target_cate_emb.get_shape())

        # seq_items_attention = attention_layer(target_item_emb, seq_items_emb, features['seq'])
        # seq_cates_attention = attention_layer(target_cate_emb, seq_cates_emb, features['seq_cate'])

    with tf.name_scope('concat'):
        # m_layer = tf.concat([seq_items_attention, seq_cates_attention, deep_output_layer], 1)
        m_layer = tf.concat([deep_output_layer], 1)
        o_layer = tf.layers.dense(inputs=m_layer, units=1, activation=None, use_bias=True)

    with tf.name_scope('logit'):
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