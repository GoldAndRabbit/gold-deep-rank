import pickle
import tensorflow as tf

AMAZON_PROTO = {
    'reviewer_id':          tf.FixedLenFeature([], tf.int64),
    'hist_item_list':       tf.VarLenFeature(tf.int64),
    'hist_category_list':   tf.VarLenFeature(tf.int64),
    'hist_length':          tf.FixedLenFeature([], tf.int64),
    'item':                 tf.FixedLenFeature([], tf.int64),
    'item_category':        tf.FixedLenFeature([], tf.int64),
    'target':               tf.FixedLenFeature([], tf.int64),
}

AMAZON_TARGET = 'target'
AMAZON_VARLEN = ['hist_item_list', 'hist_category_list']

with open('data/amazon/remap.pkl', 'rb') as f:
    _ = pickle.load(f) # uid, iid
    AMAZON_CATE_LIST  = pickle.load(f)
    AMAZON_USER_COUNT, AMAZON_ITEM_COUNT, AMAZON_CATE_COUNT, _ = pickle.load( f )

AMAZON_EMB_DIM = 64

def stack_dense_layer(dense, hidden_units, dropout_rate, batch_norm, mode, add_summary):
    with tf.variable_scope('Dense'):
        for i, unit in enumerate(hidden_units):
            dense = tf.layers.dense(dense, units=unit, activation='relu', name='dense{}'.format(i))
            if batch_norm:
                dense = tf.layers.batch_normalization(dense, center=True, scale=True, trainable=True, training=(mode==tf.estimator.ModeKeys.TRAIN))
            if dropout_rate > 0:
                dense = tf.layers.dropout(dense, rate=dropout_rate, training=(mode==tf.estimator.ModeKeys.TRAIN))
            if add_summary:
                add_layer_summary(dense.name, dense)
    return dense

def build_features():
    # everything but the feature used in attention
    f_reviewer = tf.feature_column.categorical_column_with_identity(
        'reviewer_id',
        num_buckets = AMAZON_USER_COUNT,
        default_value = 0
    )
    f_reviewer = tf.feature_column.embedding_column(f_reviewer, dimension=128)
    f_item_length = tf.feature_column.numeric_column('hist_length')
    f_dense = [f_item_length, f_reviewer]
    return f_dense


def add_layer_summary(tag, value):
    tf.summary.scalar('{}/fraction_of_zero_values'.format(tag), tf.math.zero_fraction(value))
    tf.summary.histogram('{}/activation'.format(tag),  value)

def tf_estimator_model(model_fn):
    def model_fn_helper(features, labels, mode, params):
        y = model_fn(features , labels, mode, params)
        add_layer_summary('label_mean', labels)
        if mode == tf.estimator.ModeKeys.PREDICT:
            predictions = {
                'prediction_prob': tf.sigmoid(y)
            }
            return tf.estimator.EstimatorSpec(mode=tf.estimator.ModeKeys.PREDICT, predictions=predictions)
        cross_entropy = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=labels, logits=y))

        if mode == tf.estimator.ModeKeys.TRAIN:
            optimizer = tf.train.AdagradOptimizer(learning_rate=params['learning_rate'])
            update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
            with tf.control_dependencies( update_ops ):
                train_op = optimizer.minimize(cross_entropy, global_step=tf.train.get_global_step())
            return tf.estimator.EstimatorSpec(mode, loss=cross_entropy, train_op=train_op)
        else:
            eval_metric_ops = {
                'accuracy': tf.metrics.accuracy(labels=labels, predictions=tf.to_float(tf.greater_equal(tf.sigmoid(y), 0.5))),
                'auc':      tf.metrics.auc(labels=labels, predictions=tf.sigmoid(y)),
                'pr':       tf.metrics.auc(labels=labels, predictions=tf.sigmoid(y), curve='PR')
            }
            return tf.estimator.EstimatorSpec(mode, loss=cross_entropy, eval_metric_ops=eval_metric_ops)

    return model_fn_helper

def build_estimator_helper(model_fn, params):
    def build_estimator(config):
        if config.data_name not in model_fn:
            raise Exception('Only [{}] are supported'.format(','.join(model_fn.keys()) ))

        run_config = tf.estimator.RunConfig(
            save_summary_steps=50,
            log_step_count_steps=50,
            keep_checkpoint_max = 3,
            save_checkpoints_steps =50
        )

        if 'model_type' in params:
            # PNN -> PNN/IPNN
            # FiBiNET -> field_all/field_each/field_interaction
            # EMMLP -> dense/bucketize
            model_dir = config.checkpoint_dir + '/' + params['model_type']
        else:
            model_dir = config.checkpoint_dir

        estimator = tf.estimator.Estimator(
            model_fn = model_fn[config.data_name],
            config = run_config,
            params = params[config.data_name],
            model_dir = model_dir
        )

        return estimator
    return build_estimator

def attention(queries, keys, keys_id, params):
    """
    :param queries: target embedding (batch_size * emb_dim)
    :param keys: history embedding (batch * padded_size * emb_dim)
    :param keys_id: history id (batch * padded_size)
    :return: attention_emb: weighted average of history embedding (batch * emb_dim)
    """
    # Differ from paper, for computation efficiency: outer product -> hadamard product
    padded_size = tf.shape(keys)[1]
    queries = tf.tile(tf.expand_dims(queries, axis=1), [1, padded_size, 1])         # batch * emb_dim -> batch * padded_size * emb_dim
    dense = tf.concat([keys, queries, queries-keys, queries*keys], axis=2)          # batch * padded_size * emb_dim
    for i, unit in enumerate(params['attention_hidden_units']):
        dense = tf.layers.dense(dense, units=unit, activation=tf.nn.relu, name='attention_{}'.format(i))
        add_layer_summary(dense.name, dense)
    weight = tf.layers.dense(dense, units=1, activation=tf.sigmoid, name ='attention_weight')   # batch * padded_size * 1
    zero_mask = tf.expand_dims(tf.not_equal(keys_id, 0), axis=2)                    # batch * padded_size * 1
    zero_weight = tf.ones_like(weight) * (-2 ** 32 + 1)                             # small number logits ~ 0
    weight = tf.where(zero_mask, weight, zero_weight)                               # apply zero-mask for padded keys
    weight = tf.nn.softmax(weight)                                                  # rescale weight to sum(weight)=1
    add_layer_summary('attention_weight', weight)
    attention_emb = tf.reduce_mean(tf.multiply(weight, keys), axis=1)               # weight average ->batch * emb_dim
    return attention_emb


@tf_estimator_model
def model_fn_varlen(features, labels, mode, params):
    f_dense = build_features()
    f_dense = tf.feature_column.input_layer(features, f_dense)

    # Embedding Look up: history list item and category list
    item_embedding = tf.get_variable(shape=[params['amazon_item_count'], params['amazon_emb_dim']], initializer=tf.truncated_normal_initializer(), name='item_embedding')
    cate_embedding = tf.get_variable(shape=[params['amazon_cate_count'], params['amazon_emb_dim']], initializer=tf.truncated_normal_initializer(), name='cate_embedding')

    with tf.variable_scope('Attention_Layer'):
        with tf.variable_scope('item_attention'):
            item_hist_emb = tf.nn.embedding_lookup(item_embedding, features['hist_item_list'])      # batch * padded_size * emb_dim
            item_emb = tf.nn.embedding_lookup(item_embedding, features['item'])                     # batch * emb_dim
            item_att_emb = attention(item_emb, item_hist_emb, features['hist_item_list'], params)   # batch * emb_dim

        with tf.variable_scope('category_attention'):
            cate_hist_emb = tf.nn.embedding_lookup(cate_embedding, features['hist_category_list'])     # batch * padded_size * emb_dim
            cate_emb = tf.nn.embedding_lookup(cate_embedding, features['item_category'])               # batch * emd_dim
            cate_att_emb = attention(cate_emb, cate_hist_emb, features['hist_category_list'], params)  # batch * emb_dim

    # Concat attention embedding and all other features
    with tf.variable_scope('Concat_Layer'):
        fc = tf.concat([item_att_emb, cate_att_emb, item_emb, cate_emb, f_dense], axis=1)
        add_layer_summary('fc_concat', fc)
    dense = stack_dense_layer(fc, params['hidden_units'], params['dropout_rate'], params['batch_norm'], mode, add_summary = True)
    with tf.variable_scope('output'):
        y = tf.layers.dense(dense, units =1)
        add_layer_summary( 'output', y )

    return y


build_estimator = build_estimator_helper(
    model_fn = {
        'amazon' :model_fn_varlen
    },
    params = {
        'amazon':{ 'dropout_rate' : 0.2,
                   'batch_norm' : True,
                   'learning_rate' : 0.01,
                   'hidden_units' : [80,40],
                   'attention_hidden_units':[80,40],
                   'amazon_item_count': AMAZON_ITEM_COUNT,
                   'amazon_cate_count': AMAZON_CATE_COUNT,
                   'amazon_emb_dim': AMAZON_EMB_DIM
            }
    }
)
