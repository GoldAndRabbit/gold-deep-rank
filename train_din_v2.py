import shutil
import pickle
import tensorflow as tf
from deep_seq_models.din_v2 import din_v2_model_fn

def build_estimator(model_dir, model_type):
    run_config = tf.estimator.RunConfig().replace(session_config=tf.ConfigProto(device_count={'GPU': 0}))
    with open('data/amazon/remap.pkl','rb') as f:
        _ = pickle.load(f)
        AMAZON_CATE_LIST = pickle.load(f)
        AMAZON_USER_COUNT, AMAZON_ITEM_COUNT, AMAZON_CATE_COUNT, _ = pickle.load(f)
    AMAZON_EMB_DIM = 64
    params_config = {
        'dropout_rate' : 0.2,
        'batch_norm' : True,
        'learning_rate' : 0.01,
        'hidden_units' : [80,40],
        'attention_hidden_units':[80,40],
        'amazon_item_count': AMAZON_ITEM_COUNT,
        'amazon_cate_count': AMAZON_CATE_COUNT,
        'amazon_emb_dim': AMAZON_EMB_DIM
    }
    if model_type == 'din':
        return tf.estimator.Estimator(model_fn=din_v2_model_fn, model_dir=model_dir, config=run_config, params=params_config)
    else:
        print('error')

def sequence_input_fn_v0(data_file, num_epochs, shuffle, batch_size):
    assert tf.io.gfile.exists(data_file), ('no file named: ' + str(data_file))
    AMAZON_PROTO = {
        'reviewer_id':          tf.FixedLenFeature([],tf.int64),
        'hist_item_list':       tf.VarLenFeature(tf.int64),
        'hist_category_list':   tf.VarLenFeature(tf.int64),
        'hist_length':          tf.FixedLenFeature([],tf.int64),
        'item':                 tf.FixedLenFeature([],tf.int64),
        'item_category':        tf.FixedLenFeature([],tf.int64),
        'target':               tf.FixedLenFeature([],tf.int64)
    }
    AMAZON_TARGET = 'target'
    AMAZON_VARLEN = ['hist_item_list', 'hist_category_list']
    def parse_example_helper_tfreocrd(line):
        features = tf.parse_single_example(line,features=AMAZON_PROTO)
        for i in AMAZON_VARLEN:
            features[i] = tf.sparse_tensor_to_dense(features[i])
        target = tf.reshape(tf.cast(features.pop(AMAZON_TARGET), tf.float32), [-1])
        return features, target
    dataset = tf.data.TFRecordDataset(data_file).map(parse_example_helper_tfreocrd, num_parallel_calls=8)
    return dataset

def train_sequence_data():
    all_paras = {
        'model_dir': './seq_ckpt_dir/',
        'model_type': 'din',
        'train_epoches': 1,
        'epoches_per_eval': 1,
        'train_tfrecords_data': 'data/amazon/amazon_train.tfrecords',
        'test_tfrecords_data': 'data/amazon/amazon_valid.tfrecords',
        'batch_size': 8,
    }
    print('using: ' +  all_paras['model_type'] + ' model...')
    shutil.rmtree(all_paras['model_dir'], ignore_errors=True)
    model = build_estimator(all_paras['model_dir'], all_paras['model_type'])
    model.train(
        input_fn=lambda: sequence_input_fn_v0(
            data_file=all_paras['train_tfrecords_data'],
            num_epochs=all_paras['epoches_per_eval'],
            shuffle=True,
            batch_size=all_paras['batch_size']
        )
    )
    results = model.evaluate(
        input_fn=lambda: sequence_input_fn_v0(
            data_file=all_paras['test_tfrecords_data'],
            num_epochs=1,
            shuffle=False,
            batch_size=all_paras['batch_size']
        )
    )
    for key in sorted(results):
        print('%s: %s' % (key, results[key]))


if __name__ == '__main__':
    tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.INFO)
    tf.compat.v1.set_random_seed(1)
    train_sequence_data()

