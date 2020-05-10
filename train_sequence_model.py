import shutil
import tensorflow as tf
from utils.census_ctr_feat_config import SEQ_COLUMNS, SEQ_COLUMNS_DEFAULTS
from deep_ctr_models.din import din_model_fn

def build_estimator(model_dir, model_type):
    run_config = tf.estimator.RunConfig().replace(session_config=tf.ConfigProto(device_count={'GPU': 0}))
    if model_type == 'din':
        return tf.estimator.Estimator(model_fn=din_model_fn, model_dir=model_dir, config=run_config)
    else:
        print('error')

def sequence_input_fn(data_file, num_epochs, shuffle, batch_size):
    assert tf.gfile.Exists(data_file), ('no file named : ' + str(data_file))

    def process_list_column(list_column):
        sparse_strings = tf.string_split(list_column, delimiter="|")
        return sparse_strings.values

    def parse_csv(value):
        columns = tf.decode_csv(value, record_defaults=SEQ_COLUMNS_DEFAULTS)
        features = dict(zip(SEQ_COLUMNS, columns))
        features['seq'] = process_list_column([features['seq']])
        features['seq_cate'] = process_list_column([features['seq_cate']])
        labels = tf.equal(features.pop('label'), '1')
        labels = tf.reshape(labels, [-1])
        return features, labels

    dataset = tf.data.TextLineDataset(data_file).skip(1)
    if shuffle:
        dataset = dataset.shuffle(buffer_size=5000)
    dataset = dataset.map(parse_csv, num_parallel_calls=5)
    dataset = dataset.repeat(num_epochs)
    dataset = dataset.batch(batch_size)
    iterator = dataset.make_one_shot_iterator()
    features, labels = iterator.get_next()
    return features, labels


def train_sequence_data():
    all_paras = {
        'model_dir': './seq_ckpt_dir/',
        'model_type': 'din',
        'train_epoches': 2,
        'epoches_per_eval': 1,
        'train_data': 'toy_data/ama_ele_train.csv',
        'test_data': 'toy_data/ama_ele_test.csv',
        'batch_size': 64,
    }
    print('using: ' +  all_paras['model_type'] + ' model...')
    shutil.rmtree(all_paras['model_dir'], ignore_errors=True)
    model = build_estimator(all_paras['model_dir'], all_paras['model_type'])
    model.train(
        input_fn=lambda: sequence_input_fn(
            data_file=all_paras['train_data'],
            num_epochs=all_paras['epoches_per_eval'],
            shuffle=True,
            batch_size=all_paras['batch_size']
        )
    )
    results = model.evaluate(
        input_fn=lambda: sequence_input_fn(
            data_file=all_paras['test_data'],
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

