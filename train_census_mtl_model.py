import shutil
import tensorflow as tf
from deep_mtl_models.essm import essm_model_fn
from deep_mtl_models.mmoe import mmoe_model_fn

from feat_config.census_mtl_feat_config import CENSUS_COLUMNS, CENSUS_COLUMN_DEFAULTS

def build_estimator(model_dir, model_type, paras):
    MODEL_FN_MAP = {
        'essm': essm_model_fn,
        'mmoe': mmoe_model_fn,
    }
    assert model_type in MODEL_FN_MAP.keys(), ('no model named : ' + str(model_type))
    run_config = tf.estimator.RunConfig().replace(session_config=tf.ConfigProto(device_count={'GPU': 0}))
    return tf.estimator.Estimator(model_fn=MODEL_FN_MAP[model_type],
                                  model_dir=model_dir,
                                  config=run_config,
                                  params=paras)


def input_fn_from_csv_file(data_file, num_epochs, shuffle, batch_size):
    assert tf.gfile.Exists(data_file), ('no file named : ' + str(data_file))

    def parse_csv(value):
        columns = tf.decode_csv(value,record_defaults=CENSUS_COLUMN_DEFAULTS)
        features = dict(zip(CENSUS_COLUMNS, columns))
        clk = tf.equal(features.pop('income_bracket'), '>50K')
        clk = tf.reshape(clk, [-1])
        pay = tf.equal(features.pop('marital_status'), 'Never-married')
        pay = tf.reshape(pay, [-1])
        return features, {'ctr': tf.to_float(clk), 'cvr': tf.to_float(pay)}

    dataset = tf.data.TextLineDataset(data_file)
    if shuffle:
        dataset = dataset.shuffle(buffer_size=5000)
    dataset = dataset.map(parse_csv, num_parallel_calls=5)
    dataset = dataset.repeat(num_epochs)
    dataset = dataset.batch(batch_size)
    iterator = dataset.make_one_shot_iterator()
    features, labels = iterator.get_next()
    return features, labels


def train_essm_census_data():
    all_paras = {
        'model_dir': './ckpt_dir/',
        'model_type': 'essm',
        'train_epoches': 2,
        'epoches_per_eval': 2,
        'train_data': 'census_data/adult.data',
        'test_data': 'census_data/adult.test',
        'train_data_tfrecords_dir': 'census_data/census_adult.tfrecords',
        'batch_size': 8,
    }
    print('using: ' +  all_paras['model_type'] + ' model...')
    shutil.rmtree(all_paras['model_dir'], ignore_errors=True)
    model = build_estimator(all_paras['model_dir'], all_paras['model_type'], paras=all_paras)
    model.train(
        input_fn=lambda: input_fn_from_csv_file(
            data_file=all_paras['train_data'],
            num_epochs=all_paras['epoches_per_eval'],
            shuffle=True,
            batch_size=all_paras['batch_size']
        )
    )
    results = model.evaluate(
        input_fn=lambda: input_fn_from_csv_file(
            data_file=all_paras['test_data'],
            num_epochs=1,
            shuffle=False,
            batch_size=all_paras['batch_size']
        )
    )
    for key in sorted(results):
        print('%s: %s' % (key, results[key]))


def train_mmoe_census_data():
    all_paras = {
        'model_dir': './ckpt_dir/',
        'model_type': 'mmoe',
        'train_epoches': 2,
        'epoches_per_eval': 2,
        'train_data': 'census_data/adult.data',
        'test_data': 'census_data/adult.test',
        'batch_size': 8,
    }
    print('using: ' +  all_paras['model_type'] + ' model...')
    shutil.rmtree(all_paras['model_dir'], ignore_errors=True)
    model = build_estimator(all_paras['model_dir'], all_paras['model_type'], paras=all_paras)
    model.train(
        input_fn=lambda: input_fn_from_csv_file(
            data_file=all_paras['train_data'],
            num_epochs=all_paras['epoches_per_eval'],
            shuffle=True,
            batch_size=all_paras['batch_size']
        )
    )
    results = model.evaluate(
        input_fn=lambda: input_fn_from_csv_file(
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
    # train_essm_census_data()
    train_mmoe_census_data()

