import shutil
import tensorflow as tf
from utils.census_ctr_feat_config import CENSUS_COLUMNS, CENSUS_COLUMN_DEFAULTS
from deep_ctr_models.wdl import wdl_model_fn
from deep_ctr_models.dcn import dcn_model_fn
from deep_ctr_models.autoint import autoint_model_fn
from deep_ctr_models.xdeepfm import xdeepfm_model_fn
from deep_ctr_models.deepfm import deepfm_model_fn
from deep_ctr_models.resnet import res_model_fn


def build_estimator(ckpt_dir, model_name, paras):
    MODEL_FN_MAP = {
        'wdl':          wdl_model_fn,
        'dcn':          dcn_model_fn,
        'autoint':      autoint_model_fn,
        'xdeepfm':      xdeepfm_model_fn,
        'deepfm':       deepfm_model_fn,
        'res_model_fn': res_model_fn,
    }
    assert model_name in MODEL_FN_MAP.keys(), ('no model named : ' + str(model_name))
    run_config = tf.estimator.RunConfig().replace(session_config=tf.ConfigProto(device_count={'GPU': 0}))
    return tf.estimator.Estimator(model_fn=MODEL_FN_MAP[model_name],
                                  model_dir=ckpt_dir,
                                  config=run_config,
                                  params=paras)


def input_fn_from_csv_file(data_file, num_epochs, shuffle, batch_size):
    assert tf.io.gfile.exists(data_file), ('no file named : ' + str(data_file))

    def parse_csv(value):
        columns = tf.decode_csv(value,record_defaults=CENSUS_COLUMN_DEFAULTS)
        features = dict(zip(CENSUS_COLUMNS, columns))
        labels = tf.equal(features.pop('income_bracket'), '>50K')
        labels = tf.reshape(labels, [-1])
        labels = tf.to_float(labels)
        return features, labels

    dataset = tf.data.TextLineDataset(data_file)
    if shuffle:
        dataset = dataset.shuffle(buffer_size=5000)
    dataset = dataset.map(parse_csv, num_parallel_calls=5)
    dataset = dataset.repeat(num_epochs)
    dataset = dataset.batch(batch_size)
    iterator = dataset.make_one_shot_iterator()
    features, labels = iterator.get_next()
    return features, labels


def input_fn_from_tfrecords(data_file, num_epochs, shuffle, batch_size):
    assert tf.gfile.Exists(data_file), ('no file named: ' + str(data_file))

    def _parse_census_TFRecords_fn(record):
        features = {
            # int
            'age':            tf.io.FixedLenFeature([], tf.int64),
            'fnlwgt':         tf.io.FixedLenFeature([], tf.int64),
            'education_num':  tf.io.FixedLenFeature([], tf.int64),
            'capital_gain':   tf.io.FixedLenFeature([], tf.int64),
            'capital_loss':   tf.io.FixedLenFeature([], tf.int64),
            'hours_per_week': tf.io.FixedLenFeature([], tf.int64),
            # string
            'gender':         tf.io.FixedLenFeature([], tf.string),
            'education':      tf.io.FixedLenFeature([], tf.string),
            'marital_status': tf.io.FixedLenFeature([], tf.string),
            'relationship':   tf.io.FixedLenFeature([], tf.string),
            'race':           tf.io.FixedLenFeature([], tf.string),
            'workclass':      tf.io.FixedLenFeature([], tf.string),
            'native_country': tf.io.FixedLenFeature([], tf.string),
            'occupation':     tf.io.FixedLenFeature([], tf.string),
            'income_bracket': tf.io.FixedLenFeature([], tf.string),
        }
        features = tf.io.parse_single_example(record, features)
        labels = tf.equal(features.pop('income_bracket'), '>50K')
        labels = tf.reshape(labels, [-1])
        labels = tf.to_float(labels)
        return features, labels

    dataset = tf.data.TFRecordDataset(data_file).map(_parse_census_TFRecords_fn,
        num_parallel_calls=10)

    if shuffle:
        dataset = dataset.shuffle(buffer_size=5000)
    dataset = dataset.repeat(num_epochs)
    dataset = dataset.batch(batch_size)
    iterator = dataset.make_one_shot_iterator()
    features, labels = iterator.get_next()
    return features, labels


def train_census_data():
    data_process_config = {
        'train_data_dir': 'toy_data/adult.data',
        'test_data_dir': 'toy_data/adult.test',
        'ckpt_dir': './ckpt_dir/',
    }
    train_prosess_config = {
        'model_name': 'dcn',
        'train_epoches': 8,
        'batch_size': 8,
        'epoches_per_eval': 2,
        'learning_rate': 0.01,
        'optimizer': 'adam',
        'shuffle': True
    }
    params_config = {}
    params_config.update(data_process_config)
    params_config.update(train_prosess_config)
    print('using: ' +  params_config['model_name'] + ' model...')
    shutil.rmtree(params_config['ckpt_dir'], ignore_errors=True)
    model = build_estimator(params_config['ckpt_dir'], params_config['model_name'], paras=params_config)
    model.train(
        input_fn=lambda: input_fn_from_csv_file(
            data_file=params_config['train_data_dir'],
            num_epochs=params_config['epoches_per_eval'],
            shuffle=True if params_config['shuffle']==True else False,
            batch_size=params_config['batch_size']
        )
    )
    results = model.evaluate(
        input_fn=lambda: input_fn_from_csv_file(
            data_file=params_config['test_data_dir'],
            num_epochs=1,
            shuffle=False,
            batch_size=params_config['batch_size']
        )
    )
    for key in sorted(results):
        print('%s: %s' % (key, results[key]))


def train_census_data_from_tfrecords():
    data_process_config = {
        'train_data_tfrecords_dir': 'toy_data/census_adult.tfrecords',
        'test_data_tfrecords_dir': 'toy_data/census_test.tfrecords',
        'ckpt_dir': './ckpt_dir/',
    }
    train_prosess_config = {
        'model_name': 'dcn',
        'train_epoches': 8,
        'batch_size': 8,
        'epoches_per_eval': 2,
        'learning_rate': 0.01,
        'optimizer': 'adam',
        'shuffle': True
    }
    params_config = {}
    params_config.update(data_process_config)
    params_config.update(train_prosess_config)
    print('using: ' +  params_config['model_name'] + ' model...')
    shutil.rmtree(params_config['ckpt_dir'], ignore_errors=True)
    model = build_estimator(params_config['ckpt_dir'], params_config['model_name'], paras=params_config)
    model.train(
        input_fn=lambda: input_fn_from_tfrecords(
            data_file=params_config['train_data_tfrecords_dir'],
            num_epochs=params_config['epoches_per_eval'],
            shuffle=True,
            batch_size=params_config['batch_size']
        )
    )
    results = model.evaluate(
        input_fn=lambda: input_fn_from_tfrecords(
            data_file=params_config['test_data_tfrecords_dir'],
            num_epochs=1,
            shuffle=False,
            batch_size=params_config['batch_size']
        )
    )
    for key in sorted(results):
        print('%s: %s' % (key, results[key]))


if __name__ == '__main__':
    tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.INFO)
    tf.compat.v1.set_random_seed(1)
    # train_census_data()
    train_census_data_from_tfrecords()

