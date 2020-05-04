import shutil
import tensorflow as tf
from utils.census_feat_config import CENSUS_COLUMNS, CENSUS_COLUMN_DEFAULTS
from models.wdl import wdl_model_fn
from models.dcn import dcn_model_fn
from models.autoint import autoint_model_fn
from models.xdeepfm import xdeepfm_model_fn
from models.deepfm import deepfm_model_fn
from models.resnet import res_model_fn


def build_estimator(model_dir, model_type, paras):
    run_config = tf.estimator.RunConfig().replace(session_config=tf.ConfigProto(device_count={'GPU': 0}))
    if model_type == 'wdl':
        return tf.estimator.Estimator(model_fn=wdl_model_fn, model_dir=model_dir, config=run_config, params=paras)
    elif model_type == 'dcn':
        return tf.estimator.Estimator(model_fn=dcn_model_fn, model_dir=model_dir, config=run_config, params=paras)
    elif model_type == 'autoint':
        return tf.estimator.Estimator(model_fn=autoint_model_fn, model_dir=model_dir, config=run_config, params=paras)
    elif model_type == 'xdeepfm':
        return tf.estimator.Estimator(model_fn=xdeepfm_model_fn, model_dir=model_dir, config=run_config, params=paras)
    elif model_type == 'deepfm':
        return tf.estimator.Estimator(model_fn=deepfm_model_fn, model_dir=model_dir, config=run_config, params=paras)
    elif model_type == 'res':
        return tf.estimator.Estimator(model_fn=res_model_fn, model_dir=model_dir, config=run_config, params=paras)
    else:
        print('error')


def input_fn_from_csv_file(data_file,num_epochs,shuffle,batch_size):
    assert tf.gfile.Exists(data_file), ('no file named : ' + str(data_file))

    def parse_csv(value):
        columns = tf.decode_csv(value,record_defaults=CENSUS_COLUMN_DEFAULTS)
        features = dict(zip(CENSUS_COLUMNS, columns))
        labels = tf.equal(features.pop('income_bracket'), '>50K')
        labels = tf.reshape(labels, [-1])
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
    assert tf.gfile.Exists(data_file), ('no file named : ' + str(data_file))

    def _parse_census_TFRecords_fn(record):
        features = {
            # int
            'age': tf.io.FixedLenFeature([], tf.int64),
            'fnlwgt': tf.io.FixedLenFeature([], tf.int64),
            'education_num': tf.io.FixedLenFeature([], tf.int64),
            'capital_gain': tf.io.FixedLenFeature([], tf.int64),
            'capital_loss': tf.io.FixedLenFeature([], tf.int64),
            'hours_per_week': tf.io.FixedLenFeature([], tf.int64),

            # string
            'gender': tf.io.FixedLenFeature([], tf.string),
            'education': tf.io.FixedLenFeature([], tf.string),
            'marital_status': tf.io.FixedLenFeature([], tf.string),
            'relationship': tf.io.FixedLenFeature([], tf.string),
            'race': tf.io.FixedLenFeature([], tf.string),
            'workclass': tf.io.FixedLenFeature([], tf.string),
            'native_country': tf.io.FixedLenFeature([], tf.string),
            'occupation': tf.io.FixedLenFeature([], tf.string),
            'income_bracket': tf.io.FixedLenFeature([], tf.string),
        }
        features = tf.io.parse_single_example(record, features)
        labels = tf.equal(features.pop('income_bracket'), '>50K')
        labels = tf.reshape(labels, [-1])
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
    all_paras = {
        'model_dir': './ckpt_dir/',
        'model_type': 'xdeepfm',
        'train_epoches': 8,
        'epoches_per_eval': 2,
        'train_data': 'toy_data/adult.data',
        'test_data': 'toy_data/adult.test',
        'train_data_tfrecords_dir': 'toy_data/census_adult.tfrecords',
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


def train_census_data_from_tfrecords():
    all_paras = {
        'model_dir': './ckpt_dir/',
        'model_type': 'dcn',
        'train_epoches': 8,
        'epoches_per_eval': 2,
        'train_data': 'toy_data/adult.data',
        'test_data': 'toy_data/adult.test',
        'train_data_tfrecords_dir': 'toy_data/census_adult.tfrecords',
        'test_data_tfrecords_dir': 'toy_data/census_test.tfrecords',
        'batch_size': 8,
    }
    print('using: ' +  all_paras['model_type'] + ' model...')
    shutil.rmtree(all_paras['model_dir'], ignore_errors=True)
    model = build_estimator(all_paras['model_dir'], all_paras['model_type'], paras=all_paras)
    model.train(
        input_fn=lambda: input_fn_from_tfrecords(
            data_file=all_paras['train_data_tfrecords_dir'],
            num_epochs=all_paras['epoches_per_eval'],
            shuffle=True,
            batch_size=all_paras['batch_size']
        )
    )
    results = model.evaluate(
        input_fn=lambda: input_fn_from_tfrecords(
            data_file=all_paras['test_data_tfrecords_dir'],
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
    train_census_data_from_tfrecords()

