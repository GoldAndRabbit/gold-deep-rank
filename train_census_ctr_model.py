import os
import shutil
import tensorflow as tf
from utils.census_ctr_feat_config import build_census_feat_columns
from utils.census_ctr_feat_config import CENSUS_COLUMNS, CENSUS_COLUMN_DEFAULTS
from deep_ctr_models.wdl          import wdl_model_fn
from deep_ctr_models.dcn          import dcn_model_fn
from deep_ctr_models.autoint      import autoint_model_fn
from deep_ctr_models.xdeepfm      import xdeepfm_model_fn
from deep_ctr_models.deepfm       import deepfm_model_fn
from deep_ctr_models.resnet       import res_model_fn
from deep_ctr_models.fibinet      import fibinet_model_fn
from deep_ctr_models.afm          import afm_model_fn
from deep_ctr_models.pnn          import pnn_model_fn

os.environ['CUDA_VISIBLE_DEVICES'] = '0'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

def census_input_fn_from_csv_file(data_file, num_epochs, shuffle, batch_size):
    assert tf.io.gfile.exists(data_file), ('no file named : ' + str(data_file))

    def parse_csv(value):
        columns = tf.decode_csv(value, record_defaults=CENSUS_COLUMN_DEFAULTS)
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


def census_input_fn_from_tfrecords(data_file, num_epochs, shuffle, batch_size):
    assert tf.io.gfile.exists(data_file), ('no file named: ' + str(data_file))

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

    dataset = tf.data.TFRecordDataset(data_file).map(_parse_census_TFRecords_fn, num_parallel_calls=10)
    if shuffle:
        dataset = dataset.shuffle(buffer_size=5000)
    dataset = dataset.repeat(num_epochs)
    dataset = dataset.batch(batch_size)
    iterator = dataset.make_one_shot_iterator()
    features, labels = iterator.get_next()
    return features, labels


def build_estimator(ckpt_dir, model_name, params_config):
    model_fn_map = params_config['model_fn_map']
    assert model_name in model_fn_map.keys(), ('no model named : ' + str(model_name))
    run_config = tf.estimator.RunConfig().replace(session_config=tf.ConfigProto(device_count={'GPU': 0}))
    return tf.estimator.Estimator(
        model_fn=model_fn_map[model_name],
        model_dir=ckpt_dir,
        config=run_config,
        params=params_config
    )


def train_census_data():
    feat_columns = build_census_feat_columns(emb_dim=8)
    CENSUS_PATH = '/media/psdz/hdd/Download/Census/'
    MODEL_FN_MAP = {
        'wdl':      wdl_model_fn,
        'dcn':      dcn_model_fn,
        'autoint':  autoint_model_fn,
        'xdeepfm':  xdeepfm_model_fn,
        'deepfm':   deepfm_model_fn,
        'resnet':   res_model_fn,
        'pnn':      pnn_model_fn,
        'fibinet':  fibinet_model_fn,
        'afm':      afm_model_fn,
    }
    ARGS = {
        # data/ckpt dir config
        'train_data_dir':           'toy_data/adult.data',
        'test_data_dir':            'toy_data/adult.test',
        'train_data_tfrecords_dir': 'toy_data/census_adult.tfrecords',
        'test_data_tfrecords_dir':  'toy_data/census_test.tfrecords',
        'load_tf_records_data':     False,
        'ckpt_dir':                 CENSUS_PATH + 'ckpt_dir/',
        # traning process config
        'shuffle':                  True,
        'model_name':               'fibinet',
        'optimizer':                'adam',
        'train_epoches':            1,
        'batch_size':               16,
        'epoches_per_eval':         2,
        'learning_rate':            0.01,
        'deep_layer_nerouns':       [256, 128, 64],
        'embedding_dim':            feat_columns['embedding_dim'],
        'deep_columns':             feat_columns['deep_columns'],
        'deep_fields_size':         feat_columns['deep_fields_size'],
        'wide_columns':             feat_columns['wide_columns'],
        'wide_fields_size':         feat_columns['wide_fields_size'],
        'model_fn_map':             MODEL_FN_MAP,
        'fibinet':                  {'pooling': 'max', 'reduction_ratio': 2}
    }
    print('this process will train a: ' + ARGS['model_name'] + ' model...')
    shutil.rmtree(ARGS['ckpt_dir'], ignore_errors=True)
    model = build_estimator(ARGS['ckpt_dir'], ARGS['model_name'], params_config=ARGS)
    if not ARGS.get('load_tf_records_data'):
        model.train(
            input_fn=lambda: census_input_fn_from_csv_file(
                data_file=ARGS['train_data_dir'],
                num_epochs=ARGS['epoches_per_eval'],
                shuffle=True if ARGS['shuffle']==True else False,
                batch_size=ARGS['batch_size']
            )
        )
        results = model.evaluate(
            input_fn=lambda: census_input_fn_from_csv_file(
                data_file=ARGS['test_data_dir'],
                num_epochs=1,
                shuffle=False,
                batch_size=ARGS['batch_size']
            )
        )
        for key in sorted(results):
            print('%s: %s' % (key, results[key]))
    else:
        model.train(
            input_fn=lambda: census_input_fn_from_tfrecords(
                data_file=ARGS['train_data_tfrecords_dir'],
                num_epochs=ARGS['epoches_per_eval'],
                shuffle=True,
                batch_size=ARGS['batch_size']
            )
        )
        results = model.evaluate(
            input_fn=lambda: census_input_fn_from_tfrecords(
                data_file=ARGS['test_data_tfrecords_dir'],
                num_epochs=1,
                shuffle=False,
                batch_size=ARGS['batch_size']
            )
        )
        for key in sorted(results):
            print('%s: %s' % (key,results[key]))
    # predictions = model.predict(
    #     input_fn=lambda: census_input_fn_from_tfrecords(
    #         data_file=ARGS['test_data_tfrecords_dir'],
    #         num_epochs=1,
    #         shuffle=False,
    #         batch_size=ARGS['batch_size']
    #     )
    # )
    # for x in predictions:
    #     print(x['probabilities'][0])
    #     print(x['label'][0])


if __name__ == '__main__':
    tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.INFO)
    tf.compat.v1.set_random_seed(1)
    train_census_data()

