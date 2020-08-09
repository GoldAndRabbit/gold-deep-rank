import pickle
import shutil
from const import *
from config import *
from layers import *
from deep_seq_models.din_v2 import din_v2_model_fn

os.environ['CUDA_VISIBLE_DEVICES'] = '0'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

def seq_input_fn(data_file, is_predict, config):

    def _parse_example_helper_tfreocrd(line):
        features = tf.parse_single_example(line,features=AMAZON_PROTO)
        for i in AMAZON_VARLEN:
            features[i] = tf.sparse_tensor_to_dense(features[i])
        target = tf.reshape(tf.cast(features.pop(AMAZON_TARGET), tf.float32), [-1])
        return features, target

    def func():
        dataset = tf.data.TFRecordDataset(data_file).map(_parse_example_helper_tfreocrd, num_parallel_calls=8)
        if not is_predict:
            dataset = dataset.shuffle(MODEL_PARAMS['buffer_size']).repeat(MODEL_PARAMS['num_epochs'])
        if 'varlen' in config.input_type:
            dataset = dataset.padded_batch(batch_size=MODEL_PARAMS['batch_size'], padded_shapes=config.pad_shape)
        else:
            dataset = dataset.batch(MODEL_PARAMS['batch_size'])
        return dataset
    return func


def build_seq_estimator(model_dir):
    run_config = tf.estimator.RunConfig().replace(
        session_config=tf.ConfigProto(device_count={'GPU': 0}),
        save_summary_steps=2000,
        log_step_count_steps=2000,
        keep_checkpoint_max=3,
        save_checkpoints_steps=5000
    )
    with open('data/amazon/remap.pkl','rb') as f:
        _ = pickle.load(f)
        AMAZON_CATE_LIST = pickle.load(f)
        AMAZON_USER_COUNT, AMAZON_ITEM_COUNT, AMAZON_CATE_COUNT, _ = pickle.load(f)
    AMAZON_EMB_DIM = 64
    params_config = {
        'dropout_rate':             0.1,
        'batch_norm':               True,
        'learning_rate':            0.01,
        'hidden_units':             [256, 256, 256],
        'attention_hidden_units':   [64, 32],
        'amazon_item_count':        AMAZON_ITEM_COUNT,
        'amazon_cate_count':        AMAZON_CATE_COUNT,
        'amazon_emb_dim':           AMAZON_EMB_DIM
    }
    return tf.estimator.Estimator(model_fn=din_v2_model_fn, model_dir=model_dir, config=run_config, params=params_config)

def test():
    model = 'DIN'
    model_dir = './test/'
    config = CONFIG(model_name=model, data_name='amazon')
    shutil.rmtree(model_dir, ignore_errors=True)
    model = build_seq_estimator(model_dir)
    model.train(input_fn=seq_input_fn(data_file='./data/amazon/amazon_train.tfrecords', is_predict=False, config=config))
    results = model.evaluate(input_fn=seq_input_fn(data_file='./data/amazon/amazon_valid.tfrecords', is_predict=False, config=config))
    for key in sorted(results):
        print('%s: %s' % (key,results[key]))


if __name__ =='__main__':
    tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.INFO)
    tf.compat.v1.set_random_seed(1)
    test()

