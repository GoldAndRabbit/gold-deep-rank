import tensorflow.feature_column as fc

SEQ_COLUMNS = ['user_id', 'seq', 'item_id', 'seq_cate', 'item_cate', 'label']
SEQ_COLUMNS_DEFAULTS = [[''], [''], [''], [''], [''], ['0']]


def build_ama_ele_columns():
    feature_columns = [
        fc.embedding_column(fc.categorical_column_with_hash_bucket('user_id', hash_bucket_size=200000), dimension=32),
        fc.embedding_column(fc.categorical_column_with_hash_bucket('item_id', hash_bucket_size=1000),   dimension=32),
        # fc.embedding_column(fc.categorical_column_with_hash_bucket('seq',     hash_bucket_size=200000),dimension=32),
        # fc.embedding_column(fc.categorical_column_with_hash_bucket('seq_cate',hash_bucket_size=200000),dimension=32),
    ]
    feat_field_size = len(feature_columns)
    return feature_columns, feat_field_size



