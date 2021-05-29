import os
import pandas as pd
from sklearn.pipeline import Pipeline
import numba
import numpy as np

import json, joblib

# The feature importances for each data instance 
# are calculated with four random samples drawn from the data
# I also tried to use numba to compile to machine code but could
# not achieve it

file_path = '.'

model = joblib.load(os.path.join(file_path, 'model_pipeline.joblib'))

X_train = pd.read_csv(os.path.join(file_path, 'train.csv'), index_col=0)
X_test = pd.read_csv(os.path.join(file_path, 'test.csv'), index_col=0)


# TODO try to complete numba usage
# @numba.jit(nopython=True)
def shuffle_feature_records(full_data: np.array, feature_data: np.array, feature_index: int, m: int):
    feat_len = len(feature_data)
    feature_indices = np.arange(0, feat_len, dtype=np.uint8)
    # take random indices
    rand_indices = np.random.choice(full_data.shape[0], m, replace=False)

    # take random samples
    w = full_data[rand_indices]

    b1_vec: np.array = np.empty((m, feat_len), dtype=np.object)
    b2_vec: np.array = np.empty((m, feat_len), dtype=np.object)

    for i in np.arange(0, m):

        del_index = np.argwhere(feature_indices==feature_index)

        indices = np.delete(feature_indices, del_index)
        # remove current feature index to add later
        np.random.shuffle(indices)

        split_pt = indices.shape[0]//2

        # split indices for 2 different vectors
        ind_x, ind_w = indices[:split_pt], indices[split_pt:]

        # form the indices for b1 and b2 vectors
        ind_x_1 = np.append(ind_x, feature_index)
        ind_w_1 = np.append(ind_w, feature_index)

        # to ensure the features are assigned in the order of training data
        b1_sort_indices = np.argsort(np.append(ind_x, ind_w_1))
        b2_sort_indices = np.argsort(np.append(ind_x_1, ind_w))

        b1_vec[i] = np.concatenate((feature_data[ind_x], w[i, ind_w_1]), axis=0)[b1_sort_indices]
        b2_vec[i] = np.concatenate((feature_data[ind_x_1], w[i, ind_w]), axis=0)[b2_sort_indices]

    arrs = [b1_vec, b2_vec]

    return arrs

def get_feat_imp_data(model: Pipeline, X_train: pd.DataFrame, X_test: pd.DataFrame):
    tr_data = X_train[X_train['id'].isin(range(0, 101))].copy()
    te_data = X_test[X_test['id'].isin(range(0, 101))].copy()

    # join all the data
    data = pd.concat([tr_data, te_data], axis='rows')

    cols = data.columns.tolist()

    feat_imp = {}
    for d in data.values:
        # get id no of the data instance
        col_imp = {}
        for feat_index in range(data.shape[1]):
            b1, b2 = shuffle_feature_records(X_train.values, d, feature_index=feat_index, m=4)
            b1 = pd.DataFrame(b1, columns=cols)
            b2 = pd.DataFrame(b2, columns=cols)
            b1_pred = model.predict(b1)
            b2_pred = model.predict(b2)
            col_imp[cols[feat_index]] = np.sum(b1_pred - b2_pred) / 4

        feat_imp[d[0]] = col_imp

    return feat_imp


feats = get_feat_imp_data(model, X_train, X_test)

with open('feat_imps_1.json', 'w') as f:
    json.dump(feats, f)