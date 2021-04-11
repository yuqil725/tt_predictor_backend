import os
import pickle
import sys
from pprint import pprint

import numpy as np
from TTBenchmark.check_environment import check_env_info
from model_data_util.create_tt_data.model_data_convert import convertModelToRawData
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense


class TTPredictor:
    def __init__(self, predictor_tree_path="/content/drive/MyDrive/TT Predictor Tree", env=None):
        if env is None:
            self.env = check_env_info()
        else:
            self.env = env

        self.predictor_tree = {}
        self.create_predictor_tree(predictor_tree_path)

    def create_predictor_tree(self, predictor_tree_path):
        for f in os.listdir(predictor_tree_path):
            if f[0] == ".":
                continue
            if f not in self.predictor_tree.keys():
                self.predictor_tree[f] = {}
            for m_fn in os.listdir(os.path.join(predictor_tree_path, f)):
                if m_fn.split("_")[0] not in self.predictor_tree[f].keys():
                    self.predictor_tree[f][m_fn.split("_")[0]] = {}
                self.predictor_tree[f][m_fn.split("_")[0]][m_fn.split("_")[1]] = os.path.join(predictor_tree_path, f,
                                                                                              m_fn)

    def validate_env(self):
        _env_pass = True if "_".join(sorted(self.env.values())).lower() in self.predictor_tree else False
        return _env_pass

    def get_branch_info(self, model, x_shape, kwargs):
        training_size = x_shape[0]

        branch_info_df = convertModelToRawData(model, training_size, np.array((kwargs["batch_size"], *x_shape[1:])))

        return branch_info_df

    def validate_features(self, branch_info_df):
        env_str = "_".join(sorted(self.env.values())).lower()
        optimizer = branch_info_df.optimizer.values[0].lower()
        batch_size = str(branch_info_df.out_dim_0.values[0]).lower()
        if not optimizer in self.predictor_tree[env_str]:
            return "", optimizer, self.predictor_tree[env_str]
        if not batch_size in self.predictor_tree[env_str][optimizer]:
            return "", batch_size, self.predictor_tree[env_str][optimizer]
        model_path = self.predictor_tree[env_str][optimizer][batch_size]
        return model_path, None, None

    def validate_all(self, feature_df, kwargs):
        env_pass = self.validate_env()
        if not env_pass:
            print("Error: your current environment")
            pprint(self.env)
            print("is not one of the supporting environments:")
            pprint(list(self.predictor_tree.keys()))
            return ""
        tt_predictor_path, unsupport_feature, unsupport_feature_level = self.validate_features(feature_df)
        if tt_predictor_path == "":
            print("Error: model had feature")
            pprint(unsupport_feature)
            print("not in the supporting feature list:")
            pprint(unsupport_feature_level)
            return ""
        return tt_predictor_path

    def load_model(self, path):
        _tt_predictor = None
        fname = os.path.basename(path) + ".pkl"
        _tt_predictor = pickle.load(open(os.path.join(path, fname), "rb"))
        return _tt_predictor

    def predict(self, model, x_shape, kwargs=None):
        assert len(x_shape) == 2, "Warning: currently only support 2 dimension data"

        branch_info_df = self.get_branch_info(model, x_shape, kwargs)  # self.get_features are applied on all models
        tt_predictor_path = self.validate_all(branch_info_df, kwargs)
        assert tt_predictor_path != ""

        sys.path.append(tt_predictor_path)
        from get_features import get_features_func

        batch_size = branch_info_df.out_dim_0.values[0]
        features = get_features_func(model, x_shape[0], np.array([batch_size, *x_shape[
                                                                               1:]]))  # all get_feature_func for
        # tt_predictors should only take the three arguments. Other arguments are passed by default

        _tt_predictor = self.load_model(tt_predictor_path)
        return _tt_predictor.predict(features)


if __name__ == "__main__":
    model = Sequential()
    model.add(Dense(30))
    model.add(Dense(15))

    model.compile(optimizer="SGD", loss="mse", metrics=["accuracy"])

    kwargs = {}
    kwargs["batch_size"] = 4

    kwargs["padding"] = 33

    fake_env = {'cpu': 'x86_64',
                'cuda_v': '11.2',
                'drive_v': '460.32.03',
                'gpu_type': 'tesla_v100-sxm2',
                'tf_v': '2.4.1'}

    predictor_tree_path = "/Users/wangqiong/Documents/AIpaca/Code/TT Prediction/tt_predictor_backend/tt_predictor_backend/tt_predictor_tree"
    tt_predictor = TTPredictor(env=fake_env, predictor_tree_path=predictor_tree_path)
    tt_pred = tt_predictor.predict(model, np.array((1000, 5)), kwargs)
    print(f"TT Prediction = {tt_pred}")
