import pprint

import numpy as np
from TTBenchmark.check_environment import check_env_info
from model_data_util.create_tt_data.model_data_convert import convertModelToRawData, preprocessRawData

from constant import PREDICTOR_TREE


class TTPredictor:
    def __init__(self, predictor_tree=PREDICTOR_TREE):
        self.env = check_env_info()
        self.predictor_tree = predictor_tree

    def validate_env(self):
        _env_pass = True if "_".join(self.env.values()).lower() in self.predictor_tree else False
        return _env_pass

    def get_features(self, model, x_shape, kwargs):
        training_size = x_shape[0]

        feature_df = convertModelToRawData(model, training_size, np.array((kwargs["batch_size"], *x_shape[1:])))

        return feature_df

    def validate_features(self, feature_df):
        env_str = "_".join(self.env.values()).lower()
        optimizer = feature_df.optimizer.values[0].lower()
        batch_size = str(feature_df.out_dim_0.values[0]).lower()
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
        tt_predictor_path, unsupport_feature, unsupport_feature_level = self.validate_features(feature_df, kwargs)
        if tt_predictor_path == "":
            print("Error: model had feature")
            pprint(unsupport_feature)
            print("not in the supporting feature list:")
            pprint(unsupport_feature_level)
            return ""
        return tt_predictor_path

    def load_model(self, path):
        _tt_predictor = None
        if path[-3:] == "pkl":
            _tt_predictor = pickle.load(open(path, "rb"))
        return _tt_predictor

    def predict(self, model, x_shape, kwargs=None):
        assert len(x_shape) == 2, "Warning: currently only support 2 dimension data"

        feature_df = self.get_features(model, x_shape, kwargs)
        tt_predictor_path = self.validate_all(feature_df, kwargs)
        assert tt_predictor_path != ""

        features = preprocessRawData(feature_df, kwargs["one_hot_enc"], kwargs["padding"]).values
        features = features.reshape((-1, *features.shape))

        _tt_predictor = self.load_model(tt_predictor_path)
        return _tt_predictor.predict(features)

# if __name__ == "__main__":
#     model = Sequential()
#     model.add(Dense(30))
#     model.add(Dense(15))
#
#     model.compile(optimizer="SGD", loss="mse", metrics=["accuracy"])
#
#     kwargs = {}
#     kwargs["batch_size"] = 4
#     kwargs["one_hot_enc"] = ONE_HOT_ENC
#     kwargs["padding"] = 33
#
#     tt_predictor = TTPredictor()
#     tt_predictor.predict(model, np.array((1000, 5)), kwargs)
