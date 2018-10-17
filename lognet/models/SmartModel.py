import os
import numpy as np
import xgboost as xgb
from tqdm import tqdm

from sklearn.ensemble import RandomForestRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.externals import joblib
from sklearn.metrics import confusion_matrix
from sklearn.metrics import mean_squared_error
from torch.utils import data
from pygam import LinearGAM


class SmartModel():
    """
    Super class for all models - DeepLearning, Random Forest, Gradient Boost
    """

    def __init__(self, **kwargs):
        pass

    def _regression_error(self, X_valid, y_valid, model):
        y_pred = model.predict(X_valid.values)
        return mean_squared_error(y_valid.values, y_pred)

    def _classification_error(self, X_valid, y_valid, model):
        y_pred = model.predict(X_valid.values)
        conf_mat = confusion_matrix(y_valid.values, y_pred, labels=np.arange(self.num_classes))
        return conf_mat

    def train(self, training_data, validation_data, features, target):
        type_name = self.__class__.__name__
        model_name = type_name + ''.join(tt for tt in target) + 'x' + ''.join(ff for ff in features) + '.pkl'
        print(f'{model_name} training.')

        X_train = training_data[features]
        X_train = X_train.sample(frac=1)
        y_train = training_data[target]
        X_valid = validation_data[features]
        y_valid = validation_data[target]

        trained_model = self.model.fit(X_train.values, y_train.values)
        joblib.dump(trained_model, os.path.join(self.model_path, model_name))

        if type_name.endswith('Class'):
             err = self._classification_error(X_valid, y_valid, trained_model)
        else:
             err = self._regression_error(X_valid, y_valid, trained_model)
        return err

    def predict(self, test_data, features, target):
        type_name = self.__class__.__name__
        model_name = type_name + ''.join(tt for tt in target) + 'x' + ''.join(ff for ff in features) + '.pkl'
        print(f'Predicting with {model_name}.')
        trained_model = joblib.load(os.path.join(self.model_path, model_name))
        
        X_test = test_data[features]
        y_pred = trained_model.predict(X_test.values)
        return y_pred


class RandomForest(SmartModel):

    def __init__(self, model_path, **kwargs):
        super().__init__()
        print('Using RandomForest model.')
        self.model_params = {'n_estimators': 800, 'max_depth': 9, 'max_features': 3, 
                             'n_jobs': -1, 'oob_score': True, 'random_state': 0}
        self.model_path = model_path
        if kwargs:
            for kw in kwargs:
                self.model_params[kw] = kwargs[kw]
        self.model = RandomForestRegressor(**self.model_params)


class GradientBoost(SmartModel):

    def __init__(self, model_path, **kwargs):
        super().__init__()
        print('Using GradientBoost model.')
        self.model_params = {'n_estimators': 900, 'learning_rate': 0.03, 'gamma': 0,
                             'subsample': 0.75, 'colsample_bytree': 1.0, 'max_depth': 9, 'n_jobs': -1}
        self.model_path = model_path
        if kwargs:
            for kw in kwargs:
                self.model_params[kw] = kwargs[kw]
        self.model = xgb.XGBRegressor(**self.model_params)

    def train(self, training_data, validation_data, features, target):
        type_name = self.__class__.__name__
        model_name = type_name + ''.join(tt for tt in target) + 'x' + ''.join(ff for ff in features) + '.pkl'
        print(f'{model_name} training.')

        dtrain = xgb.DMatrix(np.asarray(training_data[features].sample(frac=1)), label=np.asarray(training_data[target]))
        dvalid = xgb.DMatrix(np.asarray(validation_data[features]), label=np.asarray(validation_data[target]))
        params = self.model.get_xgb_params()
        progress = dict()
        watchlist = [(dtrain, 'train'), (dvalid, 'eval')]

        trained_model = xgb.train(params, dtrain, num_boost_round=1000, evals=watchlist, early_stopping_rounds=20, 
                                  evals_result=progress, callbacks=[xgb.callback.print_evaluation()])
        joblib.dump(trained_model, os.path.join(self.model_path, model_name))
        return progress


class MultilayerPerceptron(SmartModel):

    def __init__(self, model_path, **kwargs):
        super().__init__()
        print('Using MultilayerPerceptron model.')
        self.model_params = {'activation': 'relu', 'alpha': 0.00001, 'batch_size': 'auto',
                             'beta_1': 0.9, 'beta_2': 0.999, 'early_stopping': False, 'epsilon': 1e-10,
                             'hidden_layer_sizes': (100,), 'learning_rate': 'constant', 
                             'learning_rate_init': 0.0001, 'max_iter': 1000, 'momentum': 0.9, 
                             'nesterovs_momentum': True, 'power_t': 0.5, 'random_state': 42, 'shuffle': True,
                             'solver': 'adam', 'tol': 0.0001, 'validation_fraction': 0.1, 'verbose': False, 
                             'warm_start': False}
        self.model_path = model_path
        if kwargs:
            for kw in kwargs:
                self.model_params[kw] = kwargs[kw]
        self.model = MLPRegressor(**self.model_params)


class GeneralizedAdditive(SmartModel):

    def __init__(self, model_path, **kwargs):
        super().__init__()
        print('Using GeneralizedAdditive model.')
        self.model_params = {'n_splines': 25}
        self.model_path = model_path
        if kwargs:
            for kw in kwargs:
                self.model_params[kw] = kwargs[kw]
        self.model = LinearGAM(**self.model_params)


class GradientBoostClass(SmartModel):

    def __init__(self, model_path, **kwargs):
        super().__init__()
        print('Using GradientBoost Classifier model.')
        self.model_params = {'n_estimators': 900, 'learning_rate': 0.03, 'gamma': 0,
                             'subsample': 0.75, 'colsample_bytree': 1.0, 'max_depth': 9, 'n_jobs': -1}
        self.model_path = model_path
        self.num_classes = kwargs.pop('num_classes', 2)
        if kwargs:
            for kw in kwargs:
                self.model_params[kw] = kwargs[kw]
        self.model = xgb.XGBClassifier(**self.model_params)