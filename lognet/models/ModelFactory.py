from lognet.models.SmartModel import TorchNN, RandomForest, GradientBoost, MultilayerPerceptron, GeneralizedAdditive, GradientBoostClass


class ModelFactory():
    """
    Factory class to select models - DeepLearning, Random Forest, Gradient Boost

    """
    @classmethod
    def select(cls, model_type, model_path, num_classes=0):
        if model_type == 'TorchNN':
            kwargs = {'batch_size': 64, 'linear_layer_sizes': [
                64], 'linear_layer_dropouts': [0.5], 'learning_rate': 0.01, 'n_epoch': 10}
            return TorchNN(model_path, **kwargs)
        if model_type == 'RF':
            kwargs = {'n_estimators': 800, 'max_depth': 9, 'max_features': 3}
            return RandomForest(model_path, **kwargs)
        if model_type == 'XGB':
            kwargs = {'n_estimators': 900, 'learning_rate': 0.03, 'gamma': 0,
                      'subsample': 0.75, 'colsample_bytree': 1.0, 'max_depth': 9}
            return GradientBoost(model_path, **kwargs)
        if model_type == 'MLP':
            kwargs = {'alpha': 0.00001, 'hidden_layer_sizes': 100,
                      'learning_rate_init': 0.0001}
            return MultilayerPerceptron(model_path, **kwargs)
        if model_type == 'GAM':
            kwargs = {'n_splines': 25}
            return GeneralizedAdditive(model_path, **kwargs)
        if model_type == 'XGBCls':
            kwargs = {'n_estimators': 900, 'learning_rate': 0.03, 'gamma': 0,
                      'subsample': 0.75, 'colsample_bytree': 1.0, 'max_depth': 9}
            return GradientBoostClass(model_path, num_classes, **kwargs)

    @classmethod
    def load(cls, model_type, model_path):
        if model_type == 'TorchNN':
            return TorchNN(model_path)
        if model_type == 'RF':
            return RandomForest(model_path)
        if model_type == 'XGB':
            return GradientBoost(model_path)
        if model_type == 'MLP':
            return MultilayerPerceptron(model_path)
        if model_type == 'GAM':
            return GeneralizedAdditive(model_path)
        if model_type == 'XGBCls':
            return GradientBoostClass(model_path)
