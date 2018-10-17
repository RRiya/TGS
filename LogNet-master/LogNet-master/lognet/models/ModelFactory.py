from lognet.models.SmartModel import RandomForest, GradientBoost, MultilayerPerceptron, GeneralizedAdditive, GradientBoostClass
from lognet.models.TorchNN import TorchNN

class ModelFactory():
    """
    Factory class to select models - DeepLearning, Random Forest, Gradient Boost

    """
    @classmethod
    def select(cls, model_path, model_type, model_params):
        if model_type == 'TorchNN':   
            return TorchNN(model_path, **model_params)
        if model_type == 'RF':
            return RandomForest(model_path, **model_params)
        if model_type == 'XGB':
            return GradientBoost(model_path, **model_params)
        if model_type == 'MLP':
            return MultilayerPerceptron(model_path, **model_params)
        if model_type == 'GAM':
            return GeneralizedAdditive(model_path, **model_params)
        if model_type == 'XGBCls':
            return GradientBoostClass(model_path, **model_params)

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
