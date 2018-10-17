import os
import numpy as np
import pandas as pd
import xgboost as xgb
import torch
from tqdm import tqdm

from sklearn.ensemble import RandomForestRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.externals import joblib
from sklearn.metrics import confusion_matrix
from torch.utils import data
import torch.nn as nn
from pygam import LinearGAM
from lognet.models import structured_models


def make_gathers(dataset):
    datalen = len(dataset)
    y_data = []
    X_data = []
 
    for ix in range(datalen):
        target, feature = dataset[ix]
        y_data.append(target)
        X_data.append(feature)
        
    return X_data, y_data       
            
       


class SmartModel():
    """
    Super class for all models - DeepLearning, Random Forest, Gradient Boost
    """

    def __init__(self, **kwargs):
        pass

    def _regression_error(self, val_loader, model):
        rmse = []
        for y_valid, X_valid in tqdm(val_loader):
            y_pred = model.predict(X_valid)
            rmse.extend(np.square(np.subtract(y_valid.numpy().flatten(),y_pred)))
        return np.sqrt(np.mean(rmse))

    def _classification_error(self, val_loader, model):
        conf_mat = np.zeros((self.num_classes, self.num_classes))
        for y_valid, X_valid in tqdm(val_loader):
            y_pred = model.predict(X_valid)
            conf_mat = np.add(conf_mat, confusion_matrix(y_valid.numpy().flatten(), y_pred, labels=np.arange(self.num_classes)))
        return conf_mat

    def train(self, training_data, validation_data):
        type_name = self.__class__.__name__
        model_name = type_name + '.pkl'
        print(f'{type_name} training.')

        X_train, y_train = make_gathers(training_data)
        print('train_len',len(X_train))
        X_val, y_val = make_gathers(validation_data)
        print('val_len',len(y_val))
        trained_model = self.model.fit(np.asarray(X_train), np.asarray(y_train))
        joblib.dump(trained_model, os.path.join(self.model_path, model_name))

        validation_loader = data.DataLoader(validation_data, batch_size=64,
                                            num_workers=8, shuffle=False, drop_last=False)
        if type_name.endswith('Class'):
            err = self._classification_error(validation_loader, trained_model)
        else:
            err = self._regression_error(validation_loader, trained_model)
        return err

    def predict(self, test_data):
        type_name = self.__class__.__name__
        model_name = type_name + '.pkl'
        print(f'Predicting with {type_name}.')

        trained_model = joblib.load(os.path.join(self.model_path, model_name))
        test_loader = data.DataLoader(test_data, batch_size=64,
                                      num_workers=8, shuffle=False, drop_last=False)
        y_pred = []
        for y_test, X_test in tqdm(test_loader):
            y_pred.extend(trained_model.predict(X_test))
        return y_pred


class TorchNN(SmartModel):

    def __init__(self, model_path, **kwargs):
        super().__init__()
        print('Using TorchNN model.')
        self.model_path = model_path
        if kwargs:
            assert (isinstance(kwargs['batch_size'],int)==True),"batch_size is not an integer"
            self.batch_size = kwargs['batch_size']
            assert (isinstance(kwargs['linear_layer_sizes'],int)==True),"linear_layer_sizes is not an integer"
            self.lin_sizes = kwargs['linear_layer_sizes']
            assert (isinstance(kwargs['linear_layer_dropouts'],float)==True),"linear_layer_dropouts is not a float"
            self.lin_dropouts = kwargs['linear_layer_dropouts']
            assert (isinstance(kwargs['learning_rate'],float)==True),"learning_rate is not a float"
            self.lr = kwargs['learning_rate']
            assert (isinstance(kwargs['n_epoch'],int)==True),"n_epoch is not an integer"
            self.n_epoch = kwargs['n_epoch']
            
            
    def train(self, training_data, validation_data):
        print('TorchNN training.')
        type_name = self.__class__.__name__
        model_name = type_name + '.pkl'
        training_loader = data.DataLoader(training_data, batch_size=self.batch_size,
                                          num_workers=12, shuffle=True, drop_last=False)
        validation_loader = data.DataLoader(validation_data, batch_size=self.batch_size,
                                            num_workers=8, shuffle=False, drop_last=False)
        y_train, X_train = training_data[0]

        n_continuous_variables = len(X_train)
        output_size = 1
        model_arch = {'emb_sz': None, 'n_conts': len(
            X_train), 'emb_do': None, 'out_sz': 1, 'lin_sz': self.lin_sizes, 'lin_do': self.lin_dropouts}
        model = structured_models.StructuredModel(None, n_continuous_variables,
                                                  None, output_size,
                                                  self.lin_sizes, self.lin_dropouts)
        model = nn.DataParallel(
            model, device_ids=range(torch.cuda.device_count()))
        model.cuda()
        optimizer = torch.optim.SGD(
            model.parameters(), lr=self.lr, momentum=0.99, weight_decay=5e-4)
        best_val_loss = 100000000.0
        for epoch in range(self.n_epoch):
            model.train()
            for i, (y, x) in tqdm(enumerate(training_loader)):
                features = x.float().cuda()
                y_expected = y.float().cuda()
                y_predicted = model(features)
                loss = torch.norm(y_predicted-y_expected) / \
                    np.sqrt(self.batch_size)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                if (i+1) % 1000 == 0:
                    print("Epoch [%d/%d], Minibatch: %d, Training loss: %.4f" %
                          (epoch+1, self.n_epoch, i, loss.data[0]))

            model.eval()
            val_batch_count = 0
            val_loss = 0.0

            for j, (y, x) in tqdm(enumerate(validation_loader)):
                with torch.no_grad():
                    val_batch_count += 1
                    features = x.float().cuda(non_blocking=True)

                    y_expected = y.float().cuda(non_blocking=True)
                    y_predicted = model(features)

                    batch_loss = torch.norm(
                        y_predicted-y_expected) / np.sqrt(self.batch_size)

                    val_loss += batch_loss

                    if (j+1) % 1000 == 0:
                        print("Epoch [%d/%d], Minibatch: %d, Validation loss: %.4f" %
                              (epoch+1, self.n_epoch, j, batch_loss.data[0]))

            val_loss = val_loss / float(val_batch_count)

            if val_loss.cpu().data[0] <= best_val_loss:
                print("Best validation error: %.4f, Current validation error: %.4f" % (
                    best_val_loss, val_loss.cpu().data[0]))
                best_val_loss = val_loss.cpu().data[0]

                state = {'epoch': epoch+1,
                         'arch': model_arch,
                         'model_state': model.module.state_dict(),
                         'optimizer_state': optimizer.state_dict(), }
                print('Saving model at: ', os.path.join(
                    self.model_path, model_name))
                torch.save(state, os.path.join(self.model_path, model_name))

    def predict(self, test_data):
        type_name = self.__class__.__name__
        model_name = type_name + '.pkl'
        print(f'Predicting with {type_name}.')
        #trained_model = joblib.load(os.path.join(self.model_path, model_name))
        checkpoint = torch.load(os.path.join(self.model_path, model_name))
        #print(checkpoint['epoch'])
        #print(checkpoint['arch'])
        # print(checkpoint['model_state'])
        arch = checkpoint['arch']
        model = structured_models.StructuredModel(arch['emb_sz'], arch['n_conts'],
                                                  arch['emb_do'], arch['out_sz'],
                                                  arch['lin_sz'], arch['lin_do'])
        model.load_state_dict(checkpoint['model_state'])
        test_loader = data.DataLoader(test_data, batch_size=64,
                                      num_workers=8, shuffle=False, drop_last=False)
        y_pred = []
        #print(len(test_loader))
        for y_test, x_test in test_loader:
            with torch.no_grad():
                features = x_test.float()
                y_pred.extend(model(features).numpy().flatten())
        return y_pred


class RandomForest(SmartModel):

    def __init__(self, model_path, **kwargs):
        super().__init__()
        print('Using RandomForest model.')
        self.model_path = model_path
        if kwargs:
            
            assert (isinstance(kwargs['n_estimators'],int)==True),"n_estimators is not an integer"
            assert (isinstance(kwargs['max_depth'],int)==True),"max_depth is not an integer"
            assert (isinstance(kwargs['max_features'],int)==True),"max_features is not an integer"
            
            self.model = RandomForestRegressor(n_estimators=kwargs['n_estimators'],
                                               max_depth=kwargs['max_depth'],
                                               max_features=kwargs['max_features'],
                                               n_jobs=-1, oob_score=True, random_state=0)


class GradientBoost(SmartModel):

    def __init__(self, model_path, **kwargs):
        super().__init__()
        print('Using GradientBoost model.')
        self.model_path = model_path
        if kwargs:
            
            assert (isinstance(kwargs['n_estimators'],int)==True),"n_estimators is not an integer"
            assert (isinstance(kwargs['learning_rate'],float)==True),"learning_rate is not float"
            assert (isinstance(kwargs['gamma'],int)==True),"gamma is not an integer"
            assert (isinstance(kwargs['subsample'],float)==True),"subsample is not float"
            assert (isinstance(kwargs['colsample_bytree'],float)==True),"colsample_bytree is not float"
            assert (isinstance(kwargs['max_depth'],int)==True),"max_depth is not an integer"
            
            self.model = xgb.XGBRegressor(n_estimators=kwargs['n_estimators'],
                                          learning_rate=kwargs['learning_rate'],
                                          gamma=kwargs['gamma'],
                                          subsample=kwargs['subsample'],
                                          colsample_bytree=kwargs['colsample_bytree'],
                                          max_depth=kwargs['max_depth'],
                                          n_jobs=-1,tree_method = 'gpu_hist')


class MultilayerPerceptron(SmartModel):

    def __init__(self, model_path, **kwargs):
        super().__init__()
        print('Using MultilayerPerceptron model.')
        self.model_path = model_path
        if kwargs:
            
            assert (isinstance(kwargs['alpha'],float)==True),"alpha is not float"
            assert (isinstance(kwargs['hidden_layer_sizes'],int)==True),"hidden_layer_sizes is not an integer"
            assert (isinstance(kwargs['learning_rate_init'],int)==True),"learning_rate_init is not float"
            
            self.model = MLPRegressor(activation='relu',
                                      alpha=kwargs['alpha'],
                                      batch_size='auto',
                                      beta_1=0.9,
                                      beta_2=0.999,
                                      early_stopping=False,
                                      epsilon=1e-10,
                                      hidden_layer_sizes=(
                                          kwargs['hidden_layer_sizes'],),
                                      learning_rate='constant',
                                      learning_rate_init=kwargs['learning_rate_init'],
                                      max_iter=1000,
                                      momentum=0.9,
                                      nesterovs_momentum=True,
                                      power_t=0.5,
                                      random_state=42,
                                      shuffle=True,
                                      solver='adam',
                                      tol=0.0001,
                                      validation_fraction=0.1,
                                      verbose=False,
                                      warm_start=False)


class GeneralizedAdditive(SmartModel):

    def __init__(self, model_path, **kwargs):
        super().__init__()
        print('Using GeneralizedAdditive model.')
        self.model_path = model_path
        if kwargs:
            self.model = LinearGAM(n_splines=kwargs['n_splines'])

class GradientBoostClass(SmartModel):

    def __init__(self, model_path, num_classes, **kwargs):
        super().__init__()
        print('Using GradientBoost Classifier model.')
        self.model_path = model_path
        self.num_classes = num_classes
        if kwargs:
            
            assert (isinstance(kwargs['n_estimators'],int)==True),"n_estimators is not an integer"
            assert (isinstance(kwargs['learning_rate'],float)==True),"learning_rate is not float"
            assert (isinstance(kwargs['gamma'],int)==True),"gamma is not an integer"
            assert (isinstance(kwargs['subsample'],float)==True),"subsample is not float"
            assert (isinstance(kwargs['colsample_bytree'],float)==True),"colsample_bytree is not float"
            assert (isinstance(kwargs['max_depth'],int)==True),"max_depth is not an integer"
            
            self.model = xgb.XGBClassifier(n_estimators=kwargs['n_estimators'],
                                           learning_rate=kwargs['learning_rate'],
                                           gamma=kwargs['gamma'],
                                           subsample=kwargs['subsample'],
                                           colsample_bytree=kwargs['colsample_bytree'],
                                           max_depth=kwargs['max_depth'],
                                           n_jobs=-1,tree_method = 'gpu_hist')