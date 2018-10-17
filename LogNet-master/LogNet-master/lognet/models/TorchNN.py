import os
import numpy as np
import torch
from tqdm import tqdm

from torch.utils import data
import torch.nn as nn
from lognet.models import structured_models

class TorchNN():

    def __init__(self, model_path, **kwargs):
        print('Using TorchNN model.')
        self.model_params = {'batch_size': 64, 'linear_layer_sizes': [64], 'linear_layer_dropouts': [0.5], 'learning_rate': 0.01, 'n_epoch': 10}
        self.model_path = model_path
        if kwargs:
            self.batch_size = kwargs.get('batch_size', self.model_params['batch_size'])
            self.lin_sizes = kwargs.get('linear_layer_sizes', self.model_params['linear_layer_sizes'])
            self.lin_dropouts = kwargs.get('linear_layer_dropouts', self.model_params['linear_layer_dropouts'])
            self.lr = kwargs.get('learning_rate', self.model_params['learning_rate'])
            self.n_epoch = kwargs.get('n_epoch', self.model_params['n_epoch'])
        
            
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
        model = structured_models.DenseModel(None, n_continuous_variables, None, output_size, self.lin_sizes, self.lin_dropouts)
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
        model = structured_models.DenseModel(arch['emb_sz'], arch['n_conts'],
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