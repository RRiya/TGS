import os, sys
import random
import argparse

import torch

import pandas as pd
import numpy as np
import torch.nn as nn
import torchvision.models as models

from torch.utils import data
from tqdm import tqdm

import visdom

from lognet.loader import structured_data_loader
from lognet.models import structured_models

from sklearn.model_selection import train_test_split

def main(args):

    training_data_path = args.training_path
    model_save_location = args.model_location 

    batch_size = args.batch_size
    learning_rate = args.learning_rate 
    n_epoch = args.epoch_count
    visualization = args.visdom

    linear_layer_sizes = args.hidden_layers
    linear_layer_dropouts = args.dropout_layers

    assert( len(linear_layer_sizes) == len(linear_layer_dropouts) )
    assert( len(linear_layer_sizes) >= 1 )
    assert( all(isinstance(item, int) for item in linear_layer_sizes) == True )
    assert( all(isinstance(item, float) for item in linear_layer_dropouts) == True )

    # TODO: Check both these paths
    full_file_list = [file for file in os.listdir(training_data_path) if file.endswith("csv")]
    random.shuffle(full_file_list)

    print('Total number of files: ', len(full_file_list))

    #TODO: This is a random number that has no meaning
    file_list = random.sample(full_file_list, 350)
    #file_list = full_file_list

    if not os.path.isfile(os.path.join(training_data_path, "headers.txt")):
        error_message = 'Path: ' + training_data_path + ' does not contain `headers.txt`'
        raise FileNotFoundError(error_message)

    header_file = os.path.join(training_data_path, "headers.txt")

    training_list, validation_list = train_test_split(file_list, test_size=0.25, random_state=424242)

    print('Logs in training dataset: ', len(training_list))
    print('Logs in validation dataset: ', len(validation_list))

    #TODO: Need to be able to take any input target - for now hardcoding to 'Sonic'
    training_dataset = structured_data_loader.StructuredDataset(training_data_path, header_file, training_list, ['Sonic'])
    validation_dataset = structured_data_loader.StructuredDataset(training_data_path, header_file, validation_list, ['Sonic'])

    y, x = training_dataset[0]
    continuous_features = len(x)
    print('Number of continuous features: ', continuous_features)

    training_loader = data.DataLoader(training_dataset, batch_size=batch_size, num_workers=12, shuffle=True, drop_last=True)
    validation_loader = data.DataLoader(validation_dataset, batch_size=batch_size, num_workers=8, shuffle=False, drop_last=False)

    print('Number of minibatches in training: ', len(training_loader))
    print('Number of minibatches in validation: ', len(validation_loader))

    model_name = 'SonicModel_' + '_'.join([str(i) for i in linear_layer_sizes]) + '_' +\
                                '_'.join([str(i) for i in linear_layer_dropouts]) + \
                                str(continuous_features) + '.pkl'

    model_path = os.path.join(model_save_location, model_name)

    if visualization:
        vis = visdom.Visdom()

        tr_win = vis.line(X=np.zeros(1,), Y=np.zeros(1,), 
            opts=dict(xlabel='Minibatches', 
                    ylabel='Training loss',
                    title='Training loss',
                    legend=['Loss']))

        val_win = vis.line(X=np.zeros(1,), Y=np.zeros(1,),
                opts=dict(xlabel='Minibatches',
                        ylabel='Validation loss',
                        title='Validation loss',
                        legend=['Loss']))

    embedding_sizes = None
    n_continuous_variables = continuous_features
    embedding_dropout = None
    output_size = 1
    use_batchnorm = True

    print('Hidden layer sizes: ', linear_layer_sizes)
    print('Hidden layer dropouts: ', linear_layer_dropouts)
    model = structured_models.StructuredModel(embedding_sizes, n_continuous_variables, embedding_dropout, output_size, linear_layer_sizes, linear_layer_dropouts)
    model = nn.DataParallel(model, device_ids = range(torch.cuda.device_count()))
    model.cuda()

    optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate, momentum=0.99, weight_decay=5e-4)
    # scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=8, verbose=True)

    best_val_loss = 10000000000.0

    for epoch in range(n_epoch):

        model.train()
    
        for i, (y, x) in tqdm(enumerate(training_loader)):
            
            features = x.float().cuda()

            y_expected  = y.float().cuda()
            y_predicted = model(features)
            loss = torch.norm(y_predicted-y_expected) / np.sqrt(batch_size)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            #TODO: Enable visdom visualization
            if visualization and ((i+1) % 200 == 0):
                vis.line(X=torch.ones((1,)) * (i + len(training_loader) * epoch), Y=torch.Tensor([loss.data[0]]).unsqueeze(0).cpu(), win=tr_win, update='append')

            if (i+1) % 1000 == 0:
                print("Epoch [%d/%d], Minibatch: %d, Training loss: %.4f" %(epoch+1, n_epoch, i, loss.data[0]))
    
        model.eval()
    
        val_batch_count = 0
        val_loss = 0.0
    
        for j, (y, x) in tqdm(enumerate(validation_loader)):

            with torch.no_grad():
                val_batch_count += 1
                features = x.float().cuda(non_blocking=True)

                y_expected  = y.float().cuda(non_blocking=True)
                y_predicted = model(features)

                batch_loss = torch.norm(y_predicted-y_expected) / np.sqrt(batch_size)

                val_loss += batch_loss

                #TODO: Enable visdom visualization
                if visualization and ((j+1) % 200 == 0):
                    vis.line(X=torch.ones((1,)) * (j + len(validation_loader) * epoch), Y=torch.Tensor([batch_loss.data[0]]).unsqueeze(0).cpu(), win=val_win, update='append')

                if (j+1) % 1000 == 0:
                    print("Epoch [%d/%d], Minibatch: %d, Validation loss: %.4f" %(epoch+1, n_epoch, j, batch_loss.data[0]))

        val_loss = val_loss / float(val_batch_count)
    
        # scheduler.step(val_loss.cpu().data[0])
    
        if val_loss.cpu().data[0] <= best_val_loss:
            print("Best validation error: %.4f,  Current validation error: %.4f" %(best_val_loss, val_loss.cpu().data[0])) 
            best_val_loss = val_loss.cpu().data[0]
        
            # TODO: Enable restarting - needs to write more than just the state
            state = {'epoch': epoch+1,
                     'model_state': model.state_dict(),
                     'optimizer_state': optimizer.state_dict(),}
        
            print('Saving model at: ', model_path)
            torch.save(state, model_path)

if __name__ == '__main__':
    help_string = "PyTorch LogNet training"

    parser = argparse.ArgumentParser(description=help_string)

    parser.add_argument('-t', '--training-path', type=str, metavar='DIR', help='Path where the training data exists', required=True)
    parser.add_argument('-m', '--model-location', type=str, metavar='DIR', help='Path where the trained model needs to be stored', required=True)

    parser.add_argument('-bs', '--batch-size', type=int, metavar='N', help='Batch size (default: 20)', default=20, required=False)
    parser.add_argument('-epoch', '--epoch-count', type=int, metavar='N', help='Number of epochs (default: 40)', default=40, required=False)
    parser.add_argument('-lr', '--learning-rate', type=float, metavar='LR', help='Initial learning rate (default: 0.005)', default=5e-3, required=False)
    parser.add_argument('--visdom', type=bool, help='Enable realtime visualization (default: True)', default=True, required=False)

    parser.add_argument('-hl', '--hidden-layers', type=int, nargs='+', metavar='LIST', default=[100, 50, 25],
                        help='Size of linear layers (default: [100, 50, 25])', required=False)
    parser.add_argument('-do', '--dropout-layers', type=float, nargs='+', metavar='LIST', default=[0.01, 0.005, 0.005],
                        help='Dropout of linear layers (default: [0.01, 0.005, 0.005])', required=False)

    args = parser.parse_args()

    main(args)

    sys.exit(1)

