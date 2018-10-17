import torch
import torch.nn as nn
import torch.nn.init as weight_init
import torch.nn.functional as F

def initialize_embedding(x):
    x = x.weight.data
    sc = 2.0 / (x.size(1) + 1)
    x.uniform_(-sc, sc)

class DenseModel(nn.Module):
    """Implements a dense model.

    Arguments:
        embedding_sizes (list): Size of the embedding layers
        n_continuous_variables (int): Number of continuous variables
        embedding_dropout (float): Dropout value for the embedding layer
        output_size (int): Number of values predicted
        layer_sizes (list): Number of nodes in the hidden layers
        linear_layer_dropouts (list): Dropout values for the hidden layers
    """
    def __init__(self, embedding_sizes, n_continuous_variables,
                 embedding_dropout, output_size, layer_sizes, 
                 linear_layer_dropouts, output_range=None,
                 use_batchnorm=False, is_regression=True,
                 is_multilabel=False) -> None:
        
        super().__init__()
        
        if embedding_sizes is not None:
            self.embeddings = nn.ModuleList([nn.Embedding(c, s) for c, s in embedding_sizes])
        
            for embedding in self.embeddings:
                initialize_embedding(embedding)
        
            n_embedding = sum(embedding.embedding_dim for embedding in self.embeddings)
        else:
            self.embeddings = None
            n_embedding = 0
            
        self.n_embedding = n_embedding
        self.n_continous = n_continuous_variables
        
        layer_sizes = [n_embedding+n_continuous_variables] + layer_sizes
        
        self.linear_layers = nn.ModuleList([nn.Linear(layer_sizes[i], layer_sizes[i+1]) for i in range(len(layer_sizes)-1)])
        
        self.batchnorms = nn.ModuleList([nn.BatchNorm1d(size) for size in layer_sizes[1:]])
        
        for o in self.linear_layers:
            weight_init.kaiming_normal(o.weight.data)
        
        self.output_layer = nn.Linear(layer_sizes[-1], output_size)
        
        weight_init.kaiming_normal(self.output_layer.weight.data)
        
        if embedding_sizes is not None:
            self.embedding_dropout = nn.Dropout(embedding_dropout)
        
        self.dropouts = nn.ModuleList([nn.Dropout(coeff) for coeff in linear_layer_dropouts])
        self.continuous_batchnorm = nn.BatchNorm1d(n_continuous_variables)
        
        self.use_batchnorm = use_batchnorm
        self.output_range = output_range
        
        self.is_regression = is_regression
        self.is_multilabel = is_multilabel
    
    def forward(self, input_continuous, input_categorical=None):
        
        if self.n_embedding != 0:
            x = [e(input_categorical[:,i]) for i, e in enumerate(self.embeddings)]
            x = torch.cat(x, 1)
            x = self.embedding_dropout(x)
        
        if self.n_continous != 0:
            x2 = self.continuous_batchnorm(input_continuous)
            if self.n_embedding !=0:
                x = torch.cat([x, x2], 1)
            else:
                x = x2
        
        for l, d, b in zip(self.linear_layers, self.dropouts, self.batchnorms):
            x = F.relu(l(x))
            if self.use_batchnorm:
                x = b(x)
            x = d(x)
        
        x = self.output_layer(x)
        
        if not self.is_regression:
            if self.is_multilabel:
                x = F.sigmoid(x)
            else:
                x = F.log_softmax(x)
        elif self.output_range:
            x = F.sigmoid(x)
            x = x*(self.output_range[1] - self.output_range[0])
            x = x + self.output_range[0]
        
        return x
