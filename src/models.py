import torch.nn as nn
import torch.nn.functional as F
import torch
import numpy as np
from src.unet import UNet
from  src.dataset import count_features

def load_model(training_params, hparam, device='cpu'):
    """ Get model: Linear, UNet or MSDNet
        Args:
            training_params (dict):
                Dictionary containing training parameters defined in training_params.json
            hparam (Ordered tuple):
                Model hyperparameters

        Returns:
            model (nn.Module):
                Neural network or linear regression model

    """
    n_features= count_features(training_params.features)

    if 'model_architecture' not in vars(training_params):
        model = UNet(n_features,
                     n_classes=1,
                     depth=hparam.depth,
                     wf=hparam.wf,
                     padding=bool(hparam.padding),
                     batch_norm=bool(hparam.batch_norm),
                     up_mode=hparam.up_mode,
                     final_relu=True
                     )
        print('Model architecture name not defined: using UNet.')
        
    elif training_params.model_architecture == 'linear':
        model = LinearRegression(n_features, output_size=1)

    elif training_params.model_architecture == 'unet':
        model = UNet(n_features,
                     n_classes=1,
                     depth=hparam.depth,
                     wf=hparam.wf,
                     padding=bool(hparam.padding),
                     batch_norm=bool(hparam.batch_norm),
                     up_mode=hparam.up_mode,
                     final_relu=True
                     )

    model = model.to(device)

    return model


def init_bias(model, device, val):    
    for name, parameter in model.named_modules():
        if name == 'final_linear':
            init_bias = nn.Parameter(torch.Tensor(np.ones(model.final_linear.bias.shape)*val).to(device))
            model.final_linear.bias = init_bias
    return model


class LinearRegression(nn.Module):
    def __init__(self, input_size, output_size=1, bias=True):
        super(LinearRegression, self).__init__()
        self.linear = nn.Linear(input_size, output_size, bias=bias)
        
    def forward(self, x):
        out = self.linear(x)
        return out

