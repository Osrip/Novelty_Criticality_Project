

import torch
import torch.nn.functional as F


def sigmoid_activation(x):
    return torch.sigmoid(5 * x)


def tanh_activation(x):
    return torch.tanh(2.5 * x)


def abs_activation(x):
    return torch.abs(x)


def gauss_activation(x):
    return torch.exp(-5.0 * x**2)


def identity_activation(x):
    return x


def sin_activation(x):
    return torch.sin(x)


def relu_activation(x):
    return F.relu(x)


str_to_activation = {
    'sigmoid': sigmoid_activation,
    'tanh': tanh_activation,
    'abs': abs_activation,
    'gauss': gauss_activation,
    'identity': identity_activation,
    'sin': sin_activation,
    'relu': relu_activation,
}
