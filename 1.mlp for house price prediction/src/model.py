import torch.nn as nn


class MLP(nn.Module):
    """
    Flexible multi-layer perceptron for tabular regression.

    Args:
        input_dim     (int):       number of input features
        hidden_dims   (list[int]): sizes of hidden layers, e.g. [256, 128]
        output_dim    (int):       number of output units (1 for regression)
        dropout       (float):     dropout probability after each hidden layer
        activation    (str):       'relu' | 'gelu' | 'tanh'
        use_batch_norm(bool):      apply BatchNorm1d before each activation
    """
    def __init__(
        self,
        input_dim,
        hidden_dims,
        output_dim=1,
        dropout=0.0,
        activation='relu',
        use_batch_norm=False
    ):
        super().__init__()
        # YOUR CODE HERE
        activation_dict={'relu':nn.ReLU(),'gelu':nn.GELU(),'tanh':nn.Tanh()}
        layers = []
        if len(hidden_dims) == 0:
            layers.append(nn.Linear(input_dim,output_dim))
        else:
            layers.append(nn.Linear(input_dim,hidden_dims[0]))
            if use_batch_norm:
                layers.append(nn.BatchNorm1d(hidden_dims[0]))
            layers.append(activation_dict[activation])
            layers.append(nn.Dropout(dropout))
            for i in range(len(hidden_dims)):
                if i < len(hidden_dims)-1:
                    layers.append(nn.Linear(hidden_dims[i],hidden_dims[i+1]))
                    if use_batch_norm:
                        layers.append(nn.BatchNorm1d(hidden_dims[i+1]))
                    layers.append(activation_dict[activation])
                    layers.append(nn.Dropout(dropout))
                else:
                    layers.append(nn.Linear(hidden_dims[-1],output_dim))
        self.model = nn.Sequential(*layers)

    def forward(self, x):
        # YOUR CODE HERE
        return self.model(x).squeeze(1)