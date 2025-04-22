import torch
import torch.nn as nn

class MLP(nn.Module):
    def __init__(self, dims, activation):
        assert len(dims) > 2

        super(MLP, self).__init__()
        layers = []
        for i in range(len(dims) - 1):
            if i != 0:
                layers.append(activation())
            layers.append(nn.Linear(dims[i], dims[i + 1]))
        self.network = nn.Sequential(*layers)

    def forward(self, x):
        return self.network(x)

class YNet(nn.Module):
    def __init__(self, 
                 left_pipe_dims, 
                 right_pipe_dims, 
                 out_pipe_dims, 
                 activation):
        assert left_pipe_dims[-1] + right_pipe_dims[-1] == out_pipe_dims[0]

        super(YNet, self).__init__()
        self.left_pipe = MLP(left_pipe_dims, activation)
        self.right_pipe = MLP(right_pipe_dims, activation)
        self.out_pipe = MLP(out_pipe_dims, activation)

    def forward(self, x, y):
        assert x.shape[0] == y.shape[0], f'Left and right inputs must align: {x.shape, y.shape}'

        return self.out_pipe(torch.cat((self.left_pipe(x), self.right_pipe(y)), dim=1))