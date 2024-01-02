import torch
import torch.nn as nn
from torch.autograd import Function, Variable

# https://gist.github.com/daskol/05439f018465c8fb42ae547b8cc8a77b
class Maxout(nn.Module):
    """Class Maxout implements maxout unit introduced in paper by Goodfellow et al, 2013.
    
    :param in_feature: Size of each input sample.
    :param out_feature: Size of each output sample.
    :param n_channels: The number of linear pieces used to make each maxout unit.
    :param bias: If set to False, the layer will not learn an additive bias.
    """
    
    def __init__(self, in_features, out_features, n_channels, bias=True):
        super().__init__()
        
        self.in_features = in_features
        self.out_features = out_features
        self.n_channels = n_channels
        self.weight = nn.Parameter(torch.Tensor(n_channels * out_features, in_features))
        
        if bias:
            self.bias = nn.Parameter(torch.Tensor(n_channels * out_features))
        else:
            self.register_parameter('bias', None)
            
        self.reset_parameters()
    
    def forward(self, input):
        a = nn.functional.linear(input, self.weight, self.bias)
        b = nn.functional.max_pool1d(a.unsqueeze(-3), kernel_size=self.n_channels)
        return b.squeeze()
    
    def reset_parameters(self):
        irange = 0.005
        nn.init.uniform_(self.weight, -irange, irange)
        if self.bias is not None:
            nn.init.uniform_(self.bias, -irange, irange)

    def extra_repr(self):
        return (f'in_features={self.in_features}, '
                f'out_features={self.out_features}, '
                f'n_channels={self.n_channels}, '
                f'bias={self.bias is not None}')

class VanillaDiscriminator(nn.Module):
    def __init__(self, input_size):
        super(VanillaDiscriminator, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(input_size, 1024),
            Maxout(input_size, 1024, 1),
            nn.Dropout(0.3),
            nn.Linear(1024, 512),
            Maxout(1024, 512, 1),
            nn.Dropout(0.3),
            nn.Linear(512, 256),
            Maxout(512, 256, 1),
            nn.Dropout(0.3),
            nn.Linear(256, 1),
            Maxout(256, 1, 1)
        )
        
    def forward(self, data):
        return self.model(data)


if __name__ == "__main__":
    a = VanillaDiscriminator(28 * 28)
    print(a)
