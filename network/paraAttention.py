from .attention import *
from .convlstm import *
from .spps import *


class ParaAttention(nn.Module):
    def __init__(self, input_dim, hidden_dim, kernel_size, num_layers,
                 batch_first=False, bias=True, return_all_layers=False):
        super(ParaAttention, self).__init__()
        self.way1_spps = SPPS(input_dim=input_dim, hidden_dim=hidden_dim, kernel_size=kernel_size,
                              num_layers=num_layers, batch_first=batch_first, bias=bias,
                              return_all_layers=return_all_layers)
        self.way2_convlstm = Cascade2ConvLSTM(input_dim=input_dim, hidden_dim=hidden_dim, kernel_size=kernel_size,
                                              num_layers=num_layers, batch_first=batch_first, bias=bias,
                                              return_all_layers=return_all_layers)
        self.way2_att = SelfAttention(input_size=hidden_dim, output_size=hidden_dim)
        self.way3_dbconvlstm = BiConvLSTM(input_dim=input_dim, hidden_dim=hidden_dim, kernel_size=kernel_size,
                                          num_layers=num_layers, batch_first=batch_first, bias=bias,
                                          return_all_layers=return_all_layers)
        self.fc = nn.Linear(hidden_dim, 2)

    def forward(self, x):
        w1, _, _ = self.way1_spps(x)

        w2, _ = self.way2_convlstm(x)
        att_score, att_weights_ = self.att(w2)

        w3, _ = self.way3_dbconvlstm(x)

        return w1 + att_score + w3
