import torch
from torch import nn, Tensor
import torch.nn.functional as F
from torch.nn import TransformerEncoder, TransformerEncoderLayer
from torch.utils.data import dataset
import math

class TransformerModel(nn.Module):

    def __init__(self, output_dim: int, d_model: int, nhead: int, d_hid: int,
                 nlayers: int, dropout: float = 0.5, encoder=None):
        super().__init__()
        self.model_type = 'Transformer'
        self.pos_encoder = PositionalEncoding(d_model, dropout)
        encoder_layers = TransformerEncoderLayer(d_model, nhead, d_hid, dropout)
        self.transformer_encoder = TransformerEncoder(encoder_layers, nlayers)
        if encoder is None:
            self.emb = nn.Embedding(ntoken, d_model)
        else:
            self.emb = encoder
        self.d_model = d_model
        self.out = nn.Linear(d_model, output_dim)

        self.init_weights()

    def init_weights(self) -> None:
        initrange = 0.1
        self.emb.weight.data.uniform_(-initrange, initrange)
        self.out.bias.data.zero_()
        self.out.weight.data.uniform_(-initrange, initrange)

    def forward(self, src: Tensor, target_length: int) -> Tensor:
        """
        Args:
            src: Tensor, shape [seq_len, batch_size]
            src_mask: Tensor, shape [seq_len, seq_len]

        Returns:
            output Tensor of shape [seq_len, batch_size, ntoken]
        """

        decoder_input = torch.zeros(target_length, src.shape[1], src.shape[2], device=src.device)
        inputsall = torch.cat([src, decoder_input], dim=0)
        inputsall = self.emb(inputsall)
        inputsall = self.pos_encoder(inputsall)
        # src_mask = generate_square_subsequent_mask(target_length)
        # decoder_mask = generate_square_decoder_mask(len(inputsall), target_length).to(inputsall.device)
        # print(inputsall.shape)
        output = self.transformer_encoder(inputsall)
        # print(output.shape)

        output = self.out(output)
        return output[len(src):], None

    def init_sequence(self, batch_size):
        pass

    def calculate_num_params(self):
        """Returns the total number of parameters."""
        num_params = 0
        for p in self.parameters():
            num_params += p.data.view(-1).size(0)
        return num_params


def generate_square_subsequent_mask(sz: int) -> Tensor:
    """Generates an upper-triangular matrix of -inf, with zeros on diag."""
    return torch.triu(torch.ones(sz, sz) * float('-inf'), diagonal=1)

def generate_square_decoder_mask(seq_len: int, decoder_len: int) -> Tensor:
    """Generates an upper-triangular matrix of -inf, with zeros on diag."""
    mask1 = torch.ones(seq_len, decoder_len) * float('-inf')
    mask2 = torch.zeros(seq_len, seq_len-decoder_len) * float('-inf')
    return torch.cat([mask1,mask2],dim=-1)

class PositionalEncoding(nn.Module):

    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, 1, d_model)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x: Tensor) -> Tensor:
        """
        Args:
            x: Tensor, shape [seq_len, batch_size, embedding_dim]
        """
        x = x + self.pe[:x.size(0)]
        return self.dropout(x)
