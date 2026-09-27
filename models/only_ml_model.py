#Issues in model is normalization for dec_query input
import torch
import torch.nn as nn
import torch.nn.functional as F
from models.positional_encoding import PositionalEmbedding
class only_ml_block(nn.Module):
    def __init__(self,enc_in,seq_len,enc_out,enc_layers,d_model=512,n_head=8,dropout=0.0,activation="gelu",
                 device=torch.device('cuda:0')):
        super(only_ml_block,self).__init__()

        self.device = device
        self.d_model = d_model

        self.pos_encoder = PositionalEmbedding(d_model, max_len=seq_len)
        self.input_projection = nn.Sequential(nn.Linear(enc_in, d_model),nn.GELU()) #From OBD length to D_Model

        encoder_layer = nn.TransformerEncoderLayer(d_model=d_model,nhead=n_head, dropout=dropout,activation=activation,batch_first=True)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer,num_layers=enc_layers)
        self.output_proj = nn.Linear(d_model,enc_out)



    def forward(self,enc_in):
        B = enc_in.size(0)

        x = self.input_projection(enc_in)
        pos = self.pos_encoder(x)
        x = x + pos

        x = self.transformer_encoder(x)
        pseudo_m = self.output_proj(x[:,-1,:])
        return pseudo_m

