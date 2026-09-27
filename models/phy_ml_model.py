#Issues in model is normalization for dec_query input
import torch
import torch.nn as nn
import torch.nn.functional as F
from models.positional_encoding import PositionalEmbedding
class phy_ml_block(nn.Module):
    def __init__(self,
                 enc_in,
                 dec_in,
                 seq_len_enc,
                 seq_len_dec,
                 enc_out,
                 enc_layers,
                 dec_layers,
                 d_model=512,
                 n_head=8,
                 dropout=0.1,
                 activation="gelu",
                 device=torch.device('cuda:0')):
        super(phy_ml_block,self).__init__()

        self.device = device
        self.d_model = d_model
        self.seq_len_dec = seq_len_dec

        self.encoder_pos_encoder = PositionalEmbedding(d_model, max_len=seq_len_enc)
        self.encoder_input_projection = nn.Sequential(nn.Linear(enc_in, d_model),nn.GELU())

        self.decoder_input_projection = nn.Sequential(nn.Linear(dec_in, d_model),nn.GELU())
        self.decoder_pos_encoder = PositionalEmbedding(d_model, max_len=seq_len_dec)

        encoder_layer = nn.TransformerEncoderLayer(d_model=d_model,nhead=n_head, dropout=dropout,activation=activation,batch_first=True)
        self.encoder = nn.TransformerEncoder(encoder_layer,num_layers=enc_layers)
        decoder_layer = nn.TransformerDecoderLayer(d_model=d_model, nhead=n_head, dropout=dropout, activation=activation, batch_first=True)
        self.decoder = nn.TransformerDecoder(decoder_layer, num_layers=dec_layers)
        self.output_proj = nn.Linear(d_model,enc_out) #enc_out is nothing but dec_out


    def forward(self,enc_in,dec_in):
        enc = self.encoder_input_projection(enc_in)
        enc = enc + self.encoder_pos_encoder(enc)
        memory = self.encoder(enc)   # [B, seq_len_enc, d_model]

        # ----- Decoder -----
        dec = self.decoder_input_projection(dec_in)
        dec = dec + self.decoder_pos_encoder(dec)

        # generate causal mask for autoregression
        tgt_mask = nn.Transformer.generate_square_subsequent_mask(
            self.seq_len_dec
        ).to(enc_in.device)

        out = self.decoder(dec, memory,tgt_mask=tgt_mask)
        out = self.output_proj(out)  # [B, seq_len_dec, enc_out]
        return out

