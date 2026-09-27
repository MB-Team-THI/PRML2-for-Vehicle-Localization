#Issues in model is normalization for dec_query input
import torch
import torch.nn as nn
import torch.nn.functional as F
from models.positional_encoding import PositionalEmbedding
class ml_block(nn.Module):
    def __init__(self,enc_in,seq_len,enc_out,dec_x_in, enc_layers,dec_layers,d_model=512,n_head=8,dropout=0.0,activation="gelu",
                 device=torch.device('cuda:0')):
        super(ml_block,self).__init__()

        self.device = device
        self.d_model = d_model

        #Restricts the ML model output
        self.register_buffer('physics_min', torch.tensor([-5,-5,-10,-10,-10], dtype=torch.float32))
        self.register_buffer('physics_max', torch.tensor([5,50,10,10,10], dtype=torch.float32))

        self.pos_encoder = PositionalEmbedding(d_model, max_len=seq_len)
        self.input_projection = nn.Sequential(nn.Linear(enc_in, d_model),nn.GELU()) #From OBD length to D_Model

        encoder_layer = nn.TransformerEncoderLayer(d_model=d_model,nhead=n_head, dropout=dropout,activation=activation,batch_first=True)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer,num_layers=enc_layers)
        self.output_proj = nn.Linear(d_model,enc_out)

        #self.state_proj = nn.Linear(in_features= dec_x_in + (dec_x_in * dec_x_in), out_features=d_model)
        self.state_proj = nn.Linear(in_features=dec_x_in, out_features=d_model)
        self.residual_norm = nn.LayerNorm(6)
        decoder_layer = nn.TransformerDecoderLayer(d_model=d_model, nhead=n_head, dropout=dropout, activation=activation, batch_first=True)
        self.kalman_decoder = nn.TransformerDecoder(decoder_layer, num_layers=dec_layers)
        self.state_dim = dec_x_in
        self.meas_dim = enc_out
        self.k_out = self.state_dim #'* self.meas_dim
        self.kalman_gain_proj = nn.Sequential(
                                nn.Linear(d_model, self.k_out),
                                nn.Tanh())

    def forward(self,enc_in,x_post_1):
        B = enc_in.size(0)

        x = self.input_projection(enc_in)
        pos = self.pos_encoder(x)
        x = x + pos

        memory = self.transformer_encoder(x)
        pseudo_m = self.output_proj(memory[:,-1,:])
        #pseudo_m = self.physics_output_sigmoid(pseudo_m)

        #Query generation
        # x_post_flat = x_post_1  # [B, D]
        # #p_post_flat = p_post_1.view(B, -1)
        # mins = torch.tensor([0, -5, -5, -10, -10, -10], device=x_post_flat.device, dtype=x_post_flat.dtype)
        # maxs = torch.tensor([2*torch.pi, 5, 50, 10, 10, 10], device=x_post_flat.device, dtype=x_post_flat.dtype)
        #
        # # Normalize each feature to [-1, 1]
        # x_post_norm = 2 * (x_post_flat - mins) / (maxs - mins) - 1
        x_post_norm = self.residual_norm(x_post_1) #Trying residual layer norm

        query = self.state_proj(x_post_norm).unsqueeze(1)

        # Cross-attention decoding
        decoder_out = self.kalman_decoder(tgt=query, memory=memory)
        kalman_gain = self.kalman_gain_proj(decoder_out.squeeze(1))
        #Kalman_gain_matrix = kalman_gain.view(-1,self.state_dim,self.meas_dim)
        K = torch.zeros((B, 6, 5), device=self.device)
        K[:, 0, 0] = kalman_gain[:, 0]
        K[:, 1, 0] = kalman_gain[:, 1]
        K[:, 2, 1] = kalman_gain[:, 2]
        K[:, 3, 2] = kalman_gain[:, 3]
        K[:, 4, 3] = kalman_gain[:, 4]
        K[:, 5, 4] = kalman_gain[:, 5]
        return pseudo_m, K

    def physics_output_sigmoid(self, x):
        return self.physics_min + (self.physics_max - self.physics_min) * torch.sigmoid(x)

