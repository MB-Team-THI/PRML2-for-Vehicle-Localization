#Issues in model is normalization for dec_query input
import torch
import torch.nn as nn
import torch.nn.functional as F
from models.positional_encoding import PositionalEmbedding
class final_ml_block(nn.Module):
    def __init__(self,
                 enc_in,
                 dec_in,
                 seq_len_enc,
                 seq_len_dec,
                 enc_out,
                 enc_layers,
                 dec_layers,
                 a_max,
                 v_max,
                 #scaler,
                 d_model=512,
                 n_head=8,
                 dropout=0.1,
                 activation="gelu",
                 device=torch.device('cuda:0')):
        super(final_ml_block,self).__init__()

        self.device = device
        self.d_model = d_model
        self.seq_len_dec = seq_len_dec

        self.a_max = a_max
        self.v_max = v_max

        self.encoder_pos_encoder = PositionalEmbedding(d_model, max_len=seq_len_enc)
        self.encoder_input_projection = nn.Sequential(nn.Linear(enc_in, d_model),nn.GELU())

        self.decoder_input_projection = nn.Sequential(nn.Linear(dec_in, d_model),nn.GELU())
        self.decoder_pos_encoder = PositionalEmbedding(d_model, max_len=seq_len_dec)

        encoder_layer = nn.TransformerEncoderLayer(d_model=d_model,nhead=n_head, dropout=dropout,activation=activation,batch_first=True)
        self.encoder = nn.TransformerEncoder(encoder_layer,num_layers=enc_layers)
        decoder_layer = nn.TransformerDecoderLayer(d_model=d_model, nhead=n_head, dropout=dropout, activation=activation, batch_first=True)
        self.decoder = nn.TransformerDecoder(decoder_layer, num_layers=dec_layers)
        self.mean_head = nn.Linear(d_model,enc_out)#enc_out is nothing but dec_out
        self.cov_head = nn.Linear(d_model, (enc_out*(enc_out + 1)) // 2)

    def forward(self,enc_in,dec_in,scaler):
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

        out = self.decoder(dec, memory, tgt_mask=tgt_mask)

        #Network final heads
        mean = self.mean_head(out)  # [B, seq_len_dec, enc_out]
        raw_cov = self.cov_head(out)

        #Apply physics guard for mean
        mean = self.physics_guard(scaler,mean)

        # build covariance
        L = self.build_lower_triangular(raw_cov, enc_out=mean.shape[-1])
        cov = L @ L.transpose(-1, -2)
        return mean, cov

    def physics_guard(self, scaler,out, eps=1e-6):

        mu = torch.tensor(scaler.mean, device=out.device, dtype=out.dtype)
        sigma = torch.tensor(scaler.std, device=out.device, dtype=out.dtype)

        # 1. Denormalize
        out_phys = out * sigma + mu
        vel = out_phys[..., 0:2]
        acc = out_phys[..., 2:4]

        # 2. Clamp vector magnitudes
        def clamp_norm(x, max_norm):
            norm = torch.norm(x, dim=-1, keepdim=True)
            scale = torch.clamp(max_norm / (norm + eps), max=1.0)
            return x * scale

        vel = clamp_norm(vel, self.v_max)
        acc = clamp_norm(acc, self.a_max)

        # 3. Recombine
        out_phys = torch.cat([vel, acc], dim=-1)

        # 4. Back to z-space
        out_norm = (out_phys - mu) / sigma
        return out_norm

    def build_lower_triangular(self, raw, enc_out):
        B, T, _ = raw.shape
        D = enc_out

        # Output tensor
        L = torch.zeros(B, T, D, D, device=raw.device)

        # Lower-triangular indices
        tril_idx = torch.tril_indices(D, D, 0, device=raw.device)  # [2, num_entries]

        # Flatten batch and time dimensions
        raw_flat = raw.reshape(B * T, -1)  # [B*T, num_tril_entries]
        L_flat = L.reshape(B * T, D, D)

        # Assign lower-triangular entries for each (B*T) slice
        L_flat[:, tril_idx[0], tril_idx[1]] = raw_flat

        # Reshape back
        L = L_flat.view(B, T, D, D)

        # Extract diagonals and apply softplus
        diag_idx = torch.arange(enc_out, device=raw.device)
        diag = torch.nn.functional.softplus(L[..., diag_idx, diag_idx]) + 1e-3

        # Extract off-diagonals
        off_diag = L - torch.diag_embed(L[..., diag_idx, diag_idx])
        off_diag = off_diag.tanh() #newly added
        # Apply tanh only to off-diagonals, then add back diagonals
        L = off_diag.tanh() + torch.diag_embed(diag)
        return L

