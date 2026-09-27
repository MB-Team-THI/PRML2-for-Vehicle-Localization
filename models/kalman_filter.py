import numpy as np
import torch
import torch.nn as nn

#Initially this is trained as a Kalman Filter (KF) for simplicity, acutal problem requires Extended Kalman Filter (EKF).
#State vector - [psi,psi dot,vx,vy,ax,ay] . After code runs well. Check for x and y position
class KalmanFilterBlock(nn.Module):
    #need to initialize the init with actual equations for A.
    def __init__(self,state_dim,obs_dim,dt):
        super(KalmanFilterBlock,self).__init__()
        self.state_dim = state_dim
        self.obs_dim = obs_dim
        self.dt = dt
        A = torch.tensor([
                        [1, self.dt, 0,  0,  0,  0],  # dψ/dψ̇ = dt
                        [0,  1, 0,  0,  0,  0],  # ψ̇ constant
                        [0,  0, 1,  0, self.dt,  0],  # vx += ax*dt
                        [0,  0, 0,  1,  0, self.dt],  # vy += ay*dt
                        [0,  0, 0,  0,  1,  0],  # ax constant
                        [0,  0, 0,  0,  0,  1],  # ay constant
                        ], dtype=torch.float32)  # [1, D, D]

        H = torch.tensor([
                            [0, 1, 0, 0, 0, 0],  # ψ̇
                            [0, 0, 1, 0, 0, 0],  # vx
                            [0, 0, 0, 1, 0, 0],  # vy
                            [0, 0, 0, 0, 1, 0],  # ax
                            [0, 0, 0, 0, 0, 1],  # ay
                        ], dtype=torch.float32)
        Q = torch.eye(state_dim) * 1e-3  # [D, D]
        R = torch.eye(obs_dim) * 1e-3    # [O, O]
        I = torch.eye(state_dim).unsqueeze(0)

        self.register_buffer('A', A)
        self.register_buffer('H', H)
        self.register_buffer('Q', Q)
        self.register_buffer('R', R)
        self.register_buffer('I', I)

    def forward(self,x_t_1,P_t_1,z_t):

        B = x_t_1.size(0) #batch dimension
        A = self.A.expand(B, -1, -1)
        H = self.H.expand(B, -1, -1)
        I = self.I.expand(B, -1, -1)

        # Prediction
        x_pred = torch.bmm(A, x_t_1.unsqueeze(-1)).squeeze(-1)
        P_pred = torch.bmm(A, torch.bmm(P_t_1, A.transpose(1, 2))) + self.Q

        # Innovation
        S = torch.bmm(H, torch.bmm(P_pred, H.transpose(1, 2))) + self.R
        S = 0.5 * (S + S.transpose(-1, -2))  # force symmetry
        eps = 1e-6  # jitter for numerical stability
        I_obs = torch.eye(S.size(-1), device=S.device).expand_as(S)
        S = S + eps * I_obs  # regularization

        # Kalman gain via Cholesky solve
        PHt = torch.bmm(P_pred, H.transpose(1, 2))  # [B, D, O]
        L = torch.linalg.cholesky(S)  # [B, O, O]

        K_t = torch.cholesky_solve(PHt.transpose(1, 2), L)  # [B, O, D]
        K_t = K_t.transpose(1, 2)

        # Update
        x_post = x_pred + torch.bmm(K_t, z_t.unsqueeze(-1)).squeeze(-1)  # [B, D]
        P_post = torch.bmm(I - torch.bmm(K_t, H), P_pred)

        #Physical limits
        x_post[:, 0] = x_post[:, 0] % (2 * torch.pi) #Angle wrapping for x_post

        return x_post, P_post

