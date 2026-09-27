import torch
import torch.nn as nn
class EKFBlock(nn.Module):
    def __init__(self, state_dim, obs_dim, dt, process_noise=1e-3, meas_noise=1e-3):
        super(EKFBlock, self).__init__()
        self.state_dim = state_dim
        self.obs_dim = obs_dim
        self.dt = dt
        self.register_buffer('Q', torch.eye(state_dim) * process_noise)
        self.register_buffer('R', torch.eye(obs_dim) * meas_noise) #yaw rate from OBD is prediction

    def state_function(self, x):

        x_pos, y_pos, yaw, yaw_rate, vx, vy, ax, ay = x[:,0], x[:,1], x[:,2], x[:,3], x[:,4], x[:,5], x[:,6], x[:,7]

        cos_yaw = torch.cos(yaw)
        sin_yaw = torch.sin(yaw)

        # Rotation matrices batch-wise
        vel_global_x = cos_yaw * vx - sin_yaw * vy
        vel_global_y = sin_yaw * vx + cos_yaw * vy

        acc_global_x = cos_yaw * ax - sin_yaw * ay
        acc_global_y = sin_yaw * ax + cos_yaw * ay

        # Update positions
        x_new = x.clone()
        x_new[:,0] = x_pos + vel_global_x * self.dt + 0.5 * acc_global_x * self.dt**2
        x_new[:,1] = y_pos + vel_global_y * self.dt+ 0.5 * acc_global_y * self.dt**2

        # Yaw and yaw rate
        x_new[:,2] = yaw + yaw_rate * self.dt
        x_new[:,3] = yaw_rate

        # Velocity updates in vehicle frame
        x_new[:,4] = vx + ax * self.dt
        x_new[:,5] = vy + ay * self.dt

        # Accelerations assumed constant
        x_new[:,6] = ax
        x_new[:,7] = ay
        return x_new

    def F_batch(self, x):
        """
        Analytic Jacobian of state_function, batch-wise
        x: [B, D]
        returns: [B, D, D]
        """
        B = x.shape[0]
        yaw = x[:, 2]
        dt = self.dt
        cos_yaw = torch.cos(yaw)
        sin_yaw = torch.sin(yaw)

        F = torch.eye(self.state_dim, device=x.device).unsqueeze(0).repeat(B, 1, 1)

        # dpos/dvx, dpos/dvy, dpos/dax, dpos/day
        F[:, 0, 4] = cos_yaw * dt
        F[:, 0, 5] = -sin_yaw * dt
        F[:, 0, 6] = 0.5 * cos_yaw * dt ** 2
        F[:, 0, 7] = -0.5 * sin_yaw * dt ** 2

        F[:, 1, 4] = sin_yaw * dt
        F[:, 1, 5] = cos_yaw * dt
        F[:, 1, 6] = 0.5 * sin_yaw * dt ** 2
        F[:, 1, 7] = 0.5 * cos_yaw * dt ** 2

        # dpos/dyaw
        vx = x[:, 4]
        vy = x[:, 5]
        ax = x[:, 6]
        ay = x[:, 7]
        F[:, 0,
        2] = -sin_yaw * vx * dt - cos_yaw * vy * dt - 0.5 * sin_yaw * ax * dt ** 2 + 0.5 * cos_yaw * ay * dt ** 2
        F[:, 1, 2] = cos_yaw * vx * dt - sin_yaw * vy * dt + 0.5 * cos_yaw * ax * dt ** 2 + 0.5 * sin_yaw * ay * dt ** 2

        # dvel/dax, dvel/day
        F[:, 4, 6] = dt
        F[:, 5, 7] = dt

        # yaw update
        F[:, 2, 3] = dt
        return F

    def h_batch(self, x):

        return x[:, [3, 4, 5, 6, 7]]

    def H_batch(self, x):
        """
        Analytic Jacobian of h_batch: z = [yaw_rate, vx, vy, ax, ay]
        returns: [B, O, D]
        """
        B = x.shape[0]
        H = torch.zeros(B, self.obs_dim, self.state_dim, device=x.device)
        H[:, 0, 3] = 1.0  # yaw_rate
        H[:, 1, 4] = 1.0  # vx
        H[:, 2, 5] = 1.0  # vy
        H[:, 3, 6] = 1.0  # ax
        H[:, 4, 7] = 1.0  # ay
        return H

    def batch_jacobian(self, func, x):
        """
        Compute Jacobians for batch: x [B, D], returns [B, M, D]
        """
        B, D = x.shape
        x = x.requires_grad_(True)
        # Compute function output once for shape
        y = func(x)
        M = y.shape[1]
        J = torch.zeros(B, M, D, device=x.device)

        # Loop over output dimensions only (usually small, e.g., 5)
        for i in range(M):
            grad_outputs = torch.zeros_like(y)
            grad_outputs[:, i] = 1.0
            grads = torch.autograd.grad(outputs=y, inputs=x, grad_outputs=grad_outputs,
                                        retain_graph=True, create_graph=True)[0]
            J[:, i, :] = grads
        return J

    def wrap_to_minus2pi_0(self,yaw):
        return (yaw % (2 * torch.pi)) - 2 * torch.pi

    def forward(self, x_t_1, P_t_1, z_t):
        B = x_t_1.size(0)

        # ---- Prediction ----
        x_pred = self.state_function(x_t_1)
        F = self.F_batch(x_t_1)    # [B, D, D]
        P_pred = torch.bmm(F, torch.bmm(P_t_1, F.transpose(1, 2))) + self.Q.to(x_t_1.device)

        # ---- Measurement update ----
        z_pred = self.h_batch(x_pred)
        H = self.H_batch(x_pred)    # [B, O, D]

        S = torch.bmm(H, torch.bmm(P_pred, H.transpose(1, 2))) + self.R.to(x_t_1.device)
        S = 0.5 * (S + S.transpose(-1, -2)) + 1e-6 * torch.eye(self.obs_dim, device=S.device).expand_as(S)

        # Kalman gain
        PHt = torch.bmm(P_pred, H.transpose(1, 2))
        L = torch.linalg.cholesky(S)
        K_t = torch.cholesky_solve(PHt.transpose(1, 2), L).transpose(1, 2)

        # Update
        y = z_t - z_pred
        x_post = x_pred + torch.bmm(K_t, y.unsqueeze(-1)).squeeze(-1)
        x_post[:, 2] = self.wrap_to_minus2pi_0(x_post[:, 2])
        P_post = torch.bmm(torch.eye(self.state_dim, device=x_t_1.device).expand_as(P_pred) - torch.bmm(K_t, H), P_pred)
        return x_post, P_post


