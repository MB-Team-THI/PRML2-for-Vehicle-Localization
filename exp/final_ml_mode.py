from datasets.dataset_loader import KalmanAwareDataset
from datasets.preprocessing import load_config, preprocess_dataset
from models.ml_model import ml_block
from models.kalman_filter import KalmanFilterBlock
from models.extended_kalman_filter import EKFBlock
from exp.kalman_ml_basic import exp_kalman_ml_basic
from models.final_ml_model_w_uncertainty import final_ml_block
from utils.tools import EarlyStopping, adjust_learning_rate
from utils.metric import metric
from transformers import get_cosine_schedule_with_warmup
import torch.distributions as dist
import io
import numpy as np
from collections import OrderedDict
import matplotlib.pyplot as plt
import torch
from torch.utils.tensorboard import SummaryWriter
import torch.nn as nn
from torch import optim
from torch.utils.data import DataLoader
from sklearn.preprocessing import StandardScaler

import os
import time

import warnings

warnings.filterwarnings('ignore')



class final_ml_model(exp_kalman_ml_basic):
    def __init__(self, config, args):
        super().__init__(config, args)
        self.config = config
        self.args = args
        self.writer = SummaryWriter()

    def _final_ml_model(self):
        cfg = self.config["model"]
        model = final_ml_block(
            enc_in=cfg["enc_in"],
            dec_in=cfg["dec_in"],
            seq_len_enc=cfg["seq_len_enc"],
            seq_len_dec=cfg["seq_len_dec"],
            enc_out=cfg["enc_out"],
            enc_layers=cfg["enc_layers"],
            dec_layers=cfg["dec_layers"],
            d_model=cfg["d_model"],
            n_head=cfg["n_head"],
            dropout=cfg["dropout"],
            a_max = cfg["a_max"],
            v_max = cfg["v_max"],
            activation=cfg["activation"],
        )

        if self.args.use_multi_gpu and self.args.use_gpu:
            model = nn.DataParallel(model, device_ids=self.args.device_ids)
        return model

    def _build_model_kalman(self):
        cfg = self.config["KalmanFilter"]
        ekf_model = EKFBlock(state_dim=cfg["state_dim"], obs_dim=cfg["obs_dim"], dt=0.02)
        if self.args.use_multi_gpu and self.args.use_gpu:
            ekf_model = nn.DataParallel(ekf_model, device_ids=self.args.device_ids)
        else:
            ekf_model = ekf_model.to(self.device)
        return ekf_model

    def _get_data(self, flag):
        df = preprocess_dataset(self.config)
        data_set = KalmanAwareDataset(df, self.config, flag)
        print(flag, len(data_set))

        if flag == 'test' or flag == "val":
            shuffle_flag = False;
            drop_last = True;
            batch_size = self.args.batch_size;

        else:
            shuffle_flag = True;
            drop_last = True;
            batch_size = self.args.batch_size;

        data_loader = DataLoader(
            data_set,
            batch_size=batch_size,
            shuffle=shuffle_flag,
            num_workers=self.args.num_workers,
            drop_last=drop_last)

        return data_set, data_loader

    def _select_optimizer(self):
        model_optim = optim.Adam(self.final_ml_model.parameters(), lr=self.config["training"]["learning_rate"])
        return model_optim

    def _select_criterion(self):
        return nn.MSELoss()

    def _nll_criterion(self):
        def nll_loss(mean, cov, target):
            """
            mean:  [B, T, D]
            cov:   [B, T, D, D]  (positive definite)
            target: [B, T, D]
            """
            B, T, D = mean.shape

            # flatten batch and time for distributions
            mean_flat = mean.reshape(B*T, D)
            cov_flat  = cov.reshape(B*T, D, D)
            target_flat = target.reshape(B*T, D)

            # Add small jitter for numerical stability
            eps = 1e-3
            cov_flat = cov_flat + torch.eye(D, device=cov.device).unsqueeze(0) * eps

            # Define multivariate normal
            mvn = dist.MultivariateNormal(loc=mean_flat, covariance_matrix=cov_flat)

            # Negative log likelihood
            nll = -mvn.log_prob(target_flat)  # [B*T]
            return nll.mean()

        return nll_loss

    def _nll_criterion_physics(self):
        def nll_loss(mean, cov, target, dataset):
            """
            mean:  [B, T, D]
            cov:   [B, T, D, D]  (positive definite)
            target: [B, T, D]
            """
            nll_loss._loss_type = "physics"
            B, T, D = mean.shape

            # flatten batch and time for distributions
            mean_flat = mean.reshape(B*T, D)
            cov_flat  = cov.reshape(B*T, D, D)
            target_flat = target.reshape(B*T, D)

            # Add small jitter for numerical stability
            eps = 1e-3
            cov_flat = cov_flat + torch.eye(D, device=cov.device).unsqueeze(0) * eps

            # Define multivariate normal
            mvn = dist.MultivariateNormal(loc=mean_flat, covariance_matrix=cov_flat)

            # Negative log likelihood
            nll = -mvn.log_prob(target_flat)  # [B*T]


            #Physics based losses
            mean_rescaled = dataset.scaler_target.inverse_transform((mean))
            vx, vy, ax, ay = mean_rescaled[..., 0], mean_rescaled[..., 1], mean_rescaled[..., 2], mean_rescaled[..., 3]
            dt = 0.02
            # (a) Velocity derivative ≈ acceleration consistency
            dvx_dt = (vx[:, 2:] - vx[:, :-2]) / (2*dt)
            dvy_dt = (vy[:, 2:] - vy[:, :-2]) / (2*dt)
            ax_pred = ax[:, 1:-1]
            ay_pred = ay[:, 1:-1]
            consistency_loss = ((dvx_dt - ax_pred) ** 2 + (dvy_dt - ay_pred) ** 2).mean()
            consistency_loss = consistency_loss * (dt**2)

            # (b) Jerk penalty (smooth accelerations)
            jerk_loss = ((ax[:, 1:] - ax[:, :-1]) ** 2 + (ay[:, 1:] - ay[:, :-1]) ** 2).mean()
            λ_phys = 0.5
            λ_jerk = 0.5
            #print(nll.mean() ,λ_phys * consistency_loss,λ_jerk * jerk_loss)
            return nll.mean() + λ_phys * consistency_loss + λ_jerk * jerk_loss
        return nll_loss

    def _nll_criterion_kalman(self):
        def nll_loss(mean,cov,target,pose,dataset):
            """
            mean:  [B, T, D]
            cov:   [B, T, D, D]  (positive definite)
            target: [B, T, D]
            """
            B, T, D = mean.shape
            nll_loss._loss_type = "kalman"


            #Pose loss implementation (First 4 values are x,y,yaw and yaw rate)
            mean_pose = mean[...,:3]
            mean_pose_minus_init = mean_pose - mean_pose[:,0:1,:]
            gt_pose = pose[:,150:,:3]
            gt_pose_minus_init = gt_pose - gt_pose[:,0:1,:]
            #scaling
            scaler_pose_mean =  torch.tensor([5.8357, 0.2822,0], device=mean_pose.device)  # shape [2]
            scaler_pose_std = torch.tensor([1.0488e+01, 5.0725e+00,-np.pi/8], device=mean_pose.device)

            mean_pose_scaled = (mean_pose_minus_init - scaler_pose_mean) / scaler_pose_std
            gt_pose_scaled = (gt_pose_minus_init - scaler_pose_mean) / scaler_pose_std

            pose_loss_criterion = nn.MSELoss(reduction="sum")
            pose_loss = pose_loss_criterion(mean_pose_scaled,gt_pose_scaled)
            pose_loss = 10* torch.tanh(pose_loss / 10)
            # flatten batch and time for distributions
            mean_scaled = dataset.scaler_target.transform(mean[:,:,4:]) #Converting the vx,vy,ax,ay to normalized space
            B, T, D = mean_scaled.shape
            mean_flat = mean_scaled.reshape(B*T,D)
            cov_flat  = cov.reshape(B*T,D,D)
            target_flat = target.reshape(B*T,D)


            # Add small jitter for numerical stability
            eps = 1e-3
            cov_flat = cov_flat + torch.eye(D, device=cov.device).unsqueeze(0) * eps

            # Define multivariate normal
            mvn = dist.MultivariateNormal(loc=mean_flat, covariance_matrix=cov_flat)

            # Negative log likelihood
            nll = -mvn.log_prob(target_flat)  # [B*T]
            #print("nll",nll.mean(),"pose_loss",pose_loss)
            return nll.mean() + pose_loss
        return nll_loss

    def _nll_criterion_position(self):
        def nll_loss(mean, cov, target,pose):
            """
            mean:  [B, T, D]
            cov:   [B, T, D, D]  (positive definite)
            target: [B, T, D]
            """
            B, T, D = mean.shape

            # flatten batch and time for distributions
            mean_flat = mean.reshape(B*T, D)
            cov_flat  = cov.reshape(B*T, D, D)

            #Pose transformation for vehicle frame.
            pose = pose[:,150:,:]
            c = torch.cos(pose[:, 0, 2])
            s = torch.sin(pose[:, 0, 2])
            z = torch.zeros_like(s)
            o = torch.ones_like(s)
            gTv0 = torch.stack([
                torch.stack([c, -s, pose[:, 0, 0]], axis=1),
                torch.stack([s, c, pose[:, 0, 1]], axis=1),
                torch.stack([z, z, o], axis=1),
            ], axis=1).float().to(pose.device)  # N x 100 x

            #yaw_diff_basic = target[..., 2:3] - target[:, 0:1, 2:3]
            #yaw_difference = torch.remainder(yaw_diff_basic + math.pi, 2 * math.pi) - math.pi

            gPv = torch.stack([pose[..., 0], pose[..., 1], torch.ones_like(pose[..., 0])], axis=1)

            v0Pv = (torch.linalg.inv(gTv0) @ gPv).permute(0, 2, 1)[..., :2]

            euc_norm = torch.norm(v0Pv, dim=-1)
            velocities = torch.gradient(euc_norm, dim=-1, spacing=0.02)[0]

            keys = velocities < 35
            delta_true = v0Pv #torch.concat([v0Pv, yaw_difference], axis=2)

            # delta_x = torch.where((delta_x > 50) | (delta_x < -50), torch.tensor(0.0, device=delta_x.device), delta_x) #removes outliers in dataset concatenation
            # delta_y = torch.where((delta_y > 50) | (delta_y < -50), torch.tensor(0.0, device=delta_y.device), delta_y) #removes outliers in dataset concatenation

            # delta_true = torch.stack([delta_x, delta_y], dim=-1)

            # mean = torch.tensor([0.00769, 0.02241], device=delta_true.device)  # shape [2]
            # std = torch.tensor([4.10385, 6.02158], device=delta_true.device)

            mean = torch.tensor([5.8357, 0.2822], device=delta_true.device)  # shape [2]
            std = torch.tensor([1.0488e+01, 5.0725e+00], device=delta_true.device)

            delta_true_norm = (delta_true - mean) / std
            ################################################################pose transformation over for loss
            keys = keys.reshape(B*T)
            combined = torch.cat([delta_true_norm, target], dim=-1)
            combined_flat = combined.reshape(B*T, D)

            # Add small jitter for numerical stability
            eps = 1e-3
            cov_flat = cov_flat + torch.eye(D, device=cov.device).unsqueeze(0) * eps

            # Define multivariate normal
            mvn = dist.MultivariateNormal(loc=mean_flat[keys], covariance_matrix=cov_flat[keys])

            # Negative log likelihood
            nll = -mvn.log_prob(combined_flat[keys])  # [B*T]
            return nll.mean()
        return nll_loss


    def _physics_loss_criterion(self, dt=0.02, w_sup=1.0, w_kin=1.0, w_yaw=1.0):
        """Combined Physics-aware Loss: MSE,Kinematic body frame consistency, yaw rate integration consistency"""

        def loss_fn(preds, target, data, cospsi=None, sinpsi=None):
            diff_sq = (preds - target) ** 2
            sup_loss = (diff_sq).mean()

            # Physics losses
            preds_unscaled = data.scaler_target.inverse_transform((preds[:, :]))
            r = preds_unscaled[..., 0]
            vx = preds_unscaled[..., 1]
            vy = preds_unscaled[..., 2]
            ax = preds_unscaled[..., 3]
            ay = preds_unscaled[..., 4]

            # shift in time (t → t+1)
            vx_t, vx_tp1 = vx[:, :-1], vx[:, 1:]
            vy_t, vy_tp1 = vy[:, :-1], vy[:, 1:]
            r_t = r[:, :-1]
            ax_t = ax[:, :-1]
            ay_t = ay[:, :-1]

            # expected updates from rigid-body kinematics
            vx_pred = vx_t + dt * (ax_t + r_t * vy_t)
            vy_pred = vy_t + dt * (ay_t - r_t * vx_t)

            # kinematic loss = MSE of residuals
            kin_loss = ((vx_tp1 - vx_pred) ** 2 + (vy_tp1 - vy_pred) ** 2).mean()

            # Yaw loss
            yaw_pred_angle = torch.sum(preds[:, :, 0], dim=1) * dt
            yaw_true_angle = torch.sum(target[:, :, 0], dim=1) * dt

            yaw_loss = torch.mean((yaw_pred_angle - yaw_true_angle) ** 2)

            total_loss = w_sup * sup_loss + w_kin * kin_loss + w_yaw * yaw_loss
            return total_loss, {"sup": sup_loss.item(),
                                "kin": kin_loss.item(),
                                "yaw": float(yaw_loss) if isinstance(yaw_loss, torch.Tensor) else 0.0}

        return loss_fn

    def vali(self, vali_data, vali_loader, criterion):
        self.final_ml_model.eval()
        total_loss = []
        print("validation",criterion)
        with torch.no_grad():
            for i, (batch_x, batch_y, batch_p) in enumerate(vali_loader):
                device = torch.device(f"cuda:{self.args.gpu}" if torch.cuda.is_available() else "cpu")
                batch_x = batch_x.float().to(device)
                batch_y = batch_y.float().to(device)
                batch_p = batch_p.float().to(device)
                if getattr(criterion, "_loss_type", None) == "physics":
                    pred, var, true = self._process_one_batch_ML(vali_data, batch_x, batch_y, batch_p)
                    loss = criterion(pred.detach(), var.detach(), true.detach(), vali_data)
                else:
                    pred, var, true = self._process_one_batch_ML_kalman(vali_data, batch_x, batch_y, batch_p)
                    loss = criterion(pred.detach(), var.detach(), true.detach(), batch_p.detach(), vali_data)
                total_loss.append(loss.item())
        total_loss = np.average(total_loss)
        self.final_ml_model.train()
        return total_loss

    def train(self, setting):
        torch.autograd.set_detect_anomaly(True)
        train_data, train_loader = self._get_data(flag='train')
        vali_data, vali_loader = self._get_data(flag='val')
        test_data, test_loader = self._get_data(flag='test')
        train_iters = 0  # Variable created for logging.
        path = os.path.join(self.args.checkpoints, setting)
        if not os.path.exists(path):
            os.makedirs(path)
        if self.args.use_amp:
            scaler = torch.cuda.amp.GradScaler()
        time_now = time.time()

        train_steps = len(train_loader)
        early_stop_physics = EarlyStopping(patience=self.args.patience, verbose=True, name="physics_best")
        early_stop_finetune = EarlyStopping(patience=self.args.patience, verbose=True, name="finetune_best")
        model_optim = self._select_optimizer()
        if self.args.lradj == 'type3':
            total_steps = len(train_loader) * self.args.train_epochs
            scheduler = get_cosine_schedule_with_warmup(
                model_optim,
                num_warmup_steps=500,
                num_training_steps=total_steps
            )
        else:
            scheduler = None

        print("Check loss implementation - nll")
        criterion = self._nll_criterion_physics() #self._physics_loss_criterion_v2()  #  #self._weighted_select_criterion()

        total_params = sum(p.numel() for p in self.final_ml_model.parameters() if p.requires_grad)
        print(f'Total parameters: {total_params}')

        for epoch in range(self.args.train_epochs):
            iter_count = 0
            train_loss = []
            self.final_ml_model.train()
            epoch_time = time.time()
            device = torch.device(f"cuda:{self.args.gpu}" if torch.cuda.is_available() else "cpu")
            for i, (batch_x, batch_y, batch_p) in enumerate(train_loader):

                iter_count += 1

                batch_x = batch_x.float().to(device)
                batch_y = batch_y.float().to(device)
                batch_p = batch_p.float().to(device)

                model_optim.zero_grad()
                if epoch < int(0.8 * self.args.train_epochs):
                    criterion = self._nll_criterion_physics()
                    pred, var, true = self._process_one_batch_ML(train_data, batch_x, batch_y, batch_p)
                    loss = criterion(pred,var,true,train_data)
                else:
                    criterion = self._nll_criterion_kalman()
                    pred, var, true = self._process_one_batch_ML_kalman(train_data, batch_x, batch_y, batch_p)
                    loss = criterion(pred, var, true, batch_p, train_data)

                #loss = criterion(pred, true, train_data)
                #self.writer.add_scalar('Loss/train', float(loss[0].item()), train_iters)
                #self.writer.add_scalar('Loss/sup', loss[1]['sup'], train_iters)
                #self.writer.add_scalar('Loss/kin', loss[1]['kin'], train_iters)
                #self.writer.add_scalar('Loss/yaw', loss[1]['yaw'], train_iters)
                train_iters += 1
                #train_loss.append(loss[0].item())

                if (i + 1) % 10 == 0:
                    print("\titers: {0}, epoch: {1} | loss: {2:.7f}".format(i + 1, epoch + 1, loss.item()))
                    speed = (time.time() - time_now) / iter_count
                    left_time = speed * ((self.args.train_epochs - epoch) * train_steps - i)
                    print('\tspeed: {:.4f}s/iter; left time: {:.4f}s'.format(speed, left_time))
                    iter_count = 0
                    time_now = time.time()

                if self.args.use_amp:
                    scaler.scale(loss).backward()
                    scaler.step(model_optim)
                    scaler.update()
                else:
                    loss.backward()
                    model_optim.step()
                if scheduler is not None:
                    scheduler.step()

            print("Epoch: {} cost time: {}".format(epoch + 1, time.time() - epoch_time))
            train_loss = np.average(train_loss)

            del batch_x, batch_y, batch_p, pred, var, true, loss
            torch.cuda.empty_cache()

            vali_loss = self.vali(vali_data, vali_loader, criterion)
            self.writer.add_scalar('Loss/val', float(vali_loss), epoch + 1)
            #test_loss = self.vali(test_data, test_loader, criterion)
            #self.writer.add_scalar('Loss/test', float(test_loss), epoch + 1)
            #print("Epoch: {0}, Steps: {1} | Train Loss: {2:.7f} Vali Loss: {3:.7f} Test Loss: {4:.7f}".format(
               # epoch + 1, train_steps, train_loss, vali_loss, test_loss))
            if epoch < int(0.8 * self.args.train_epochs):
                early_stop_physics(vali_loss, self.final_ml_model, path,allow_stop=False)
            else:
                early_stop_finetune(vali_loss, self.final_ml_model, path,allow_stop=True)
                if early_stop_finetune.early_stop:
                    print("Early stopping")
                    break

            if self.args.lradj in ['type1', 'type2']:
                adjust_learning_rate(model_optim, epoch + 1, self.args)

        best_model_path = path + '/' + 'checkpoint.pth'
        self.final_ml_model.load_state_dict(torch.load(best_model_path))
        self.writer.flush()
        self.writer.close()
        return self.final_ml_model

    # During testing there was error for multi GPU training saved model to single GPU inference
    # I think error is due to model not running properly in loaded mode. Single GPU inference is best
    # This script loads model in single GPU and gets results
    def test(self, setting, config,type):
        test_data, test_loader = self._get_data(flag='test')
        # Use this block if u want to load best model and load results (comment out three lines)
        # Be careful with inverse scaler as datasets should be same on which this model was trained
        print(f'Testing... {type}')
        path = os.path.join(self.args.checkpoints, setting)
        if type=="fine_tune":
            best_model_path = path + '/' + 'finetune_best.pth'
        elif type=="physics_best":
            best_model_path = path + '/' + 'physics_best.pth'
        else:
            best_model_path = path + '/' + 'checkpoint.pth'
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        state_dict = torch.load(best_model_path, map_location=device)
        history_size = config["data"]['history_size']
        # unwrapping data parrallel.
        if isinstance(self.final_ml_model, torch.nn.DataParallel):
            self.final_ml_model = self.final_ml_model.module
        else:
            self.final_ml_model = self.final_ml_model
        new_state_dict = OrderedDict()
        for k, v in state_dict.items():
            if k.startswith('module.'):
                name = k[7:]  # remove 'module.' prefix
            else:
                name = k
            new_state_dict[name] = v

        # Load the cleaned state dict
        self.final_ml_model.load_state_dict(new_state_dict)

        # Move the model to single GPU device
        device = torch.device('cuda:1' if torch.cuda.is_available() else 'cpu')
        self.final_ml_model.to(device)

        # Set to evaluation mode
        self.final_ml_model.eval()
        with torch.no_grad():
            preds = []
            trues = []
            gt_poses = []
            variances = []
            start_time = time.time()
            total_batches = len(test_loader)
            percent_step = max(1, total_batches // 20)
            for i, (batch_x, batch_y, batch_p) in enumerate(test_loader):
                if (i + 1) % percent_step == 0 or i == 0:
                    print(f"Progress: {100 * (i + 1) / total_batches:.1f}% ({i + 1}/{total_batches} batches)")
                batch_x = batch_x.float().to(device)
                batch_y = batch_y.float().to(device)
                batch_p = batch_p.float().to(device)
                if type=="fine_tune":
                    pred, variance, true = self._process_one_batch_ML_kalman(
                        test_data, batch_x, batch_y, batch_p)
                else:
                    pred, variance, true = self._process_one_batch_ML(
                        test_data, batch_x, batch_y, batch_p)

                # plt.plot(pred[0,:,2].cpu().numpy())
                # plt.plot(true[0,:,0].cpu().numpy())
                # plt.show(block=True)
                # This part of the code was modified to reproject the data back to slip angles (inversion of the conversion process)
                true_rescaled = test_data.scaler_target.inverse_transform((true[:, -1, :])).unsqueeze(-1)  # modified
                if torch.isnan(pred).any():
                    print("error")
                if pred.size(-1) == 6:
                    means_pos = np.asarray([5.8357, 0.2822])
                    std_pos  = np.asarray([1.0488e+01, 5.0725e+00])
                    test_data.scaler_target.mean = np.concatenate([means_pos, test_data.scaler_target.mean])
                    test_data.scaler_target.std = np.concatenate([std_pos, test_data.scaler_target.std])
                if pred.size(-1) == 8:
                    pred_rescaled = pred[:,-1,4:].unsqueeze(-1)
                else:
                    pred_rescaled = test_data.scaler_target.inverse_transform((pred[:, -1, :])).unsqueeze(-1)  # modified
                pose = batch_p[:, -1, :].unsqueeze(-1)
                variance = variance[:, -1, :,:].unsqueeze(-1)
                preds.append(pred_rescaled.detach().cpu().numpy())
                trues.append(true_rescaled.detach().cpu().numpy())
                variances.append(variance.detach().cpu().numpy())
                gt_poses.append(pose.detach().cpu().numpy())
            end_time = time.time()
            run_time = (end_time - start_time) / len(test_loader)
            print(run_time)
            preds = np.array(preds)
            trues = np.array(trues)
            variances = np.array(variances)
            gt_poses = np.array(gt_poses)
            print('test shape:', preds.shape, trues.shape)
            preds = preds.reshape(-1, preds.shape[-2], preds.shape[-1]).squeeze(-1)
            trues = trues.reshape(-1, trues.shape[-2], trues.shape[-1]).squeeze(-1)
            variances = variances.reshape(-1, 4, 4)
            gt_poses = gt_poses.reshape(-1, gt_poses.shape[-2], gt_poses.shape[-1]).squeeze(-1)
            print('test shape:', preds.shape, trues.shape, gt_poses.shape)

            # Result save
            folder_path = './results/' + setting + '/'
            if not os.path.exists(folder_path):
                os.makedirs(folder_path)

            mae, mse, rmse, mape, mspe, max_ae = metric(preds[:,:], trues)
            print('mse:{}, mae:{},max_ae:{}'.format(mse, mae, max_ae))

            np.save(folder_path + f'metrics_{type}.npy', np.array([mae, mse, rmse, mape, mspe]))
            np.save(folder_path + f'pred_{type}.npy', preds)
            np.save(folder_path + f'true_{type}.npy', trues)
            np.save(folder_path + f'varaince_{type}.npy', variances)
            np.save(folder_path + f'gt_poses_{type}.npy', gt_poses)
            return

    def _process_one_batch_ML(self, dataset, batch_x, batch_y, batch_p):
        window_length = self.config["data"]["window_size"]
        history_size = self.config["data"]["history_size"]
        z_t,var = self.final_ml_model(batch_x[:, 0:history_size, :], batch_x[:, history_size:window_length, :],dataset.scaler_target)
        return z_t, var, batch_y[:, history_size:window_length, :]

    def _process_one_batch_ML_kalman(self, dataset, batch_x, batch_y, batch_p):
        window_length = self.config["data"]["window_size"]
        history_size = self.config["data"]["history_size"]
        z_t,variance = self.final_ml_model(batch_x[:, 0:history_size, :], batch_x[:, history_size:window_length, :],dataset.scaler_target)
        z_t = dataset.scaler_target.inverse_transform((z_t[:, :]))
        x_t_1 = batch_p[:,history_size - 1, :-1] #-1 removes the OBD yaw rate data
        z_t = torch.cat([batch_p[:,history_size:, 3:4],z_t],dim=-1)
        # x_t_1 = batch_y[:, history_size - 1, :] commented out for normal kalman filter
        # yaw0 = torch.zeros(x_t_1.size(0), 1, device=x_t_1.device)
        # x_t_1 = torch.cat([yaw0, x_t_1], dim=1)
        batch_size = x_t_1.shape[0]
        state_dim = x_t_1.shape[1]
        init_var = 1e-5
        p_t_1 = torch.eye(state_dim, device=x_t_1.device).unsqueeze(0).repeat(batch_size, 1, 1) * init_var
        x_posteriors = []
        for i in range(0, window_length - history_size):
            x_t, residual = self.kalman_filter(x_t_1, p_t_1, z_t[:,i])
            x_posteriors.append(x_t[:,:].unsqueeze(1))
            x_t_1 = x_t
        return torch.cat(x_posteriors, dim=1), variance, batch_y[:, history_size:window_length, :]

# The dataset object contains the statistics of the entire dataset and the scaler transform carried out in it
# We can use it to inverse the scaler transform and get actual values.
# This is implemented for test function.
