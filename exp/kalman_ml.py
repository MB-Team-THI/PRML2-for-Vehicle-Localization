from datasets.dataset_loader import KalmanAwareDataset
from datasets.preprocessing import load_config,preprocess_dataset
from models.ml_model import ml_block
from models.kalman_filter import KalmanFilterBlock
from exp.kalman_ml_basic import exp_kalman_ml_basic
from models.only_ml_model import only_ml_block
from utils.tools import EarlyStopping, adjust_learning_rate
from utils.metric import metric
import io
import numpy as np
import matplotlib.pyplot as plt
import torch
from torch.utils.tensorboard import SummaryWriter
import torch.nn as nn
from torch import optim
from torch.utils.data import DataLoader

import os
import time

import warnings

warnings.filterwarnings('ignore')


class Exp_Kalman_ml(exp_kalman_ml_basic):
    def __init__(self, config,args):
        super().__init__(config, args)
        self.config = config
        self.args = args
        self.writer = SummaryWriter()

    def _build_ml_model(self):
        if self.config["model"]["model"] == "ml_block":
            cfg = self.config["model"]
            model = ml_block(
                    enc_in=cfg["enc_in"],
                    seq_len=cfg["seq_len"],
                    enc_out=cfg["enc_out"],
                    dec_x_in=cfg["dec_x_in"],
                    enc_layers=cfg["enc_layers"],
                    dec_layers=cfg["dec_layers"],
                    d_model=cfg["d_model"],
                    n_head=cfg["n_head"],
                    dropout=cfg["dropout"],
                    activation=cfg["activation"],
                )

        if self.args.use_multi_gpu and self.args.use_gpu:
            model = nn.DataParallel(model, device_ids=self.args.device_ids)
        return model

    def _build_only_ml_model(self):
        if self.config["model"]["model"] == "ml_block":
            cfg = self.config["model"]
            model = only_ml_block(
                enc_in=cfg["enc_in"],
                seq_len=cfg["seq_len"],
                enc_out=cfg["enc_out"],
                enc_layers=cfg["enc_layers"],
                d_model=cfg["d_model"],
                n_head=cfg["n_head"],
                dropout=cfg["dropout"],
                activation=cfg["activation"],
            )

        if self.args.use_multi_gpu and self.args.use_gpu:
            model = nn.DataParallel(model, device_ids=self.args.device_ids)
        return model

    def _build_model_kalman(self):
        cfg = self.config["KalmanFilter"]
        kf_model = KalmanFilterBlock(state_dim=cfg["state_dim"], obs_dim=cfg["obs_dim"], dt=0.02)
        if self.args.use_multi_gpu and self.args.use_gpu:
            kf_model = nn.DataParallel(kf_model, device_ids=self.args.device_ids)
        else:
            kf_model = kf_model.to(self.device)
        return kf_model


    def _get_data(self, flag):
        df = preprocess_dataset(self.config)
        data_set = KalmanAwareDataset(df,self.config,flag)
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
        model_optim = optim.Adam(self.ml_model.parameters(), lr=self.config["training"]["learning_rate"])
        return model_optim


    def _select_criterion(self):
        return nn.MSELoss()

    def vali(self, vali_data, vali_loader, criterion):
        self.ml_model.eval()
        total_loss = []
        for i, (batch_x, batch_y, batch_p) in enumerate(vali_loader):
            pred, true = self._process_one_batch(
                vali_data, batch_x, batch_y, batch_p)
            loss = criterion(pred.detach().cpu(), true.detach().cpu())
            total_loss.append(loss)
        total_loss = np.average(total_loss)
        self.ml_model.train()
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
        early_stopping = EarlyStopping(patience=self.args.patience, verbose=True)

        model_optim = self._select_optimizer()
        criterion = self._select_criterion()

        total_params = sum(p.numel() for p in self.ml_model.parameters() if p.requires_grad)
        print(f'Total parameters: {total_params}')

        for epoch in range(self.args.train_epochs):
            iter_count = 0
            train_loss = []
            self.ml_model.train()
            #self.only_ml_model.train()
            self.kalman_filter.train()
            epoch_time = time.time()
            device = torch.device(f"cuda:{self.args.gpu}" if torch.cuda.is_available() else "cpu")
            for i, (batch_x, batch_y, batch_p) in enumerate(train_loader):
                iter_count += 1

                batch_x = batch_x.float().to(device)
                batch_y = batch_y.float().to(device)
                batch_p = batch_p.float().to(device)

                model_optim.zero_grad()
                if epoch < 2:
                    initial_train=True
                    pred, true = self._process_one_batch(train_data, batch_x, batch_y, batch_p,initial_train)
                else:
                    pred, true = self._process_one_batch(train_data, batch_x, batch_y, batch_p, initial_train=False)
                loss = criterion(pred, true)
                self.writer.add_scalar('Loss/train', float(loss.item()), train_iters)
                train_iters += 1
                train_loss.append(loss.item())

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

            print("Epoch: {} cost time: {}".format(epoch + 1, time.time() - epoch_time))
            train_loss = np.average(train_loss)
            vali_loss = self.vali(vali_data, vali_loader, criterion)
            self.writer.add_scalar('Loss/val', float(vali_loss), epoch + 1)
            test_loss = self.vali(test_data, test_loader, criterion)
            self.writer.add_scalar('Loss/test', float(test_loss), epoch + 1)

            print("Epoch: {0}, Steps: {1} | Train Loss: {2:.7f} Vali Loss: {3:.7f} Test Loss: {4:.7f}".format(
                epoch + 1, train_steps, train_loss, vali_loss, test_loss))
            early_stopping(vali_loss, self.model, path)
            if early_stopping.early_stop:
                print("Early stopping")
                break

            adjust_learning_rate(model_optim, epoch + 1, self.args)

        best_model_path = path + '/' + 'checkpoint.pth'
        self.ml_model.load_state_dict(torch.load(best_model_path))
        self.writer.flush()
        self.writer.close()
        return self.ml_model

    # def test(self, setting):
    #     test_data, test_loader = self._get_data(flag='test')
    #     # Use this block if u want to load best model and load results (comment out three lines)
    #     # Be careful with inverse scaler as datasets should be same on which this model was trained
    #     path = os.path.join(self.args.checkpoints, setting)
    #     best_model_path = path + '/' + 'checkpoint.pth'
    #     self.model.load_state_dict(torch.load(best_model_path))
    #
    #     self.model.eval()
    #
    #     preds = []
    #     trues = []
    #     variance = []
    #     start_time = time.time()
    #     for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(test_loader):
    #         pred, true = self._process_one_batch(
    #             test_data, batch_x, batch_y, batch_x_mark, batch_y_mark)
    #         # This part of the code was modified to reproject the data back to slip angles (inversion of the conversion process)
    #         true_rescaled = test_data.inverse_transform((true[:, :, -1])).unsqueeze(-1)  # modified
    #         if self.args.distribution == 'Student-t':
    #             pred_rescaled = test_data.inverse_transform((pred[:, :, 1])).unsqueeze(-1)
    #             variance_rescaled = ((pred[:, :, 2] * pred[:, :, 2]) * (pred[:, :, 0] / (pred[:, :, 0] - 2)) * (
    #                         6.73920 / .96210)).unsqueeze(-1)
    #             variance.append(variance_rescaled.detach().cpu().numpy())
    #         else:
    #             pred_rescaled = test_data.inverse_transform((pred[:, :, -1])).unsqueeze(-1)  # modified
    #         preds.append(pred_rescaled.detach().cpu().numpy())
    #         trues.append(true_rescaled.detach().cpu().numpy())
    #
    #     end_time = time.time()
    #     run_time = (end_time - start_time) / len(test_loader)
    #     print(run_time)
    #     preds = np.array(preds)
    #     trues = np.array(trues)
    #     if self.args.distribution == 'Student-t':
    #         variance = np.array(variance)
    #         variance = variance.reshape(-1, variance.shape[-2], variance.shape[-1])
    #     print('test shape:', preds.shape, trues.shape)
    #     preds = preds.reshape(-1, preds.shape[-2], preds.shape[-1])
    #     trues = trues.reshape(-1, trues.shape[-2], trues.shape[-1])
    #     print('test shape:', preds.shape, trues.shape)
    #
    #     # Result save
    #     folder_path = './results/' + setting + '/'
    #     if not os.path.exists(folder_path):
    #         os.makedirs(folder_path)
    #
    #     mae, mse, rmse, mape, mspe, max_ae = metric(preds, trues)
    #     print('mse:{}, mae:{},max_ae:{}'.format(mse, mae, max_ae))
    #
    #     np.save(folder_path + 'metrics.npy', np.array([mae, mse, rmse, mape, mspe]))
    #     np.save(folder_path + 'pred.npy', preds)
    #     np.save(folder_path + 'true.npy', trues)
    #     if self.args.distribution == 'Student-t':
    #         np.save(folder_path + 'variance_pred.npy', variance)
    #     return


    def _process_one_batch(self, dataset, batch_x, batch_y, batch_p, initial_train):
        window_length = self.config["data"]["window_size"]
        history_size = self.config["data"]["history_size"]

        x_t_1 = batch_p[:,history_size-1,:]
        residual_x_t_1 = batch_p[:,history_size-1,:]*0
        batch_size = x_t_1.shape[0]
        state_dim = x_t_1.shape[1]
        init_var = 1e-4
        p_t_1 = torch.eye(state_dim, device=x_t_1.device).unsqueeze(0).repeat(batch_size, 1, 1) * init_var
        x_posteriors = []
        if initial_train:
            z_t,K_t = self.ml_model(batch_x[:, 0:history_size, :],residual_x_t_1)
            return dataset.scaler_target.transform(z_t), batch_y[:, history_size]
        else:
            for i in range(0,window_length - history_size):
                z_t, K_t = self.ml_model(batch_x[:,i:i+history_size,:],residual_x_t_1)#p_t_1
                #print(i,residual_x_t_1[0,:], x_t_1[0,:], z_t[0,:],K_t[0,:])
                #z_t_scaled = dataset.inverse_target(z_t)
                #x_t, p_t = self.kalman_filter(x_t_1,p_t_1,z_t,K_t)
                x_t,residual = self.kalman_filter(x_t_1,p_t_1,z_t,K_t)
                x_posteriors.append(x_t.unsqueeze(1))
                x_t_1 = x_t
                residual_x_t_1 = residual
                #p_t_1 = p_t
            return torch.cat(x_posteriors, dim=1), batch_p[:,history_size::,:]

    def _process_one_batch_ML(self, dataset, batch_x, batch_y, batch_p):
        window_length = self.config["data"]["window_size"]
        history_size = self.config["data"]["history_size"]
        z_t= self.only_ml_model(batch_x[:,0:history_size,:])
        return z_t, batch_y[:,history_size,:]





# The dataset object contains the statistics of the entire dataset and the scaler transform carried out in it
# We can use it to inverse the scaler transform and get actual values.
# This is implemented for test function.