# -*- coding: utf-8 -*-
# ---------------------

from time import time
import numpy as np
import torch
from torch import optim
from torch.utils.data import DataLoader
from conf import Conf
from dataset.ts_dataset import TSDataset
from models.semantic_fusion_ts import selftf_model
from progress_bar import ProgressBar
from utils import QuantileLoss, MSELoss, symmetric_mean_absolute_percentage_error, unnormalize_tensor, plot_temporal_serie
import data_formatters.utils as utils
from models.transformer import Transformer
from models.transformer_grn.transformer import Transformer as GRNTransformer
import pdb
import os
from datetime import datetime
from pathlib import Path

class TS(object):
    """
    Class for loading and test the pre-trained model
    """

    def __init__(self, cnf):
        # type: (Conf) -> Trainer

        self.cnf = cnf
        self.data_formatter = utils.make_data_formatter(cnf.ds_name)

        loader = TSDataset
        dataset_test = loader(self.cnf, self.data_formatter)
        dataset_test.test()

        category_to_vectors = dataset_test.category_to_vectors

        # init model
        model_choice = self.cnf.all_params["model"]
        #pdb.set_trace()
        if model_choice == "transformer":
            # Baseline transformer
            self.model = Transformer(self.cnf.all_params)
        elif model_choice == "selftf_transformer":
            # Temporal fusion transformer
            self.model = selftf_model.SELFTF(self.cnf.all_params, category_to_vectors)
        elif model_choice == "grn_transformer":
            # Transformer + GRN to encode static vars
            self.model = GRNTransformer(self.cnf.all_params)
        else:
            raise NameError

        self.model = self.model.to(cnf.device)

        # init optimizer
        #self.optimizer = optim.Adagrad(params=self.model.parameters(), lr=cnf.lr)
        self.optimizer = optim.Adam(params=self.model.parameters(), lr=cnf.lr)
        #self.loss = QuantileLoss(cnf.quantiles)
        self.loss = MSELoss(cnf.quantiles)

        # init test loader
        self.test_loader = DataLoader(
            dataset=dataset_test, batch_size=cnf.batch_size,
            num_workers=cnf.n_workers, shuffle=False, pin_memory=True,
        )
        #pdb.set_trace()

        # init logging stuffs
        self.log_path = cnf.exp_log_path
        self.log_freq = len(self.test_loader)
        self.train_losses = []
        self.test_loss = []
        self.test_rmse_loss = []
        self.test_mse_loss = []
        self.test_mae_loss = []
        self.test_losses = {'p10': [], 'p50': [], 'p90': []}
        self.test_smape = []

        # starting values
        self.epoch = 0
        self.best_test_loss = None

        # init progress bar
        self.progress_bar = ProgressBar(max_step=self.log_freq, max_epoch=self.cnf.epochs)

        # possibly load checkpoint
        self.load_ck()

        print("Finished preparing datasets.")


    def load_ck_bak(self):
        """
        load training checkpoint
        """
        ck_path = self.log_path / self.cnf.exp_name + '_best.pth'
        if ck_path.exists():
            ck = torch.load(ck_path)
            print(f'[loading checkpoint \'{ck_path}\']')
            self.model.load_state_dict(ck)

    def find_latest_dir(self, base_path):  
        latest_dir = None  
        latest_time = None  

        for item in os.listdir(base_path):  
            item_path = os.path.join(base_path, item)  
            if os.path.isdir(item_path):  
                try:  
                    item_time = datetime.strptime(item, '%m-%d-%Y - %H-%M-%S')  
                except ValueError:  
                    continue  

                if latest_dir is None or item_time > latest_time:  
                    latest_dir = item_path  
                    latest_time = item_time  

        return latest_dir  


    def load_ck(self):
        """
        load training checkpoint
        """
        #ck_path = ./'log'/self.cnf.all_params["model"]/self.cnf.exp_name/self.cnf.exp_name + '_best.pth'
        #ck_path_root = f'./log/{self.cnf.all_params["model"]}/{self.cnf.exp_name}'
        #pdb.set_trace()
        #latest_log_dir = self.find_latest_dir(ck_path_root)
        #model_name='weibo_1234_4w_adam_transformer_rmse_48'
        #model_name='weibo_1234_4w_adam_bert_rmse_39'
        #model_name='weibo_1234_2w_adam_bert_rmse_49'
        #model_name='weibo_1234_4w_adam_bert_rmse_seed2468_50'
        #model_name='weibo_14_4w_adam_cate_rmse_seed2468_52'
        #model_name='weibo_model_1'
        #model_name='weibo_1234_4wfix_adam_selftf_text_vocab3m_notrain_rmse_67'
        #model_name='weibo_1234_4wfix_adam_selftf_text_vocab3m_notrain_rmse_piconly_72'
        model_name='weibo_1234_4wfix_adam_selftf_text_vocab3m_notrain_rmse_pic_title_75'
        ck_path_root = f'./log/{model_name}'
        ck_path = Path(f'{ck_path_root}/{self.cnf.exp_name}_best.pth')
        #pdb.set_trace()
        if ck_path.exists():
            ck = torch.load(ck_path)
            print(f'[loading checkpoint \'{ck_path}\']')
            self.model.load_state_dict(ck)

    def rmse_loss(self, y_pred, y):
        loss = (y_pred- y.unsqueeze(2)) ** 2  # 在第2维增加一个维度以便与preds进行广播
        mse_loss = torch.mean(torch.mean(loss, dim=2), dim=1)
        rmse_loss = torch.sqrt(mse_loss).mean()
        return rmse_loss

    def mse_loss(self, y_pred, y):
        loss = (y_pred- y.unsqueeze(2)) ** 2  # 在第2维增加一个维度以便与preds进行广播
        mse_loss = torch.mean(torch.mean(loss, dim=2), dim=1).mean()
        return mse_loss

    def mae_loss(self, y_pred, y):
        loss = torch.abs(y_pred - y.unsqueeze(2)) 
        mae_loss = torch.mean(torch.mean(loss, dim=2), dim=1).mean() 
        return mae_loss 

    def test(self):
        """
        Quick test and plot prediction without saving or logging stuff on tensorboarc
        """
        with torch.no_grad():
            self.model.eval()
            p10_forecast, p10_forecast, p90_forecast, target = None, None, None, None

            t = time()
            for step, sample in enumerate(self.test_loader):

                # Hide future predictions from input vector, set to 0 (or 1) values where timestep > encoder_steps
                steps = self.cnf.all_params['num_encoder_steps']
                pred_len = sample['outputs'].shape[1]
                x = sample['inputs'].float().to(self.cnf.device)
                x[:, steps:, 0] = 1

                # Feed input to the model
                if self.cnf.all_params["model"] == "transformer" or self.cnf.all_params["model"] == "grn_transformer":

                    # Auto-regressive prediction
                    for i in range(pred_len):
                        output = self.model.forward(x)
                        x[:, steps + i, 0] = output[:, i, 1]
                    output = self.model.forward(x)

                elif self.cnf.all_params["model"] == "selftf_transformer":
                    output, _, _ = self.model.forward(x)
                else:
                    raise NameError

                output = output.squeeze()
                y, y_pred = sample['outputs'].squeeze().float().to(self.cnf.device), output

                # Compute loss
                loss, _ = self.loss(y_pred, y)
                #pdb.set_trace()
                rmse_loss = self.rmse_loss(y_pred, y)
                mse_loss = self.mse_loss(y_pred, y)
                mae_loss = self.mae_loss(y_pred, y)
                smape = symmetric_mean_absolute_percentage_error(output[:, :, 1].detach().cpu().numpy(),
                                                                 sample['outputs'][:, :, 0].detach().cpu().numpy())

                # De-Normalize to compute metrics
                target = unnormalize_tensor(self.data_formatter, y, sample['identifier'][0][0])
                p10_forecast = unnormalize_tensor(self.data_formatter, y_pred[..., 0], sample['identifier'][0][0])
                p50_forecast = unnormalize_tensor(self.data_formatter, y_pred[..., 1], sample['identifier'][0][0])
                p90_forecast = unnormalize_tensor(self.data_formatter, y_pred[..., 2], sample['identifier'][0][0])

                # Compute metrics
                self.test_losses['p10'].append(self.loss.numpy_normalised_quantile_loss(p10_forecast, target, 0.1))
                self.test_losses['p50'].append(self.loss.numpy_normalised_quantile_loss(p50_forecast, target, 0.5))
                self.test_losses['p90'].append(self.loss.numpy_normalised_quantile_loss(p90_forecast, target, 0.9))

                self.test_loss.append(loss.item())
                self.test_rmse_loss.append(rmse_loss.item())
                self.test_mse_loss.append(mse_loss.item())
                self.test_mae_loss.append(mae_loss.item())
                self.test_loss.append(loss.item())
                self.test_smape.append(smape)

            # Plot serie prediction
            p1, p2, p3, target = np.expand_dims(p10_forecast, axis=-1), np.expand_dims(p50_forecast, axis=-1), \
                                 np.expand_dims(p90_forecast, axis=-1), np.expand_dims(target, axis=-1)
            p = np.concatenate((p1, p2, p3), axis=-1)
            plot_temporal_serie(p, target)

            # Log stuff
            for k in self.test_losses.keys():
                mean_test_loss = np.mean(self.test_losses[k])
                print(f'\t● AVG {k} Loss on TEST-set: {mean_test_loss:.6f} │ T: {time() - t:.2f} s')

            # log log log
            mean_test_loss = np.mean(self.test_loss)
            mean_rmse_loss = np.mean(self.test_rmse_loss)
            mean_mse_loss = np.mean(self.test_mse_loss)
            mean_mae_loss = np.mean(self.test_mae_loss)
            mean_smape = np.mean(self.test_smape)
            print(f'\t● AVG mean_test_loss Loss on TEST-set: {mean_test_loss:.6f} │ T: {time() - t:.2f} s')
            print(f'\t● AVG mean_rmse_loss Loss on TEST-set: {mean_rmse_loss:.6f} │ T: {time() - t:.2f} s')
            print(f'\t● AVG mean_mse_loss Loss on TEST-set: {mean_mse_loss:.6f} │ T: {time() - t:.2f} s')
            print(f'\t● AVG mean_mae_loss  Loss on TEST-set: {mean_mae_loss:.6f} │ T: {time() - t:.2f} s')
            print(f'\t● AVG SMAPE on TEST-set: {mean_smape:.6f} │ T: {time() - t:.2f} s')
