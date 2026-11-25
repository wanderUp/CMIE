import gc
import os
import time
import warnings
from datetime import datetime

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import optim
from tqdm import tqdm

from data_provider.data_factory import data_provider
from exp.exp_basic import Exp_Basic
from utils.CMIELoss import CMIELoss
from utils.metrics import metric
from utils.tools import EarlyStopping, adjust_learning_rate, visual

warnings.filterwarnings('ignore')


class Exp_Long_Term_Forecast(Exp_Basic):
    """
    Long-term forecasting experiment with efficiency tracking.

    This version fixes:
    1. LazyModule / UninitializedParameter errors during parameter counting.
    2. GPU memory being reported as 0MB when the model is actually on CUDA.
    3. THOP modifying the training model during FLOPs calculation.

    Recorded metrics:
    - Params / trainable params
    - FLOPs, best-effort via THOP
    - Total training time
    - Average epoch time
    - Train ms/iter
    - Peak training GPU memory
    - Inference ms/batch
    - Inference ms/sample
    - Peak inference GPU memory
    """

    def __init__(self, args):
        super(Exp_Long_Term_Forecast, self).__init__(args)

    def _build_model(self):
        model = self.model_dict[self.args.model].Model(self.args, self.device).float()
        if self.args.use_multi_gpu and self.args.use_gpu:
            model = nn.DataParallel(model, device_ids=self.args.device_ids)
        return model

    def _get_data(self, flag, vali_test=False):
        data_set, data_loader = data_provider(self.args, flag, vali_test)
        return data_set, data_loader

    def _select_optimizer(self):
        return optim.AdamW(
            self.model.parameters(),
            lr=self.args.learning_rate,
            weight_decay=self.args.weight_decay,
        )

    def _unpack(self, outputs):
        """Return plain tensor whether model returns dict (e.g., CALF) or tensor (e.g., CMIE)."""
        if isinstance(outputs, dict):
            return outputs['outputs_time']
        return outputs

    def _select_criterion(self):
        if self.args.model == "CALF":
            criterion = nn.MSELoss()
        elif self.args.model == "CMIE":
            criterion = CMIELoss(self.args.task_name, self.args.output_loss, self.args.lambda_sim)
        else:
            criterion = CMIELoss(self.args.task_name, self.args.output_loss)
        return criterion

    # -------------------------------------------------------------------------
    # Efficiency helpers
    # -------------------------------------------------------------------------
    def _actual_model(self):
        """Return the underlying model when DataParallel is used."""
        return self.model.module if isinstance(self.model, nn.DataParallel) else self.model

    def _model_device(self):
        """Return the actual device where model parameters live."""
        try:
            return next(self.model.parameters()).device
        except StopIteration:
            return torch.device(self.device)

    def _cuda_available(self):
        dev = self._model_device()
        return torch.cuda.is_available() and dev.type == 'cuda'

    def _sync_cuda(self):
        if self._cuda_available():
            torch.cuda.synchronize(self._model_device())

    def _reset_peak_memory(self):
        if self._cuda_available():
            dev = self._model_device()
            torch.cuda.set_device(dev)
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats(dev)

    def _peak_allocated_mb(self):
        if not self._cuda_available():
            return 0.0
        dev = self._model_device()
        return torch.cuda.max_memory_allocated(dev) / (1024 ** 2)

    def _infer_input_dim(self):
        """
        Infer input channel dimension for dummy input.

        Priority:
        args.N -> args.enc_in -> args.c_in -> args.dec_in
        """
        for name in ['N', 'enc_in', 'c_in', 'dec_in']:
            if hasattr(self.args, name):
                value = getattr(self.args, name)
                if value is not None:
                    return int(value)
        raise AttributeError("Cannot infer input channel dimension. Please set args.N or args.enc_in.")

    def _clean_thop_attrs(self, model):
        """Remove THOP-added attributes recursively."""
        for m in model.modules():
            for attr in ['total_ops', 'total_params']:
                if hasattr(m, attr):
                    try:
                        delattr(m, attr)
                    except Exception:
                        pass

    def _warmup_lazy_parameters(self):
        """
        Initialize LazyModule parameters by running a dummy forward once.

        Some models contain LazyLinear/LazyConv or UninitializedParameter.
        Calling p.numel() before the first forward will raise:
        ValueError: Attempted to use an uninitialized parameter...
        """
        device = self._model_device()
        seq_len = int(self.args.seq_len)
        n_channels = self._infer_input_dim()
        dummy_x = torch.randn(1, seq_len, n_channels, device=device)

        was_training = self.model.training
        self.model.eval()
        try:
            with torch.no_grad():
                _ = self.model(dummy_x)
            self._sync_cuda()
        finally:
            if was_training:
                self.model.train()
            del dummy_x
            gc.collect()
            if self._cuda_available():
                torch.cuda.empty_cache()

    def _safe_count_parameters(self):
        """
        Count parameters safely when LazyModule may still have UninitializedParameter.
        """
        try:
            from torch.nn.parameter import UninitializedParameter
        except Exception:
            UninitializedParameter = None

        n_params = 0
        n_trainable = 0

        for p in self.model.parameters():
            if UninitializedParameter is not None and isinstance(p, UninitializedParameter):
                continue
            try:
                num = p.numel()
            except ValueError:
                continue
            n_params += num
            if p.requires_grad:
                n_trainable += num

        return n_params, n_trainable

    def _compute_flops_with_thop(self):
        """
        Best-effort FLOPs calculation with THOP.

        Returns:
            flops_str: formatted string, e.g., 12.345G, or 'N/A'
        """
        try:
            # Make sure LazyModule parameters are initialized before deepcopy/profile.
            self._warmup_lazy_parameters()

            import copy
            from thop import clever_format, profile

            device = self._model_device()
            source_model = self._actual_model()

            # Deepcopy prevents THOP from adding total_ops/total_params to the training model.
            flops_model = copy.deepcopy(source_model).to(device)
            flops_model.eval()
            self._clean_thop_attrs(flops_model)

            seq_len = int(self.args.seq_len)
            n_channels = self._infer_input_dim()
            dummy_x = torch.randn(1, seq_len, n_channels, device=device)

            with torch.no_grad():
                macs, _ = profile(flops_model, inputs=(dummy_x,), verbose=False)

            # THOP reports MACs. Common convention: 1 MAC ≈ 2 FLOPs.
            flops = float(macs) * 2.0
            flops_str, _ = clever_format([flops, 0], "%.3f")

            del flops_model, dummy_x
            gc.collect()
            if self._cuda_available():
                torch.cuda.empty_cache()

            return flops_str

        except Exception as e:
            print("  [FLOPs] skip: {}".format(repr(e)))
            return 'N/A'

    def _init_efficiency_stats(self):
        self.total_train_time = 0.0
        self.avg_epoch_time = 0.0
        self.ms_per_iter = 0.0
        self.train_max_memory_mb = 0.0
        self.inf_ms_per_batch = 0.0
        self.inf_ms_per_sample = 0.0
        self.inf_max_memory_mb = 0.0
        self.flops_str = 'N/A'

        # Initialize LazyModule parameters before counting.
        try:
            self._warmup_lazy_parameters()
        except Exception as e:
            print("[Efficiency] Lazy warmup skip: {}".format(repr(e)))

        self.n_params, self.n_trainable = self._safe_count_parameters()

    def _print_efficiency_summary(self):
        print(
            "Params: {:.2f}M (trainable {:.2f}M) | FLOPs: {} | "
            "Train GPU mem: {:.0f}MB | Avg epoch: {:.2f}s | ms/iter: {:.3f}".format(
                self.n_params / 1e6,
                self.n_trainable / 1e6,
                self.flops_str,
                self.train_max_memory_mb,
                self.avg_epoch_time,
                self.ms_per_iter,
                )
        )

    def _efficiency_text(self):
        return (
            ', params:{:.2f}M(trainable:{:.2f}M)'
            ', flops:{}'
            ', train_time:{:.1f}s'
            ', avg_epoch:{:.2f}s'
            ', train_ms/iter:{:.3f}'
            ', train_mem:{:.0f}MB'
            ', inf_ms/batch:{:.3f}'
            ', inf_ms/sample:{:.3f}'
            ', inf_mem:{:.0f}MB'
        ).format(
            self.n_params / 1e6,
            self.n_trainable / 1e6,
            self.flops_str,
            self.total_train_time,
            self.avg_epoch_time,
            self.ms_per_iter,
            self.train_max_memory_mb,
            self.inf_ms_per_batch,
            self.inf_ms_per_sample,
            self.inf_max_memory_mb,
            )

    # -------------------------------------------------------------------------
    # Train / validation / test
    # -------------------------------------------------------------------------
    def train(self, setting):
        train_data, train_loader = self._get_data(flag='train')
        self._set_channel_stats(train_data)
        self._set_channel_groups(train_data)
        vali_data, vali_loader = self._get_data(flag='val')
        test_data, test_loader = self._get_data(flag='test', vali_test=True)

        path = os.path.join(self.args.checkpoints, setting)
        if not os.path.exists(path):
            os.makedirs(path)

        time_now = time.time()
        train_steps = len(train_loader)
        early_stopping = EarlyStopping(patience=self.args.patience, verbose=True)

        model_optim = self._select_optimizer()
        criterion = self._select_criterion()
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            model_optim, T_max=self.args.tmax, eta_min=1e-8
        )

        self._init_efficiency_stats()

        # FLOPs are computed before training, then peak memory is reset.
        self.flops_str = self._compute_flops_with_thop()
        print("FLOPs: {}".format(self.flops_str))

        # Important: reset peak memory AFTER Lazy warmup and FLOPs calculation.
        self._reset_peak_memory()
        n_epochs_trained = 0

        for epoch in range(self.args.train_epochs):
            iter_count = 0
            train_loss = []

            self.model.train()
            self._sync_cuda()
            epoch_time = time.time()

            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in tqdm(
                    enumerate(train_loader),
                    desc="Processing_train{0}".format(iter_count),
            ):
                iter_count += 1
                model_optim.zero_grad()

                epsilon = self.args.epsilon
                data_range = batch_x.max() - batch_x.min()
                noise = epsilon * (torch.rand_like(batch_x) - 0.5) * data_range
                batch_x = batch_x + noise

                device = self._model_device()
                batch_x = batch_x.float().to(device)
                batch_y = batch_y.float().to(device)
                batch_x_mark = batch_x_mark.float().to(device)
                batch_y_mark = batch_y_mark.float().to(device)

                outputs = self.model(batch_x)
                outputs = self._unpack(outputs)

                if self.args.model == "CMIE":
                    # DataParallel wrapper does not directly expose conv_embeds.
                    model_for_attr = self._actual_model()
                    loss = criterion(outputs, batch_y, conv_embeds=model_for_attr.conv_embeds)
                else:
                    loss = criterion(
                        outputs[:, -self.args.pred_len:, :],
                        batch_y[:, -self.args.pred_len:, :],
                    )

                train_loss.append(loss.item())

                if (i + 1) % 100 == 0:
                    self._sync_cuda()
                    print("\titers: {0}, epoch: {1} | loss: {2:.7f}".format(
                        i + 1, epoch + 1, loss.item()
                    ))
                    speed = (time.time() - time_now) / iter_count
                    left_time = speed * ((self.args.train_epochs - epoch) * train_steps - i)
                    print('\tspeed: {:.4f}s/iter; left time: {:.4f}s'.format(speed, left_time))
                    iter_count = 0
                    time_now = time.time()

                loss.backward()
                model_optim.step()

            self._sync_cuda()
            epoch_cost = time.time() - epoch_time
            self.total_train_time += epoch_cost
            n_epochs_trained += 1
            print("Epoch: {} cost time: {:.2f}s".format(epoch + 1, epoch_cost))
            train_loss = np.average(train_loss)

            vali_loss = self.vali(vali_data, vali_loader, criterion)
            test_loss = self.vali(test_data, test_loader, criterion)

            print("Epoch: {0}, Steps: {1} | Train Loss: {2:.7f} Vali Loss: {3:.7f} Test Loss: {4:.7f}".format(
                epoch + 1, train_steps, train_loss, vali_loss, test_loss
            ))

            if self.args.cos:
                scheduler.step()
                print("lr = {:.10f}".format(model_optim.param_groups[0]['lr']))
            else:
                adjust_learning_rate(model_optim, epoch + 1, self.args)

            early_stopping(vali_loss, self.model, path)
            if early_stopping.early_stop:
                print("Early stopping")
                break

        self._sync_cuda()
        self.train_max_memory_mb = self._peak_allocated_mb()
        self.avg_epoch_time = self.total_train_time / n_epochs_trained if n_epochs_trained > 0 else 0.0
        self.ms_per_iter = (
            self.total_train_time / (n_epochs_trained * train_steps) * 1000
            if n_epochs_trained > 0 and train_steps > 0 else 0.0
        )

        self._print_efficiency_summary()

        best_model_path = path + '/' + 'checkpoint.pth'
        print(best_model_path)
        self.model.load_state_dict(torch.load(best_model_path))

        return self.model

    def vali(self, vali_data, vali_loader, criterion):
        total_loss = []
        iter_count = 0
        self.model.eval()
        device = self._model_device()

        with torch.no_grad():
            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in tqdm(
                    enumerate(vali_loader),
                    desc="Processing_vali{}".format(iter_count),
            ):
                batch_x = batch_x.float().to(device)
                batch_y = batch_y.float()
                batch_x_mark = batch_x_mark.float().to(device)
                batch_y_mark = batch_y_mark.float().to(device)

                outputs = self.model(batch_x)
                outputs = self._unpack(outputs)

                outputs_ensemble = outputs[:, -self.args.pred_len:, :]
                batch_y = batch_y[:, -self.args.pred_len:, :].to(device)

                pred = outputs_ensemble.detach().cpu()
                true = batch_y.detach().cpu()
                loss = F.mse_loss(pred, true)
                total_loss.append(loss)

        total_loss = np.average(total_loss)
        return total_loss

    def test(self, setting, test=0):
        if self.args.zero_shot:
            self.args.data = self.args.target_data
            self.args.data_path = f"{self.args.data}.csv"

        test_data, test_loader = self._get_data(flag='test')
        if test:
            print('loading model')
            self.model.load_state_dict(
                torch.load(os.path.join('./checkpoints/' + setting, 'checkpoint.pth')),
                strict=False,
            )

        # Ensure efficiency stats exist even if test() is called directly.
        if not hasattr(self, 'n_params'):
            self._init_efficiency_stats()
            self.flops_str = self._compute_flops_with_thop()
            print("FLOPs: {}".format(self.flops_str))

        preds = []
        trues = []
        folder_path = './test_results/' + setting + '/'
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)

        n_test_samples = 0
        device = self._model_device()

        # Important: reset peak memory immediately before inference.
        self._reset_peak_memory()

        self.model.eval()
        self._sync_cuda()
        inf_start = time.time()

        with torch.no_grad():
            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(test_loader):
                batch_x = batch_x.float().to(device)
                batch_y = batch_y.float().to(device)
                batch_x_mark = batch_x_mark.float().to(device)
                n_test_samples += batch_x.size(0)

                outputs = self.model(batch_x[:, -self.args.seq_len:, :])
                outputs = self._unpack(outputs)

                outputs_ensemble = outputs[:, -self.args.pred_len:, :]
                batch_y = batch_y[:, -self.args.pred_len:, :]

                pred = outputs_ensemble.detach().cpu().numpy()
                true = batch_y.detach().cpu().numpy()
                preds.append(pred)
                trues.append(true)

                if i % 20 == 0:
                    input = batch_x.detach().cpu().numpy()
                    if test_data.scale and self.args.inverse:
                        shape = input.shape
                        input = test_data.inverse_transform(input.squeeze(0)).reshape(shape)
                    gt = np.concatenate((input[0, :, -1], true[0, :, -1]), axis=0)
                    pd = np.concatenate((input[0, :, -1], pred[0, :, -1]), axis=0)
                    visual(gt, pd, os.path.join(folder_path, str(i) + '.pdf'))

        self._sync_cuda()
        inf_time = time.time() - inf_start
        n_batches = len(test_loader)
        self.inf_ms_per_batch = (inf_time / n_batches) * 1000 if n_batches > 0 else 0.0
        self.inf_ms_per_sample = (inf_time / n_test_samples) * 1000 if n_test_samples > 0 else 0.0
        self.inf_max_memory_mb = self._peak_allocated_mb()

        preds = np.array(preds)
        trues = np.array(trues)
        print('test shape:', preds.shape, trues.shape)
        preds = preds.reshape(-1, preds.shape[-2], preds.shape[-1])
        trues = trues.reshape(-1, trues.shape[-2], trues.shape[-1])
        print('test shape:', preds.shape, trues.shape)

        folder_path = './results/' + setting + '/'
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)

        mae, mse, rmse, mape, mspe = metric(preds, trues)
        print('mse:{}, mae:{},rmse:{},mape:{},mspe:{}'.format(mse, mae, rmse, mape, mspe))
        print(
            "Inference: {:.3f} ms/batch | {:.3f} ms/sample | GPU mem: {:.0f}MB".format(
                self.inf_ms_per_batch,
                self.inf_ms_per_sample,
                self.inf_max_memory_mb,
            )
        )

        model_prefix = self.args.model_id.split('_')[0]
        with open("result_{}.txt".format(model_prefix), 'a') as f:
            f.write(setting + "  \n")
            f.write('mse:{:.6f}, mae:{:.6f}'.format(mse, mae))
            f.write(self._efficiency_text())
            f.write('\n')
            formatted_time = datetime.now().strftime("%Y年%m月%d日 %H:%M")
            f.write(formatted_time)
            f.write('\n\n')

        with open(os.path.join('./checkpoints/' + setting, 'result.txt'), 'w') as f:
            f.write('mse:{}, mae:{},rmse:{},mape:{},mspe:{}'.format(mse, mae, rmse, mape, mspe))
            f.write(self._efficiency_text())

        np.save(folder_path + 'metrics.npy', np.array([mae, mse, rmse, mape, mspe]))
        np.save(folder_path + 'pred.npy', preds)
        np.save(folder_path + 'true.npy', trues)

        return
