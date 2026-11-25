import torch

from models import CALF
from models import CMIE


class Exp_Basic(object):
    def __init__(self, args):
        self.args = args
        self.model_dict = {
            "CALF": CALF,
            "CMIE": CMIE,
        }
        self.device = self._acquire_device()
        self.model = self._build_model().to(self.device)

    def _build_model(self):
        raise NotImplementedError
        return None

    def _acquire_device(self):
        if self.args.use_gpu:
            if self.args.use_multi_gpu:
                self.args.devices = self.args.devices.replace(' ', '')
                device_ids = self.args.devices.split(',')
                self.args.device_ids = [int(id_) for id_ in device_ids]
                self.args.gpu = self.args.device_ids[0]
            device = torch.device('cuda:{}'.format(self.args.gpu))
            print('Use GPU: cuda:{}'.format(self.args.gpu))
        else:
            device = torch.device('cpu')
            print('Use CPU')
        return device

    def _get_data(self):
        pass

    def _set_channel_stats(self, train_data):
        model = self.model.module if isinstance(self.model, torch.nn.DataParallel) else self.model
        if hasattr(model, 'set_channel_stats') and getattr(train_data, 'scaler', None) is not None:
            from utils.tools import channel_stats_from_dataset
            model.set_channel_stats(channel_stats_from_dataset(train_data))

    def _set_channel_groups(self, train_data):
        model = self.model.module if isinstance(self.model, torch.nn.DataParallel) else self.model
        if hasattr(model, 'set_channel_groups') and getattr(self.args, 'merge_threshold', 0.0) > 0 \
                and getattr(train_data, 'data_x', None) is not None:
            from utils.tools import channel_groups_from_dataset
            model.set_channel_groups(channel_groups_from_dataset(train_data, self.args.merge_threshold))

    def vali(self):
        pass

    def train(self):
        pass

    def test(self):
        pass
