import torch
from abc import abstractmethod
from numpy import inf
import numpy as np
import os
import copy

class BaseTrainer:
    """
    Base class for all trainers
    """
    def __init__(self, model, metric_ftns, config):
        self.config = config
        self.logger = config.get_logger('trainer', config['trainer']['verbosity'])

        # setup GPU device if available, move model into configured device
        self.device, device_ids = self._prepare_device(config['n_gpu'])
        self.model = model.to(self.device)
        if len(device_ids) > 1:
            self.model = torch.nn.DataParallel(model, device_ids=device_ids)

        # self.criterion = criterion
        self.metric_ftns = metric_ftns

        cfg_trainer = config['trainer']
        self.epochs = cfg_trainer['epochs']
        self.monitor = cfg_trainer.get('monitor', 'off')
        self.save_model = cfg_trainer['save_model']

        # configuration to monitor model performance and save best
        if self.monitor == 'off':
            self.mnt_mode = 'off'
        else:
            self.mnt_mode, self.mnt_metric = self.monitor.split()
            assert self.mnt_mode in ['min', 'max']
            self.early_stop = cfg_trainer.get('early_stop', inf)

        self.start_epoch = 1
        self.checkpoint_dir = config.save_dir
        if config.resume is not None:
            self._resume_checkpoint(config.resume)

    def train(self): 
        # if hasattr(self.model, 'do_pretrain'):
        #     self.logger.info('Pre-training the model...')
        #     self._pre_train()
        
        not_improved_count = 0
        mnt_best = inf if self.mnt_mode == 'min' else -inf
        for epoch in range(self.start_epoch, self.epochs + 1):
            result = self._train_epoch(epoch)
            log = {'epoch': epoch}
            log.update(result)
            if epoch % 2 == 0:
                for key, value in log.items():
                    self.logger.info('    {:15s}: {}'.format(str(key), value))

            if self.mnt_mode != 'off':
                mnt_best, not_improved_count = self.early_stopping(log, mnt_best, not_improved_count, do_save=True)
        
                if not_improved_count > self.early_stop:
                    self.logger.info("Validation performance didn\'t improve for {} epochs. "
                                        "Training stops.".format(self.early_stop))
                    if self.save_model:
                        self._save_checkpoint(epoch, save_best=True)
                    break
        
            if epoch == self.epochs:
                self.logger.info("Validation performance did not converge...")
                self._save_checkpoint(epoch, save_best=False) if self.save_model else None
            
        return self._test_epoch()  
    
    def _pre_train(self, pre_train_epochs=50000):
        not_improved_count = 0
        mnt_best = inf if self.mnt_mode == 'min' else -inf
        for epoch in range(self.start_epoch, pre_train_epochs + 1):
            log = self._pre_train_epoch(epoch)

            if self.mnt_mode != 'off':
                mnt_best, not_improved_count = self.early_stopping(log, mnt_best, not_improved_count, do_save=False)
        
                if not_improved_count > self.early_stop:
                    self.logger.info("Validation performance didn\'t improve for {} epochs. "
                                        "Pre-training stops.".format(self.early_stop))
                    break
            
            if epoch == pre_train_epochs:
                self.logger.info("Validation performance did not converge...")
            
    def early_stopping(self, log, mnt_best, not_improved_count, do_save=True):
        try:
            improved = (self.mnt_mode == 'min' and log[self.mnt_metric] < mnt_best) or \
                        (self.mnt_mode == 'max' and log[self.mnt_metric] > mnt_best)
        except KeyError:
            self.logger.warning("Warning: Metric '{}' is not found. "
                                "Model performance monitoring is disabled.".format(self.mnt_metric))
            self.mnt_mode = 'off'
            improved = False

        if improved:
            mnt_best = log[self.mnt_metric]
            if do_save:
                self.model_best_params = copy.deepcopy(self.model.state_dict())
            not_improved_count = 0
        else:
            not_improved_count += 1
        return mnt_best, not_improved_count
    
    
    def _prepare_device(self, n_gpu_use):
        n_gpu = torch.cuda.device_count()
        if n_gpu_use > 0 and n_gpu == 0:
            self.logger.warning("Warning: There\'s no GPU available on this machine,"
                                "training will be performed on CPU.")
            n_gpu_use = 0
        if n_gpu_use > n_gpu:
            self.logger.warning("Warning: The number of GPU\'s configured to use is {}, but only {} are available "
                                "on this machine.".format(n_gpu_use, n_gpu))
            n_gpu_use = n_gpu
        device = torch.device('cuda:0' if n_gpu_use > 0 else 'cpu')
        list_ids = list(range(n_gpu_use))
        return device, list_ids

    def _save_checkpoint(self, epoch, save_best=False):
        arch = type(self.model).__name__
        state = {
            'arch': arch,
            'epoch': epoch,
            'state_dict': self.model_best_params,
            'optimizer': self.model.optimizer.state_dict(),
            'monitor_best': self.mnt_best,
            'config': self.config
        }
        self.logger.info("Saving current best: model_best.pth ...")
        torch.save(state, str(self.checkpoint_dir / 'model_best.pth'))


'''
import torch
from abc import abstractmethod
from numpy import inf
import numpy as np
import os
import copy

class BaseTrainer:
    """
    Base class for all trainers
    """
    def __init__(self, model, metric_ftns, config):
        self.config = config
        self.logger = config.get_logger('trainer', config['trainer']['verbosity'])

        # setup GPU device if available, move model into configured device
        self.device, device_ids = self._prepare_device(config['n_gpu'])
        self.model = model.to(self.device)
        if len(device_ids) > 1:
            self.model = torch.nn.DataParallel(model, device_ids=device_ids)

        # self.criterion = criterion
        self.metric_ftns = metric_ftns

        cfg_trainer = config['trainer']
        self.epochs = cfg_trainer['epochs']
        self.monitor = cfg_trainer.get('monitor', 'off')
        self.save_model = cfg_trainer['save_model']

        # configuration to monitor model performance and save best
        if self.monitor == 'off':
            self.mnt_mode = 'off'
            self.mnt_best = 0
        else:
            self.mnt_mode, self.mnt_metric = self.monitor.split()
            assert self.mnt_mode in ['min', 'max']

            self.mnt_best = inf if self.mnt_mode == 'min' else -inf
            self.early_stop = cfg_trainer.get('early_stop', inf)

        self.start_epoch = 1
        self.checkpoint_dir = config.save_dir
        if config.resume is not None:
            self._resume_checkpoint(config.resume)

    @abstractmethod
    def _train_epoch(self, epoch, total_epochs):
        """
        Training logic for an epoch

        :param epoch: Current epoch number
        """
        raise NotImplementedError

    def train(self):
        """
        Full training logic
        """
        not_improved_count = 0
        for epoch in range(self.start_epoch, self.epochs + 1):
            result = self._train_epoch(epoch)

            log = {'epoch': epoch}
            log.update(result)

            if epoch % 10 == 0:
                for key, value in log.items():
                    self.logger.info('    {:15s}: {}'.format(str(key), value))

            # evaluate model performance according to configured metric, save best checkpoint as model_best
            if self.mnt_mode != 'off':
                try:
                    # check whether model performance improved or not, according to specified metric(mnt_metric)
                    improved = (self.mnt_mode == 'min' and log[self.mnt_metric] < self.mnt_best) or \
                               (self.mnt_mode == 'max' and log[self.mnt_metric] > self.mnt_best)
                except KeyError:
                    self.logger.warning("Warning: Metric '{}' is not found. "
                                        "Model performance monitoring is disabled.".format(self.mnt_metric))
                    self.mnt_mode = 'off'
                    improved = False

                if improved:
                    self.mnt_best = log[self.mnt_metric]
                    self.model_best_params = copy.deepcopy(self.model.state_dict())
                    not_improved_count = 0
                else:
                    not_improved_count += 1

                if not_improved_count > self.early_stop:
                    self.logger.info("Validation performance didn\'t improve for {} epochs. "
                                     "Training stops.".format(self.early_stop))
                    if self.save_model:
                        self._save_checkpoint(epoch, save_best=True)
                    break

            if epoch == self.epochs and self.save_model:
                self._save_checkpoint(epoch, save_best=False)
            
        return self._test_epoch()  


    def _prepare_device(self, n_gpu_use):
        """
        setup GPU device if available, move model into configured device
        """
        n_gpu = torch.cuda.device_count()
        if n_gpu_use > 0 and n_gpu == 0:
            self.logger.warning("Warning: There\'s no GPU available on this machine,"
                                "training will be performed on CPU.")
            n_gpu_use = 0
        if n_gpu_use > n_gpu:
            self.logger.warning("Warning: The number of GPU\'s configured to use is {}, but only {} are available "
                                "on this machine.".format(n_gpu_use, n_gpu))
            n_gpu_use = n_gpu
        device = torch.device('cuda:0' if n_gpu_use > 0 else 'cpu')
        list_ids = list(range(n_gpu_use))
        return device, list_ids

    def _save_checkpoint(self, epoch, save_best=False):
        arch = type(self.model).__name__
        state = {
            'arch': arch,
            'epoch': epoch,
            'state_dict': self.model_best_params,
            'optimizer': self.model.optimizer.state_dict(),
            'monitor_best': self.mnt_best,
            'config': self.config
        }
        if not save_best:
            self.logger.info("Validation performance did not converge.")
        self.logger.info("Saving current best: model_best.pth ...")
        torch.save(state, str(self.checkpoint_dir / 'model_best.pth'))

'''