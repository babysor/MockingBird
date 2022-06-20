import os
import sys
import abc
import math
import yaml
import torch
from torch.utils.tensorboard import SummaryWriter

from .option import default_hparas
from utils.util import human_format, Timer
from utils.load_yaml import HpsYaml


class BaseSolver():
    ''' 
    Prototype Solver for all kinds of tasks
    Arguments
        config - yaml-styled config
        paras  - argparse outcome
        mode   - "train"/"test"
    '''

    def __init__(self, config, paras, mode="train"):
        # General Settings
        self.config = config  # load from yaml file
        self.paras = paras    # command line args  
        self.mode = mode      # 'train' or 'test'
        for k, v in default_hparas.items():
            setattr(self, k, v)
        self.device = torch.device('cuda') if self.paras.gpu and torch.cuda.is_available() \
                    else torch.device('cpu')

        # Name experiment
        self.exp_name = paras.name
        if self.exp_name is None:
            if 'exp_name' in self.config:
                self.exp_name = self.config.exp_name
            else:
                # By default, exp is named after config file
                self.exp_name = paras.config.split('/')[-1].replace('.yaml', '')
            if mode == 'train':
                self.exp_name += '_seed{}'.format(paras.seed)
                    

        if mode == 'train':
            # Filepath setup
            os.makedirs(paras.ckpdir, exist_ok=True)
            self.ckpdir = os.path.join(paras.ckpdir, self.exp_name)
            os.makedirs(self.ckpdir, exist_ok=True)

            # Logger settings
            self.logdir = os.path.join(paras.logdir, self.exp_name)
            self.log = SummaryWriter(
                self.logdir, flush_secs=self.TB_FLUSH_FREQ)
            self.timer = Timer()

            # Hyper-parameters
            self.step = 0
            self.valid_step = config.hparas.valid_step
            self.max_step = config.hparas.max_step

            self.verbose('Exp. name : {}'.format(self.exp_name))
            self.verbose('Loading data... large corpus may took a while.')

        # elif mode == 'test':
            # # Output path
            # os.makedirs(paras.outdir, exist_ok=True)
            # self.ckpdir = os.path.join(paras.outdir, self.exp_name)

            # Load training config to get acoustic feat and build model
            # self.src_config = HpsYaml(config.src.config) 
            # self.paras.load = config.src.ckpt

            # self.verbose('Evaluating result of tr. config @ {}'.format(
                # config.src.config))

    def backward(self, loss):
        '''
        Standard backward step with self.timer and debugger
        Arguments
            loss - the loss to perform loss.backward()
        '''
        self.timer.set()
        loss.backward()
        grad_norm = torch.nn.utils.clip_grad_norm_(
            self.model.parameters(), self.GRAD_CLIP)
        if math.isnan(grad_norm):
            self.verbose('Error : grad norm is NaN @ step '+str(self.step))
        else:
            self.optimizer.step()
        self.timer.cnt('bw')
        return grad_norm

    def load_ckpt(self):
        ''' Load ckpt if --load option is specified '''
        print(self.paras)
        if self.paras.load is not None:
            if self.paras.warm_start:
                self.verbose(f"Warm starting model from checkpoint {self.paras.load}.")
                ckpt = torch.load(
                    self.paras.load, map_location=self.device if self.mode == 'train'
                                                        else 'cpu')
                model_dict = ckpt['model']
                if "ignore_layers" in self.config.model and len(self.config.model.ignore_layers) > 0:
                    model_dict = {k:v for k, v in model_dict.items()
                                  if k not in self.config.model.ignore_layers}
                    dummy_dict = self.model.state_dict()
                    dummy_dict.update(model_dict)
                    model_dict = dummy_dict
                self.model.load_state_dict(model_dict)
            else:
                # Load weights
                ckpt = torch.load(
                    self.paras.load, map_location=self.device if self.mode == 'train'
                                                else 'cpu')
                self.model.load_state_dict(ckpt['model'])

                # Load task-dependent items
                if self.mode == 'train':
                    self.step = ckpt['global_step']
                    self.optimizer.load_opt_state_dict(ckpt['optimizer'])
                    self.verbose('Load ckpt from {}, restarting at step {}'.format(
                        self.paras.load, self.step))
                else:
                    for k, v in ckpt.items():
                        if type(v) is float:
                            metric, score = k, v
                    self.model.eval()
                    self.verbose('Evaluation target = {} (recorded {} = {:.2f} %)'.format(
                        self.paras.load, metric, score))

    def verbose(self, msg):
        ''' Verbose function for print information to stdout'''
        if self.paras.verbose:
            if type(msg) == list:
                for m in msg:
                    print('[INFO]', m.ljust(100))
            else:
                print('[INFO]', msg.ljust(100))

    def progress(self, msg):
        ''' Verbose function for updating progress on stdout (do not include newline) '''
        if self.paras.verbose:
            sys.stdout.write("\033[K")  # Clear line
            print('[{}] {}'.format(human_format(self.step), msg), end='\r')

    def write_log(self, log_name, log_dict):
        '''
        Write log to TensorBoard
            log_name  - <str> Name of tensorboard variable 
            log_value - <dict>/<array> Value of variable (e.g. dict of losses), passed if value = None
        '''
        if type(log_dict) is dict:
            log_dict = {key: val for key, val in log_dict.items() if (
                val is not None and not math.isnan(val))}
        if log_dict is None:
            pass
        elif len(log_dict) > 0:
            if 'align' in log_name or 'spec' in log_name:
                img, form = log_dict
                self.log.add_image(
                    log_name, img, global_step=self.step, dataformats=form)
            elif 'text' in log_name or 'hyp' in log_name:
                self.log.add_text(log_name, log_dict, self.step)
            else:
                self.log.add_scalars(log_name, log_dict, self.step)

    def save_checkpoint(self, f_name, metric, score, show_msg=True):
        '''' 
        Ckpt saver
            f_name - <str> the name of ckpt file (w/o prefix) to store, overwrite if existed
            score  - <float> The value of metric used to evaluate model
        '''
        ckpt_path = os.path.join(self.ckpdir, f_name)
        full_dict = {
            "model": self.model.state_dict(),
            "optimizer": self.optimizer.get_opt_state_dict(),
            "global_step": self.step,
            metric: score
        }

        torch.save(full_dict, ckpt_path)
        if show_msg:
            self.verbose("Saved checkpoint (step = {}, {} = {:.2f}) and status @ {}".
                         format(human_format(self.step), metric, score, ckpt_path))


    # ----------------------------------- Abtract Methods ------------------------------------------ #
    @abc.abstractmethod
    def load_data(self):
        '''
        Called by main to load all data
        After this call, data related attributes should be setup (e.g. self.tr_set, self.dev_set)
        No return value
        '''
        raise NotImplementedError

    @abc.abstractmethod
    def set_model(self):
        '''
        Called by main to set models
        After this call, model related attributes should be setup (e.g. self.l2_loss)
        The followings MUST be setup
            - self.model (torch.nn.Module)
            - self.optimizer (src.Optimizer),
                init. w/ self.optimizer = src.Optimizer(self.model.parameters(),**self.config['hparas'])
        Loading pre-trained model should also be performed here 
        No return value
        '''
        raise NotImplementedError

    @abc.abstractmethod
    def exec(self):
        '''
        Called by main to execute training/inference
        '''
        raise NotImplementedError
