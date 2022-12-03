import torch
import numpy as np


class Optimizer():
    def __init__(self, parameters, optimizer, lr, eps, lr_scheduler, 
                **kwargs):

        # Setup torch optimizer
        self.opt_type = optimizer
        self.init_lr = lr
        self.sch_type = lr_scheduler
        opt = getattr(torch.optim, optimizer)
        if lr_scheduler == 'warmup':
            warmup_step = 4000.0
            init_lr = lr
            self.lr_scheduler = lambda step: init_lr * warmup_step ** 0.5 * \
                np.minimum((step+1)*warmup_step**-1.5, (step+1)**-0.5)
            self.opt = opt(parameters, lr=1.0)
        else:
            self.lr_scheduler = None
            self.opt = opt(parameters, lr=lr, eps=eps)  # ToDo: 1e-8 better?

    def get_opt_state_dict(self):
        return self.opt.state_dict()

    def load_opt_state_dict(self, state_dict):
        self.opt.load_state_dict(state_dict)

    def pre_step(self, step):
        if self.lr_scheduler is not None:
            cur_lr = self.lr_scheduler(step)
            for param_group in self.opt.param_groups:
                param_group['lr'] = cur_lr
        else:
            cur_lr = self.init_lr
        self.opt.zero_grad()
        return cur_lr 
 
    def step(self):
        self.opt.step()

    def create_msg(self):
        return ['Optim.Info.| Algo. = {}\t| Lr = {}\t (schedule = {})'
                .format(self.opt_type, self.init_lr, self.sch_type)]
