import torch
import numpy as np


class ScheduledOptim:
    """ A robust wrapper class for learning rate scheduling with warmup """

    def __init__(self, model, train_config, model_config, current_step=0):
        # Support both train_config['optimizer'] and train_config['train'] schemas
        opt_cfg = train_config.get("optimizer", train_config.get("train", {}))
        
        betas = opt_cfg.get("betas", [0.9, 0.98])
        eps = float(opt_cfg.get("eps", 1e-9))
        weight_decay = float(opt_cfg.get("weight_decay", 1e-6))
        
        self._optimizer = torch.optim.Adam(
            model.parameters(),
            betas=betas,
            eps=eps,
            weight_decay=weight_decay,
        )
        self.n_warmup_steps = opt_cfg.get("warm_up_step", opt_cfg.get("warmup_step", 1000))
        self.anneal_steps = opt_cfg.get("anneal_steps", [300000, 400000, 500000])
        self.anneal_rate = opt_cfg.get("anneal_rate", 0.3)
        self.current_step = current_step
        hidden_dim = model_config.get("transformer", {}).get("encoder_hidden", 256)
        self.init_lr = np.power(hidden_dim, -0.5)

    def step_and_update_lr(self):
        self._update_learning_rate()
        self._optimizer.step()

    def zero_grad(self):
        self._optimizer.zero_grad()

    def state_dict(self):
        return self._optimizer.state_dict()

    def load_state_dict(self, state_dict):
        self._optimizer.load_state_dict(state_dict)

    def _get_lr_scale(self):
        step = max(1, self.current_step)
        lr = np.min(
            [
                np.power(step, -0.5),
                np.power(self.n_warmup_steps, -1.5) * step,
            ]
        )
        for s in self.anneal_steps:
            if step > s:
                lr = lr * self.anneal_rate
        return lr

    def _update_learning_rate(self):
        """ Learning rate scheduling per step """
        self.current_step += 1
        lr = self.init_lr * self._get_lr_scale()

        for param_group in self._optimizer.param_groups:
            param_group["lr"] = lr
