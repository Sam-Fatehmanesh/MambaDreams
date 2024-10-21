

import torch
from torch import nn, optim
from typing import Iterable, Union

def unitwise_norm(x: torch.Tensor):
    if x.ndim <= 1:
        dim = 0
        keepdim = False
    elif x.ndim in [2, 3]:
        dim = 0
        keepdim = True
    elif x.ndim == 4:
        dim = (1, 2, 3)
        keepdim = True
    else:
        raise ValueError('Tensor dimension not supported')

    return torch.sum(x**2, dim=dim, keepdim=keepdim) ** 0.5


class AGC(optim.Optimizer):
    def __init__(self, params, base_optimizer: optim.Optimizer, clipping: float = 0.3, eps: float = 1e-3, 
                 model: nn.Module = None, ignore_agc: Union[str, Iterable[str]] = ["fc"]):
        if clipping < 0.0:
            raise ValueError(f"Invalid clipping value: {clipping}")
        if eps < 0.0:
            raise ValueError(f"Invalid eps value: {eps}")

        self.base_optimizer = base_optimizer
        self.clipping = clipping
        self.eps = eps

        if not isinstance(ignore_agc, Iterable):
            ignore_agc = [ignore_agc]

        if model is not None:
            assert ignore_agc, "You must specify ignore_agc for AGC to ignore fc-like (or other) layers"
            named_modules = dict(model.named_modules())
            for module_name in ignore_agc:
                if module_name not in named_modules:
                    raise ModuleNotFoundError(f"Module name {module_name} not found in the model")
            
            self.agc_params = [
                {'params': list(module.parameters())}
                for name, module in named_modules.items() if name not in ignore_agc
            ]
        else:
            self.agc_params = [{'params': params}]

        self.param_groups = self.base_optimizer.param_groups
        self.state = self.base_optimizer.state

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.agc_params:
            for p in group['params']:
                if p.grad is None or p.grad.numel() == 0:
                    continue

                param_norm = unitwise_norm(p)
                grad_norm = unitwise_norm(p.grad)

                if param_norm.numel() == 0 or grad_norm.numel() == 0:
                    continue

                max_norm = torch.clamp(param_norm, min=self.eps) * self.clipping
                trigger = grad_norm > max_norm

                clipped_grad = p.grad * (max_norm / torch.clamp(grad_norm, min=self.eps))
                p.grad.data.copy_(torch.where(trigger, clipped_grad, p.grad))

        return self.base_optimizer.step(closure)



    def zero_grad(self, set_to_none: bool = False):
        for group in self.agc_params:
            for p in group['params']:
                if p.grad is not None:
                    if set_to_none:
                        p.grad = None
                    else:
                        p.grad.detach_()
                        p.grad.zero_()

    def __getstate__(self):
        return {
            'base_optimizer': self.base_optimizer,
            'clipping': self.clipping,
            'eps': self.eps,
            'agc_params': self.agc_params,
        }

    def __setstate__(self, state):
        self.__dict__.update(state)
        self.param_groups = self.base_optimizer.param_groups
        self.state = self.base_optimizer.state
