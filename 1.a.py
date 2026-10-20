import copy
import math

import torch


class CustomSGDMomentum(torch.optim.Optimizer):
    def __init__(self, params, lr=1e-3, momentum=0.9):
        defaults = dict(lr=lr, momentum=momentum)
        super().__init__(params, defaults)

    def step(self, closure=None):
        loss = None
        if closure is not None:
            loss = closure()

        for group in self.param_groups:
            lr = group['lr']
            momentum = group['momentum']
            for p in group['params']:
                if p.grad is None:
                    continue
                if not hasattr(p, 'momentum_buffer'):
                    p.momentum_buffer = torch.zeros_like(p.data)
                p.momentum_buffer.mul_(momentum).add_(p.grad)
                p.data.add_(p.momentum_buffer, alpha=-lr)
        return loss


class CustomAdam(torch.optim.Optimizer):
    def __init__(self, params, lr=1e-3, betas=(0.9, 0.999), eps=1e-8):
        defaults = dict(lr=lr, betas=betas, eps=eps)
        super().__init__(params, defaults)

    def step(self, closure=None):
        loss = None
        if closure is not None:
            loss = closure()

        for group in self.param_groups:
            lr = group['lr']
            beta1, beta2 = group['betas']
            eps = group['eps']
            for p in group['params']:
                if p.grad is None:
                    continue

                state = self.state[p]
                if len(state) == 0:
                    state['step'] = 0
                    state['exp_avg'] = torch.zeros_like(p.data)
                    state['exp_avg_sq'] = torch.zeros_like(p.data)

                state['step'] += 1
                exp_avg, exp_avg_sq = state['exp_avg'], state['exp_avg_sq']
                grad = p.grad.data

                exp_avg.mul_(beta1).add_(grad, alpha=1 - beta1)
                exp_avg_sq.mul_(beta2).addcmul_(grad, grad, value=1 - beta2)

                bias_correction1 = 1 - beta1 ** state['step']
                bias_correction2 = 1 - beta2 ** state['step']
                step_size = lr / bias_correction1
                denom = exp_avg_sq.sqrt() / math.sqrt(bias_correction2)
                p.data.addcdiv_(exp_avg, denom.add(eps), value=-step_size)

        return loss


def verify_adam_matches_pytorch():
    torch.manual_seed(0)
    model1 = torch.nn.Linear(3, 1)
    model2 = torch.nn.Linear(3, 1)
    model2.load_state_dict(copy.deepcopy(model1.state_dict()))

    x = torch.randn(8, 3)
    y = torch.randn(8, 1)

    opt1 = torch.optim.Adam(model1.parameters(), lr=1e-2)
    opt2 = CustomAdam(model2.parameters(), lr=1e-2)

    for _ in range(5):
        opt1.zero_grad()
        opt2.zero_grad()

        loss1 = ((model1(x) - y) ** 2).mean()
        loss2 = ((model2(x) - y) ** 2).mean()

        loss1.backward()
        loss2.backward()

        opt1.step()
        opt2.step()

    return torch.allclose(model1.weight, model2.weight, atol=1e-5) and torch.allclose(model1.bias, model2.bias, atol=1e-5)


if __name__ == '__main__':
    result = verify_adam_matches_pytorch()
    print(f'Adam Match: {result}')
