import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import math
from sklearn.metrics import r2_score
class LinearRegression(nn.Module):
    def __init__(self, input_size, output_size):
        super(LinearRegression, self).__init__()
        self.linear = nn.Linear(input_size, output_size)

    def forward(self, x):
        return self.linear(x)


class SAM(torch.optim.Optimizer):
    def __init__(self, params, base_optimizer, rho):
        assert isinstance(base_optimizer, torch.optim.Optimizer), "base_optimizer must be an `Optimizer`"
        self.base_optimizer = base_optimizer
        self.rho = rho
        super(SAM, self).__init__(params, dict(rho=rho))
        self.param_groups = self.base_optimizer.param_groups
        for group in self.param_groups:
            group["rho"] = rho

    @torch.no_grad()
    def first_step(self, zero_grad=False):
        grad_norm = self._grad_norm()
        for group in self.param_groups:
            scale = group["rho"] / (grad_norm + 1e-7)
            for p in group["params"]:
                if p.grad is None:  
                    continue
                e_w = p.grad * scale
                if "e_w" not in self.state[p]:
                    self.state[p]["e_w"] = torch.zeros_like(p)
                self.state[p]["e_w"].copy_(e_w)
                p.add_(e_w)  
        if zero_grad:
            self.zero_grad()

    @torch.no_grad()
    def second_step(self, zero_grad=False):
        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is None:
                    continue
                if "e_w" not in self.state[p]:
                    raise KeyError("first_step must be called before second_step")
                p.sub_(self.state[p]["e_w"]) 

        self.base_optimizer.step()
        if zero_grad:
            self.zero_grad()

    @torch.no_grad()
    def step(self, closure=None, **kwargs):
        assert closure is not None, "SAM requires closure, which is not provided."
        self.first_step(True)
        with torch.enable_grad():
            closure()
        self.second_step()

    def _grad_norm(self):
        shared_device = self.param_groups[0]["params"][0].device
        norm_list = [
            p.grad.norm(p=2).to(shared_device)
            for group in self.param_groups
            for p in group["params"]
            if p.grad is not None
        ]
        if not norm_list: 
            return torch.tensor(0.0, device=shared_device)
        return torch.norm(torch.stack(norm_list), p=2)

class SSAMF(SAM):
    def __init__(self, params, base_optimizer, rho, sparsity, num_samples, update_freq):
        super(SSAMF, self).__init__(params, base_optimizer, rho)
        self.sparsity = sparsity
        self.num_samples = num_samples
        self.update_freq = update_freq
        self._initialized = False 

    @torch.no_grad()
    def init_mask(self):
        for group in self.param_groups:
            for p in group["params"]:
                if "mask" not in self.state[p]:
                    self.state[p]["mask"] = torch.ones_like(p, requires_grad=False).to(p.device)
        self._initialized = True

    @torch.no_grad()
    def update_mask(self, epoch):
        if not self._initialized:
            raise RuntimeError("Masks are not initialized. Call `init_mask()` before training.")

        for group in self.param_groups:
            for p in group["params"]:
                if "mask" in self.state[p]:
                    mask = torch.rand_like(p) < self.sparsity
                    self.state[p]["mask"].copy_(mask.float())

    @torch.no_grad()
    def first_step(self, zero_grad=False):
        if not self._initialized:
            raise RuntimeError("Masks are not initialized. Call `init_mask()` before training.")

        grad_norm = self._grad_norm()
        for group in self.param_groups:
            scale = group["rho"] / (grad_norm + 1e-7)
            for p in group["params"]:
                if p.grad is None:
                    continue
                e_w = p.grad * scale
                e_w.data *= self.state[p]["mask"]
                if "e_w" not in self.state[p]:
                    self.state[p]["e_w"] = torch.zeros_like(p)
                self.state[p]["e_w"].copy_(e_w)
                p.add_(e_w) 
        if zero_grad:
            self.zero_grad()

    @torch.no_grad()
    def step(self, closure=None, epoch=None, batch_idx=None):
        assert closure is not None, "SAM requires closure, which is not provided."
        if not self._initialized:
            raise RuntimeError("Masks are not initialized. Call `init_mask()` before training.")

        self.first_step(True)

        if epoch is not None and batch_idx is not None:
            if epoch % self.update_freq == 0 and batch_idx == 0:
                self.update_mask(epoch)

        with torch.enable_grad():
            closure()
        self.second_step()

    @torch.no_grad()
    def mask_info(self):
        live_num = 0
        total_num = 0
        for group in self.param_groups:
            for p in group["params"]:
                if "mask" in self.state[p]:
                    live_num += self.state[p]["mask"].sum().item()
                    total_num += self.state[p]["mask"].numel()
        return float(live_num) / total_num if total_num > 0 else 0.0

class SSAMD(SAM):
    def __init__(self, params, base_optimizer, rho, sparsity, drop_rate, drop_strategy, growth_strategy, update_freq):
        super(SSAMD, self).__init__(params, base_optimizer, rho)
        self.sparsity = sparsity
        self.drop_rate = drop_rate
        self.drop_strategy = drop_strategy
        self.growth_strategy = growth_strategy
        self.update_freq = update_freq

    @torch.no_grad()
    def init_mask(self):
        random_scores = []
        for group in self.param_groups:
            for p in group["params"]:
                self.state[p]['score'] = torch.rand(size=p.shape).cpu().data
                random_scores.append(self.state[p]['score'])
        random_scores = torch.cat([torch.flatten(x) for x in random_scores])
        live_num = len(random_scores) - math.ceil(len(random_scores) *self.sparsity)
        _value, _index = torch.topk(random_scores, live_num)

        mask_list = torch.zeros_like(random_scores)
        mask_list.scatter_(0, _index, torch.ones_like(_value))
        start_index = 0
        for group in self.param_groups:
            for p in group['params']:
                self.state[p]['mask'] = mask_list[start_index: start_index + p.numel()].reshape(p.shape)
                self.state[p]['mask'] = self.state[p]['mask'].to(p)
                self.state[p]['mask'].require_grad = False
                del self.state[p]['score']
                start_index = start_index + p.numel()
                assert self.state[p]['mask'].max() <= 1.0 and self.state[p]['mask'].min() >= 0.0
        assert start_index == len(mask_list)

    @torch.no_grad()
    def DeathRate_Scheduler(self, epoch):
        dr = (self.drop_rate) * (1 + math.cos(math.pi * (float(epoch - self.T_start) / (self.T_end - self.T_start)))) / 2
        return dr

    @torch.no_grad()
    def update_mask(self, epoch, **kwargs):
        death_scores = []
        growth_scores =[]
        for group in self.param_groups:
            for p in group['params']:
                death_score = self.get_score(p, self.drop_strategy)
                death_scores.append((death_score + 1e-7) * self.state[p]['mask'].data)

                growth_score = self.get_score(p, self.growth_strategy)
                growth_scores.append((growth_score + 1e-7) * (1 - self.state[p]['mask'].data))
        '''
            Death
        '''
        death_scores = torch.cat([torch.flatten(x) for x in death_scores])
        death_rate = self.DeathRate_Scheduler(epoch=epoch)
        death_num = int(min((len(death_scores) - len(death_scores) * self.sparsity)* death_rate, len(death_scores) * self.sparsity))
        d_value, d_index = torch.topk(death_scores, int((len(death_scores) - len(death_scores) * self.sparsity) * (1 - death_rate)))

        death_mask_list = torch.zeros_like(death_scores)
        death_mask_list.scatter_(0, d_index, torch.ones_like(d_value))
        '''
            Growth
        '''
        growth_scores = torch.cat([torch.flatten(x) for x in growth_scores])
        growth_num = death_num
        g_value, g_index = torch.topk(growth_scores, growth_num)

        growth_mask_list = torch.zeros_like(growth_scores)
        growth_mask_list.scatter_(0, g_index, torch.ones_like(g_value))

        '''
            Mask
        '''
        start_index = 0
        for group in self.param_groups:
            for p in group['params']:
                death_mask = death_mask_list[start_index: start_index + p.numel()].reshape(p.shape)
                growth_mask = growth_mask_list[start_index: start_index + p.numel()].reshape(p.shape)

                self.state[p]['mask'] = death_mask + growth_mask
                self.state[p]['mask'] = self.state[p]['mask'].to(p)
                self.state[p]['mask'].require_grad = False
                start_index = start_index + p.numel()
                assert self.state[p]['mask'].max() <= 1.0 and self.state[p]['mask'].min() >= 0.0


        assert start_index == len(death_mask_list)

    def get_score(self, p, score_model=None):
        if score_model == 'weight':
            return torch.abs(p.clone()).cpu().data
        elif score_model == 'gradient':
            return torch.abs(p.grad.clone()).cpu().data
        elif score_model == 'random':
            return torch.rand(size=p.shape).cpu().data
        else:
            raise KeyError

    @torch.no_grad()
    def first_step(self, zero_grad=False):
        grad_norm = self._grad_norm()
        for group in self.param_groups:
            scale = group["rho"] / (grad_norm + 1e-7)
            for p in group["params"]:
                if p.grad is None:
                    continue
                e_w = p.grad * scale
                e_w.data = e_w.data * self.state[p]['mask'] 
                if "e_w" not in self.state[p]:
                    self.state[p]["e_w"] = torch.zeros_like(p)
                self.state[p]["e_w"].copy_(e_w)
                p.add_(e_w)
        if zero_grad:
            self.zero_grad()

    @torch.no_grad()
    def step(self, closure=None, **kwargs):
        assert closure is not None, "SAM requires closure, which is not provided."

        self.first_step(True)
        self.zero_grad()
        with torch.enable_grad():
            closure()
        self.second_step()

    @torch.no_grad()
    def mask_info(self):
        live_num = 0
        total_num = 0
        for group in self.param_groups:
            for p in group['params']:
                live_num += self.state[p]['mask'].sum().item()
                total_num += self.state[p]['mask'].numel()
        return float(live_num) / total_num

def train_with_accuracy(model, optimizer, criterion, dataloader, epochs):
    losses = []
    accuracies = []

    for epoch in range(epochs):
        epoch_loss = 0
        y_true_all = []
        y_pred_all = []

        for x_batch, y_batch in dataloader:
            def closure():
                optimizer.zero_grad()
                y_pred = model(x_batch)
                loss = criterion(y_pred, y_batch)
                loss.backward()
                return loss
ss
            loss = closure()
            optimizer.step(closure)
            epoch_loss += loss.item()

            y_pred = model(x_batch).detach().numpy()
            y_true = y_batch.numpy()
            y_pred_all.extend(y_pred)
            y_true_all.extend(y_true)

        losses.append(epoch_loss / len(dataloader))
        r2 = r2_score(y_true_all, y_pred_all)
        accuracies.append(r2)

        print(f"Epoch {epoch + 1}, Loss: {losses[-1]:.4f}, R²: {r2:.4f}")

    return losses, accuracies


torch.manual_seed(42)
input_size = 10
output_size = 1
num_samples = 100
x = torch.randn(num_samples, input_size)
true_y =x @ torch.randn(input_size, output_size)
y = true_y + torch.randn(num_samples, output_size) * 0.1
dataset = torch.utils.data.TensorDataset(x, y)
dataloader = torch.utils.data.DataLoader(dataset, batch_size=16, shuffle=True)
epochs = 20

models = {
   "SSAMF": LinearRegression(input_size, output_size),
   "SSAMD": LinearRegression(input_size, output_size),
optimizers = {
    "SSAMF": SSAMF(models["SSAMF"].parameters(), base_optimizer=optim.SGD(models["SSAMF"].parameters(), lr=0.01), rho=0.5, sparsity=0.1, num_samples=5, update_freq=10),
    "SSAMD": SSAMD(models["SSAMD"].parameters(), base_optimizer=optim.SGD(models["SSAMD"].parameters(), lr=0.01), rho=0.5, sparsity=0.1, drop_rate=0.2, drop_strategy="gradient", growth_strategy="weight", update_freq=10)
}
optimizers['SSAMF'].init_mask()
optimizers['SSAMD'].init_mask()

criterion = nn.MSELoss()

all_losses = {}
all_accuracies = {}

for name, model in models.items():
    print(f"Training {name}...")
    losses, accuracies = train_with_accuracy(model, optimizers[name], criterion, dataloader, epochs)


