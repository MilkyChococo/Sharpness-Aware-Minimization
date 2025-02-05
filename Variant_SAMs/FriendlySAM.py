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
      
  
class FriendlySAM(torch.optim.Optimizer):
    def __init__(self, params, base_optimizer, rho, sigma=1, lmbda=0.5, adaptive=False):
        defaults = dict(rho=rho, adaptive=adaptive)
        super(FriendlySAM, self).__init__(params, defaults)
        self.base_optimizer = base_optimizer
        self.sigma = sigma
        self.lmbda = lmbda
        print ('FriendlySAM sigma:', self.sigma, 'lambda:', self.lmbda)

    @torch.no_grad()
    def first_step(self, zero_grad=False):

        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is None: continue
                grad = p.grad.clone()
                if not "momentum" in self.state[p]:
                    self.state[p]["momentum"] = grad
                else:
                    p.grad -= self.state[p]["momentum"] * self.sigma
                    self.state[p]["momentum"] = self.state[p]["momentum"] * self.lmbda + grad * (1 - self.lmbda)

        grad_norm = self._grad_norm()
        for group in self.param_groups:
            scale = group["rho"] / (grad_norm + 1e-12)

            for p in group["params"]:
                if p.grad is None: continue
                self.state[p]["old_p"] = p.data.clone()
                e_w = (torch.pow(p, 2) if group["adaptive"] else 1.0) * p.grad * scale.to(p)
                p.add_(e_w)

        if zero_grad: self.zero_grad()

    @torch.no_grad()
    def second_step(self, zero_grad=False):
        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is None: continue
                p.data = self.state[p]["old_p"]

        self.base_optimizer.step() 

        if zero_grad: self.zero_grad()

    @torch.no_grad()
    def step(self, closure=None):
        assert closure is not None, "Sharpness Aware Minimization requires closure, but it was not provided"
        closure = torch.enable_grad()(closure)

        self.first_step(zero_grad=True)
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
    "FSAM": LinearRegression(input_size, output_size),
}

optimizers = {
    "FSAM": FriendlySAM(models["FSAM"].parameters(), base_optimizer=optim.SGD(models["FSAM"].parameters(), lr=0.01), rho=0.5),
}
optimizers['SSAMF'].init_mask()

criterion = nn.MSELoss()

all_losses = {}
all_accuracies = {}

for name, model in models.items():
    print(f"Training {name}...")
    losses, accuracies = train_with_accuracy(model, optimizers[name], criterion, dataloader, epochs)
