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

        death_scores = torch.cat([torch.flatten(x) for x in death_scores])
        death_rate = self.DeathRate_Scheduler(epoch=epoch)
        death_num = int(min((len(death_scores) - len(death_scores) * self.sparsity)* death_rate, len(death_scores) * self.sparsity))
        d_value, d_index = torch.topk(death_scores, int((len(death_scores) - len(death_scores) * self.sparsity) * (1 - death_rate)))

        death_mask_list = torch.zeros_like(death_scores)
        death_mask_list.scatter_(0, d_index, torch.ones_like(d_value))

        growth_scores = torch.cat([torch.flatten(x) for x in growth_scores])
        growth_num = death_num
        g_value, g_index = torch.topk(growth_scores, growth_num)

        growth_mask_list = torch.zeros_like(growth_scores)
        growth_mask_list.scatter_(0, g_index, torch.ones_like(g_value))

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
class ImbSAM(SAM):
    def __init__(self, model, base_optimizer, rho=0.05):
        super(ImbSAM, self).__init__(model.parameters(), base_optimizer, rho)
        self.model = model
        self.optimizer = base_optimizer 

    @torch.no_grad()
    def first_step(self, zero_grad=False):
        for n, p in self.model.named_parameters():
            if p.grad is None:
                continue
            grad_normal = self.state[p].get("grad_normal")
            if grad_normal is None:
                self.state[p]["grad_normal"] = torch.clone(p.grad).detach()
            else:
                self.state[p]["grad_normal"].copy_(p.grad)
        if zero_grad:
            self.optimizer.zero_grad()

    @torch.no_grad()
    def second_step(self):
        grads = []
        for n, p in self.model.named_parameters():
            if p.grad is None:
                continue
            grads.append(torch.norm(p.grad, p=2))  

        grad_norm = torch.norm(torch.stack(grads), p=2) + 1.e-7  

        for n, p in self.model.named_parameters():
            if p.grad is None:
                continue

            eps = self.state[p].get("eps")
            if eps is None:
                self.state[p]["eps"] = torch.clone(p.grad).detach()
            else:
                self.state[p]["eps"].copy_(p.grad)

            self.state[p]["eps"].mul_(self.rho / grad_norm)
            self.state[p]["eps"].clamp_(-0.1, 0.1)  
            p.add_(self.state[p]["eps"])  

        self.optimizer.zero_grad()

    @torch.no_grad()
    def third_step(self):
        for n, p in self.model.named_parameters():
            if p.grad is None:
                continue
            p.sub_(self.state[p]["eps"])

            if "grad_normal" not in self.state[p]:
                raise KeyError("grad_normal missing from state")

            p.grad.add_(self.state[p]["grad_normal"])

        self.optimizer.step()
        self.optimizer.zero_grad()


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
    "SGD": LinearRegression(input_size, output_size),
    "SAM": LinearRegression(input_size, output_size),
    "SSAMF": LinearRegression(input_size, output_size),
    "FSAM": LinearRegression(input_size, output_size),
}

optimizers = {
    "SGD": optim.SGD(models["SGD"].parameters(), lr=0.01),
    "SAM": SAM(models["SAM"].parameters(), base_optimizer=optim.SGD(models["SAM"].parameters(), lr=0.01), rho=0.5),
    "SSAMF": SSAMF(models["SSAMF"].parameters(), base_optimizer=optim.SGD(models["SSAMF"].parameters(), lr=0.01), rho=0.5, sparsity=0.1, num_samples=5, update_freq=10),
    "FSAM": FriendlySAM(models["FSAM"].parameters(), base_optimizer=optim.SGD(models["FSAM"].parameters(), lr=0.01), rho=0.5),
}
optimizers['SSAMF'].init_mask()

criterion = nn.MSELoss()

all_losses = {}
all_accuracies = {}

for name, model in models.items():
    print(f"Training {name}...")
    losses, accuracies = train_with_accuracy(model, optimizers[name], criterion, dataloader, epochs)
    all_losses[name] = losses
    all_accuracies[name] = accuracies

plt.figure(figsize=(12, 6))

plt.subplot(1, 2, 1)
for name, losses in all_losses.items():
    plt.plot(losses, label=name)
plt.title("Training Loss Comparison")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.legend()

plt.subplot(1, 2, 2)
for name, accuracies in all_accuracies.items():
    plt.plot(accuracies, label=name)
plt.title("Training Accuracy (R²) Comparison")
plt.xlabel("Epoch")
plt.ylabel("R² Score")
plt.legend()

plt.tight_layout()
plt.show()
print()
optimizers = ['SGD', 'SAM', 'SSAMF', 'FSAM']
final_losses = [all_losses['SGD'][epochs-1],all_losses['SAM'][epochs-1],all_losses['SSAMF'][epochs-1],all_losses['FSAM'][epochs-1]]
final_r2_scores = [all_accuracies['SGD'][epochs-1],all_accuracies['SAM'][epochs-1],all_accuracies['SSAMF'][epochs-1],all_accuracies['FSAM'][epochs-1]]

x = np.arange(len(optimizers))
width = 0.35

fig, ax1 = plt.subplots(figsize=(10, 6))

bars1 = ax1.bar(x - width/2, final_losses, width, label='Loss', color='salmon')
ax1.set_xlabel('Optimizer')
ax1.set_ylabel('Loss', color='salmon')
ax1.tick_params(axis='y', labelcolor='salmon')
ax1.set_xticks(x)
ax1.set_xticklabels(optimizers)

for bar in bars1:
    height = bar.get_height()
    ax1.text(bar.get_x() + bar.get_width()/2, height, f'{height:.4f}', ha='center', va='bottom', color='salmon')

ax2 = ax1.twinx()
bars2 = ax2.bar(x + width/2, final_r2_scores, width, label='R²', color='lightgreen')
ax2.set_ylabel('R²', color='lightgreen')
ax2.tick_params(axis='y', labelcolor='lightgreen')
ax2.set_ylim(0.98, 1.0)

for bar in bars2:
    height = bar.get_height()
    ax2.text(bar.get_x() + bar.get_width()/2, height, f'{height:.4f}', ha='center', va='bottom', color='lightgreen')

plt.title('Comparison of Final Loss and R² Scores Across Optimizers')
fig.tight_layout()
plt.show()

