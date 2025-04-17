import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from smooth_cross_entropy import smooth_crossentropy
train_on_gpu = torch.cuda.is_available()
from torchvision import datasets
import torchvision.transforms as transforms
from torch.utils.data.sampler import SubsetRandomSampler
from torchvision import datasets, models, transforms
from SAM import SAM
from torch.utils.tensorboard import SummaryWriter
writer = SummaryWriter()
num_workers = 0
batch_size = 20
valid_size = 0.2
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

train_data = datasets.CIFAR10('data', train=True,
                              download=True, transform=transform)
test_data = datasets.CIFAR10('data', train=False,
                             download=True, transform=transform)

num_train = len(train_data)
indices = list(range(num_train))
np.random.shuffle(indices)
split = int(np.floor(valid_size * num_train))
train_idx, valid_idx = indices[split:], indices[:split]
train_sampler = SubsetRandomSampler(train_idx)
valid_sampler = SubsetRandomSampler(valid_idx)
train_loader = torch.utils.data.DataLoader(train_data, batch_size=batch_size,
    sampler=train_sampler, num_workers=num_workers)
valid_loader = torch.utils.data.DataLoader(train_data, batch_size=batch_size,
    sampler=valid_sampler, num_workers=num_workers)
test_loader = torch.utils.data.DataLoader(test_data, batch_size=batch_size,
    num_workers=num_workers)
classes = ['airplane', 'automobile', 'bird', 'cat', 'deer',
           'dog', 'frog', 'horse', 'ship', 'truck']

class Model(nn.Module):
    def __init__(self):
        super().__init__()
        base = models.resnet18(pretrained=True)
        self.base = nn.Sequential(*list(base.children())[:-1])
        in_features = base.fc.in_features
        self.drop = nn.Dropout()
        self.final = nn.Linear(in_features, 10)

    def forward(self, x):
        x = self.base(x)
        x = self.drop(x.view(-1, self.final.in_features))
        return self.final(x)

model = Model()
if train_on_gpu:
    model.cuda()

rho = 0.05
momentum = 0.9
base_optimizer = torch.optim.SGD
weight_decay = 0.0005
optimizer = SAM(model.parameters(), base_optimizer, rho=rho, lr=0.001, momentum=momentum,
                weight_decay=weight_decay)
n_epochs = 30
valid_loss_min = np.Inf  

for epoch in range(1, n_epochs + 1):
    train_loss = 0.0
    valid_loss = 0.0
    model.train()
    for data, target in train_loader:
        if train_on_gpu:
            data, target = data.cuda(), target.cuda()
        output = model(data)
        loss = smooth_crossentropy(output, target)
        loss.mean().backward()
        optimizer.first_step(zero_grad=True)

        smooth_crossentropy(model(data), target).mean().backward()
        optimizer.second_step(zero_grad=True)
        train_loss += loss.mean().item() * data.size(0)

    model.eval()
    for data, target in valid_loader:
        if train_on_gpu:
            data, target = data.cuda(), target.cuda()
        output = model(data)
        loss = smooth_crossentropy(output, target)
        valid_loss += loss.mean().item() * data.size(0)

    train_loss = train_loss / len(train_loader.sampler)
    valid_loss = valid_loss / len(valid_loader.sampler)

    print('Epoch: {} \tTraining Loss: {:.6f} \tValidation Loss: {:.6f}'.format(
        epoch, train_loss, valid_loss))

    if valid_loss <= valid_loss_min:
        print('Validation loss decreased ({:.6f} --> {:.6f}).  Saving model ...'.format(
            valid_loss_min,
            valid_loss))
        torch.save(model.state_dict(), 'model_cifar_SAM.pt')
        valid_loss_min = valid_loss
        writer.add_scalar('training loss',
                          train_loss,
                          epoch * len(train_loader))
        writer.add_scalar('valid loss',
                          valid_loss,
                          epoch * len(valid_loader))

model = Model()

model.load_state_dict(torch.load('model_cifar_SAM.pt'))

if torch.cuda.is_available():
    model.cuda()
num_workers = 0
batch_size = 20
valid_size = 0.2

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

test_data = datasets.CIFAR10('data', train=False,
                             download=True, transform=transform)
test_loader = torch.utils.data.DataLoader(test_data, batch_size=batch_size,
    num_workers=num_workers)


train_on_gpu = torch.cuda.is_available()


model.eval()
test_loss = 0.0
class_correct = list(0. for i in range(10))
class_total = list(0. for i in range(10))
for data, target in test_loader:
    if train_on_gpu:
        data, target = data.cuda(), target.cuda()
    output = model(data)
    loss = smooth_crossentropy(output, target)
    test_loss += loss.mean().item()*data.size(0)
    _, pred = torch.max(output, 1)
    correct_tensor = pred.eq(target.data.view_as(pred))
    correct = np.squeeze(correct_tensor.numpy()) if not train_on_gpu else np.squeeze(correct_tensor.cpu().numpy())
    for i in range(batch_size):
        label = target.data[i]
        class_correct[label] += correct[i].item()
        class_total[label] += 1

classes = ['airplane', 'automobile', 'bird', 'cat', 'deer',
           'dog', 'frog', 'horse', 'ship', 'truck']
test_loss = test_loss/len(test_loader.dataset)
print('Test Loss: {:.6f}\n'.format(test_loss))
for i in range(10):
    if class_total[i] > 0:
        print('Test Accuracy of %5s: %2d%% (%2d/%2d)' % (
            classes[i], 100 * class_correct[i] / class_total[i],
            np.sum(class_correct[i]), np.sum(class_total[i])))
    else:
        print('Test Accuracy of %5s: N/A (no training examples)' % (classes[i]))

print('\nTest Accuracy (Overall): %2d%% (%2d/%2d)' % (
    100. * np.sum(class_correct) / np.sum(class_total),
    np.sum(class_correct), np.sum(class_total)))
