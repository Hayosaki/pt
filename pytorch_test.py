import torch
import torch.nn as nn
import torch.nn.functional as f
import torch.optim as optim
from torch.utils import data
from torch.utils.tensorboard import SummaryWriter
import torchvision.datasets
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np

writer = SummaryWriter("logs")

# 使用matplot展示图片
def imshow(img):
    # 反归一化再转换为numpy格式
    img = img / 2 + 0.5
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.show()


class MyNet(nn.Module):
    def __init__(self):
        super(MyNet, self).__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(400, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        x = self.pool(f.relu(self.conv1(x)))
        x = self.pool(f.relu(self.conv2(x)))
        x = x.view(-1, 400)
        x = f.relu(self.fc1(x))
        x = f.relu(self.fc2(x))
        x = self.fc3(x)
        return x


def load_dataset():
    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
    train_set = torchvision.datasets.CIFAR10('./data', train=True, transform=transform)
    train_loader = data.DataLoader(train_set, batch_size=8, shuffle=True, num_workers=2)
    test_set = torchvision.datasets.CIFAR10(root='./data', train=False, transform=transform)
    test_loader = data.DataLoader(test_set, batch_size=8, shuffle=False, num_workers=2)
    return train_loader, test_loader


def train(epochs: int, load_dict: bool = False, ckpt_path: str = None):
    net = MyNet()
    # if load_dict:
    #     net.load_state_dict(torch.load(ckpt_path))
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(net.parameters(), lr=1e-4, momentum=0.9)
    train_loader, test_loader = load_dataset()
    for epoch in range(epochs):
        running_loss = running_acc = 0
        for i, batch in enumerate(train_loader, 0):
            inputs, labels = batch
            optimizer.zero_grad()
            outputs = net(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            # writer.add_scalar("loss", loss, epoch * len(train_loader) + i)
            _, predicted = torch.max(outputs.data, 1)
            running_acc += (predicted == labels).sum().item()
            running_loss += loss.item()
            log_step = 1000
            if i % log_step == log_step - 1:
                print('[epochs %d, iterations %5d] loss: %.3f, accuracy: %.3f' % (epoch + 1, i + 1, running_loss / log_step, running_acc / log_step / train_loader.batch_size))
                running_loss = running_acc = 0
    # writer.close()
    # torch.save(net.state_dict(), f'./checkpoint/SampleNet-{int(epochs * len(train_loader) / 1000)}k.pth')


# def single_test(net: nn.Module, *, load_dict: bool = False, ckpt_path: str = None):
#     _, test_loader = load_dataset()
#     if load_dict:
#         net.load_state_dict(torch.load(ckpt_path))
#     net.eval()
#     classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')
#     for batch in test_loader:
#         images, labels = batch
#
#         imshow(torchvision.utils.make_grid(images))
#         print('GroundTruth: ', ' '.join('%5s' % classes[labels[j]] for j in range(test_loader.batch_size)))
#
#         outputs = net(images)
#         predicted = torch.max(outputs, 1)[1]
#         print('Predicted: ', ' '.join('%5s' % classes[predicted[j]] for j in range(test_loader.batch_size)))


if __name__ == "__main__":
    train(3)