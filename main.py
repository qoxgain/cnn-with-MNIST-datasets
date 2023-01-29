import torch
import torch.nn as nn
from torch.autograd import Variable
from torchvision import datasets,transforms


train_dataset = datasets.MNIST('data/',download=True,train=True,
                               transform=transforms.Compose([
                                   transforms.ToTensor(),
                                  transforms.Normalize((0.1307,), (0.3081,)),
                               ]))
test_dataset = datasets.MNIST('data/',download=True,
                              transform=transforms.Compose([
                                  transforms.ToTensor(),
                                  transforms.Normalize((0.1307,), (0.3081,)),
                              ]))

train_loader = torch.utils.data.DataLoader(train_dataset,batch_size=64,shuffle=True)
test_loader = torch.utils.data.DataLoader(test_dataset,batch_size=64,shuffle=True)


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 16, kernel_size=5, padding=2)
        self.conv2 = nn.Conv2d(16, 64, kernel_size=5)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=5)
        self.mp = nn.MaxPool2d(2)
        self.relu = nn.ReLU()
        self.fc1 = nn.Linear(128, 64)
        self.fc2 = nn.Linear(64, 10)
        self.logsoftmax = nn.LogSoftmax(dim=0)

    def forward(self, x):
        in_size = x.size(0)
        out = self.relu(self.mp(self.conv1(x)))
        out = self.relu(self.mp(self.conv2(out)))
        out = self.relu(self.conv3(out))
        out = out.view(in_size, -1)
        out = self.relu(self.fc1(out))
        out = self.fc2(out)
        return self.logsoftmax(out)


model = Net()
loss_fn = nn.NLLLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=1e-2, momentum=0.5)

iteration = 20
for epoch in range(iteration):
    for t, (data, target) in enumerate(train_loader):
        data, target = Variable(data), Variable(target)
        pred = model(data)
        loss = loss_fn(pred, target)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

# save model
save_path = 'file/net.dict'
torch.save({'iteration': iteration, 'optimizer_dict': optimizer.state_dict(), 'model_dict': model.state_dict()}, save_path)

# import model
new_model = Net()
new_optimizer = torch.optim.SGD(model.parameters(), lr=1e-2, momentum=0.5)
model_data = torch.load(save_path)
new_model.load_state_dict(model_data['model_dict'])
new_optimizer.load_state_dict(model_data['optimizer_dict'])

correct = 0
for data, target in test_loader:
    data, target = Variable(data), Variable(target)
    output = model(data)
    # get the index of the max log-probability
    pred = output.data.max(1, keepdim=True)[1]
    correct += pred.eq(target.data.view_as(pred)).cpu().sum()

print('{:.3f}%\n'.format(100. * correct / len(test_loader.dataset)))

