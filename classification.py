import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import cv2
import glob
import random
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import copy
import numpy as np
import albumentations as A
from torchvision import models
from torch.utils.tensorboard import SummaryWriter

writer = SummaryWriter('./runs/ResNet-50')
def flatten(t):
    return [item for sublist in t for item in sublist]
train_data_path = './Hand_data/224/train'
train_image_paths = []
classes= []
# classes_validation = []
validation_data_path = './Hand_data/224/validation'
validation_image_paths = []

for data_path in glob.glob(train_data_path+'/*'):
    classes.append(data_path.split('/')[-1])
    train_image_paths.append(glob.glob(data_path+'/*'))
for data_path in glob.glob(validation_data_path+'/*'):
    # classes_validation.append(data_path.split('/')[-1])
    validation_image_paths.append(glob.glob(data_path+'/*'))

train_image_paths = list(flatten(train_image_paths))
validation_image_paths = list(flatten(validation_image_paths))
# random.shuffle(train_image_paths)
# train_image_paths, validation_image_paths = train_image_paths[:int(len(train_image_paths))], train_image_paths[int(len(train_image_paths)):] 

idx_to_class = {i:j for i, j in enumerate(classes)}
class_to_idx = {value:key for key,value in idx_to_class.items()}

mean = [0.5071, 0.4867, 0.4408]
stdv = [0.2675, 0.2565, 0.2761]
transforms = transforms.Compose([transforms.ToTensor(),transforms.Normalize(mean=mean, std=stdv)])

class HandDataset(Dataset):
    def __init__(self, image_paths, transform=False):
        self.image_path = image_paths
        self.transform = transform
    
    def __len__(self):
        return len(self.image_path)
    
    def __getitem__(self, idx):
        image_filepath = self.image_path[idx]
        image = cv2.imread(image_filepath)
        image = cv2.resize(image,(224,224),interpolation=cv2.INTER_CUBIC)
        # image = cv2.resize(image,(224,224),interpolation=cv2.INTER_CUBIC)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        label = image_filepath.split('/')[-2]
        label = class_to_idx[label]

        if self.transform is not None:
            image = self.transform(image)
        return image, label

train_dataset = HandDataset(train_image_paths, transforms)
valid_dataset = HandDataset(validation_image_paths, transforms)
batch_size = 32
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)
valid_loader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=True, num_workers=0)

#------Train------
# class Net(nn.Module):
#     def __init__(self):
#         super().__init__()
#         self.conv1 = nn.Conv2d(3, 12, 3, 1, 1)
#         self.pool = nn.MaxPool2d(2,2)
#         self.conv2 = nn.Conv2d(12, 24, 3, 1, 1)
#         self.drop = nn.Dropout2d(p=0.2)
#         self.fc1 = nn.Linear(32*32*24, 120)
#         self.fc1_bn=nn.BatchNorm1d(120)
#         self.fc2 = nn.Linear(120,84)
#         self.fc2_bn=nn.BatchNorm1d(84)
#         self.fc3 = nn.Linear(84,6)
#         self.fcdrop = nn.Dropout(0.5)
#     def forward(self, x):
#         x = F.relu(self.pool(self.conv1(x)))
#         x = F.relu(self.pool(self.conv2(x)))
#         # x = F.dropout(self.drop(x))
#         x = torch.flatten(x,1)
#         x = x.view(-1, 32*32*24)
#         # x = F.relu(self.fc1_bn(self.fc1(x)))
#         x = F.relu(self.fc1(x))
#         # x = F.dropout(self.fcdrop(x))
#         # x = F.relu(self.fc2_bn(self.fc2(x)))
#         x = F.relu(self.fc2(x))
#         # x = F.dropout(self.fcdrop(x))
#         x = self.fc3(x)
#         return x

# resnet
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model = models.resnet18(pretrained=True).to(device)
# model = models.resnet50().to(device)
for param in model.parameters():
    param.requires_grad = False
num_ftrs = model.fc.in_features
model.fc = nn.Sequential(
    nn.Linear(num_ftrs, 128),
    nn.BatchNorm1d(128),
    nn.ReLU(),
    nn.Dropout(),
    nn.Linear(128,84),
    nn.BatchNorm1d(84),
    nn.ReLU(),
    nn.Dropout(),
    nn.Linear(84,6)
).to(device)

# model = Net()
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)
# optimizer = optim.SGD(model.fc.parameters(), lr=0.001, momentum=0.9)
criterion = nn.CrossEntropyLoss()
model.to(device)


def validation(PATH,epoch):
    net = model
    net.load_state_dict(torch.load(PATH))
    net.eval()
    net.to(device)
    # model.load_state_dict(torch.load(PATH))
    # model.eval()
    correct = 0
    total = 0
    # since we're not training, we don't need to calculate the gradients for our outputs
    with torch.no_grad():
        for data in valid_loader:
            images, labels = data
            images, labels = images.to(device), labels.to(device)
            # calculate outputs by running images through the network
            outputs = net(images)
            # print(outputs.data.max(1,keepdim=True))
            # the class with the highest energy is what we choose as prediction
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    print('Accuracy of the network on the 360 test images: %d %%' % (
        100 * correct / total))
    writer.add_scalar('validation_rate',
                            100 * correct / total,
                            epoch)

for epoch in range(120):  # loop over the dataset multiple times
    net = model
    net.train()
    running_loss = 0.0
    for i, data in enumerate(train_loader, 0):
        # get the inputs; data is a list of [inputs, labels]
        inputs, labels = data[0].to(device), data[1].to(device)
        # zero the parameter gradients
        optimizer.zero_grad()

        # forward + backward + optimize
        outputs = net(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        # print statistics
        running_loss += loss.item()
        if i%10 == 9:
            print('[%d, %5d] loss: %.3f' %
                (epoch + 1, i + 1, running_loss / 10))
            writer.add_scalar('training loss',
                            running_loss / 10,
                            epoch * len(train_loader) + i)
            running_loss = 0.0
    PATH = './CNN_classification/ResNet18_%d.pth'%epoch
    torch.save(net.state_dict(), PATH)
    if epoch%5 == 0:
        validation(PATH,epoch)
print('Finished Training')


