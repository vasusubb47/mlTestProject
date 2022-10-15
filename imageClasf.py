# %%
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
print("pyTorch ver : ", torch.__version__)

# %%
"""
## device acustic code
"""

# %%
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(device)

# %%
# !nvidia-smi

# %%
"""
# Classification of images whether the image is of cat and dog
"""

# %%
"""

## Steps
1. Create data set of 300+ images of cat and dog
2. Load images as batch of 32
3. Create a ml model to predict
4. Predict
"""

# %%
"""
# clean the data and create a csv
"""

# %%
import pandas as pd
import os

# %%
data = None
dir = 'data'

if not os.path.isfile(os.path.join(dir, 'data.csv')):
  print("creating csv")

  subDir = os.listdir(dir)

  dataDict = {}

  for classDir in subDir:
    if os.path.isdir(os.path.join(dir, classDir)):
      dataDict[classDir] = os.listdir(os.path.join(dir, classDir))
    else:
      print(f"not dir : {classDir}")

  data = pd.DataFrame({
    'fileName': [],
    'class': []
  })

  for class_ in dataDict:
    for fn in dataDict[class_]:
      data.loc[len(data.index)] = [fn, class_]

  csv = data.to_csv(os.path.join(dir, 'data.csv'), index=False)

else:

  print("loading csv")

  data = pd.read_csv(os.path.join(dir, 'data.csv'))

print(data)

# %%
"""
# Create custom dataset loader
"""

# %%
from torch.utils.data import Dataset, random_split, DataLoader
from torch import nn
import torchvision
import torchvision.transforms as trans

# %%

class MyDataset(Dataset):
  def __init__(self, df, transform=None):
    super(MyDataset, self).__init__()
    self.df = df
    self.transform = transform

  def __len__(self):
    return len(self.df)

  def __getitem__(self, index):
    path = os.path.join(dir, self.df.iloc[index, 1], self.df.iloc[index, 0])

    img = torchvision.io.read_image(path).to(device = device)

    y_label = torch.tensor(0 if (self.df.iloc[index, 1] == 'cat') else 1).to(device = device)

    if self.transform:
      img = self.transform(img)

    # # color_channel, width, height -> width, height, color_channel
    # img = img.permute(1, 2, 0)
    # divide by 255 so that max value will be 1, so that its easier for gpu to work with
    img = img / torch.tensor(255).to(device = device)

    return(img, y_label)

# %%
"""
# split the dataset into training, validation and testing sets
"""

# %%
# transform for the dataset
transform = trans.Resize(size = (224, 224))

# Create the dataset
ds = MyDataset(data, transform)


# %%
trainCount = int(.7 * len(ds))
validationCount = int(.2 * len(ds))
testCount = len(ds) - trainCount - validationCount

print(f'trainSet count : {trainCount},\n validationSet count : {validationCount},\n testSet count : {testCount}')

# %%
batchSize = 32
numWorkers = 4

trainDataSet, validationDataSet, testingDataSet = random_split(ds, (trainCount, validationCount, testCount))

print(f'trainSet : {trainDataSet},\n validationSet : {validationDataSet},\n testSet : {testingDataSet}')

# %%
trainDataLoder = DataLoader(
  trainDataSet,
  batch_size=batchSize,
  shuffle=True,
  num_workers=0
)

validationDataLoder = DataLoader(
  validationDataSet,
  batch_size=batchSize,
  shuffle=True,
  num_workers=0
)

testDataLoder = DataLoader(
  testingDataSet,
  batch_size=batchSize,
  shuffle=True,
  num_workers=0
)

dataLoaders = {
  'train': trainDataLoder,
  'validation': validationDataLoder,
  'test': testDataLoder
}

print(dataLoaders)

# %%
class myModle (nn.Module):
  def __init__ (self):
    super(myModle, self).__init__()
    #((w - f + 2p) / s) + 1 -> ((150 - 3 + 2*1) / 1) + 1 -> 224

    #in shape = (32, 3, 224, 224)
    self.l1 = nn.Conv2d(
      in_channels=3, out_channels=12,
      kernel_size=3, stride=1, padding=1
    )
    #shape = (32, 12, 224, 224)
    self.bn1 = nn.BatchNorm2d(num_features=12)
    #shape = (32, 12, 224, 224)

    self.pool1 = nn.MaxPool2d(kernel_size=2)
    # reduce the img by 2
    # shape = (32, 12, 112, 112)

    self.l2 = nn.Conv2d(
      in_channels=12, out_channels=32,
      kernel_size=3, stride=1, padding=1
    )
    # shape = (32, 32, 112, 112)

    self.l3 = nn.Conv2d(
      in_channels=32, out_channels=64,
      kernel_size=3, stride=1, padding=1
    )
    # shape = (32, 64, 112, 112)
    self.bn3 = nn.BatchNorm2d(num_features=64)

    self.out = nn.Linear(in_features=64*112*112, out_features=2)

    # cnn(3->12) -> bn -> pool(/2) -> cnn(12->32) -> cnn(32 -> 64) -> bn -> out([64*112*112]802816 -> 2)

  def forward(self, x):
    y = self.l1(x)
    y = self.bn1(y)
    y = nn.ReLU(y)

    y = self.pool1(y)

    y = self.l2(y)
    y = nn.ReLU(y)

    y = self.l3(y)
    y = self.bn3(y)
    y = nn.ReLU(y)

    y = y.view(-1, 64*112*112)

    y = self.out(y)

    return y


# %%
# ((224 - 3 + 2*1) / 1) + 1
# 64*112*112

# %%
model = myModle().to(device= device)
print(model)

# %%
"""
# Optimizer and loss function
"""

# %%
optim = torch.optim.Adam(model.parameters(), lr = 0.001, weight_decay=0.0001)
loss = nn.CrossEntropyLoss()

print(optim, loss)

# %%
epoch = 1000

# %%
acc = []
los = []

for e in range(epoch):

  tacc = 0.0
  tloss = 0.0

  # training
  model.train()

  for _, (img, label) in enumerate(dataLoaders['train']):
    optim.zero_grad()
    y = model(img)
    l = loss(y, label)
    l.backward()
    optim.step()

    tloss += l.cpu().data * img.size(0)
    _, pred = torch.max(y.data, 1)
    tacc += int(torch.sum(pred == label.data))

  acc.append(tacc / len(trainDataSet))
  los.append(tloss / len(trainDataSet))

  if e % 100 == 0:
    print(f'epoch : {e} | acc : {tacc / len(trainDataSet)} | loss : {tloss / len(trainDataSet)}')

  # evaluation
  model.eval()

  tacc = 0.0
  tloss = 0.0

# %%
