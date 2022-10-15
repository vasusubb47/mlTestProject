import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
import torch
from torch import nn
from torch.utils.data import Dataset, random_split, DataLoader
import torchvision.transforms as trans
import torchvision

print(f"pytorch ver : {torch.__version__}")

# device diagnostics code
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'device : {device}')
# print(os.system('nvidia-smi'))

'''
# Classification of images of dogs and cats

## Steps
    1. Create data set of 300+ images of each cat and dog
    2. Create Dataset and DataLoader of batch 32 with training, validation and testing set
    3. Create a model
    4. Train the model and save
    5. Predict using model
'''

'''

# clean the data and create the csv file (once), later we can just load the csv

'''

dataDir = 'data'


def cleanAndLoadMetadata():
    metadata = None
    createCsv = True
    flCount = 0
    subDirs = [str]
    metadataFn = 'data.csv'

    for _, dirs, files in os.walk(dataDir):
        subDirs = dirs

        for subDir in subDirs:
            flCount += len(os.listdir(os.path.join(dataDir, subDir)))
        print(f'flCount : {flCount}')

        if files and (metadataFn in files):
            # load the csv
            print("Loading csv")
            metadata = pd.read_csv(os.path.join(dataDir, 'data.csv'))
            print(f'metadata : \n{metadata} ,\n size : {metadata.size}')

            if len(metadata) == flCount:
                createCsv = False
        break

    if createCsv:

        print(f'subDirs : {subDirs}')

        if isinstance(metadata, pd.DataFrame):
            print(f'The csv file is corrupted or new files are added in data folder so the csv file is deleted '
                  f'and rebuilt...')

        metadata = pd.DataFrame({
            'fileName': [None] * flCount,
            'class': [None] * flCount
        })
        ind = 0

        for subDir in subDirs:
            for imgFn in os.listdir(os.path.join(dataDir, subDir)):
                try:
                    cv2.imread(os.path.join(dataDir, subDir, imgFn))
                    metadata.loc[ind, ['fileName', 'class']] = [imgFn, subDir]
                    ind += 1
                except Exception as e:
                    print(f'coz of error removing the file {imgFn}, from subDir {subDir}')
                    os.remove(os.path.join(dataDir, subDir, imgFn))
                    print(f'e : {e}')

        # saving the dataFrame to disk as csv to load later if the program is executed again
        metadata.to_csv(os.path.join(dataDir, metadataFn), index=False)

    return metadata


# TODO: Create custom dataset
class MyDataset(Dataset):

    def __init__(self, metadata, transform=None):
        super(MyDataset, self).__init__()
        self.metadata = metadata
        self.transform = transform

    def __len__(self):
        return len(self.metadata)

    def __getitem__(self, index):
        path = os.path.join(dataDir, self.metadata.iloc[index, 1], self.metadata.iloc[index, 0])
        img = cv2.cvtColor(cv2.imread(path), cv2.COLOR_BGR2RGB)  # load the image from given path
        # cv2.COLOR_BGR2RGB to convert BGR color channels to RGB color channels
        img = torch.from_numpy(img).to(device)

        # divide by 255 so that max value will be 1, so that it's easier for gpu to work with
        img = img / torch.tensor(255).to(device)  # divide by 255 to make the range from 0-255 to 0-1

        # width, height, color_channel -> color_channel, width, height
        img = img.permute(2, 0, 1)

        if self.transform:
            img = self.transform(img)

        return img, self.metadata.iloc[index, 1]

    def display(self, index):
        img, label = self[index]

        # color_channel, width, height -> width, height, color_channel
        plt.imshow(img.permute(1, 2, 0).cpu())  # `.cpu()` is for coping img data from cuda to cpu,
        # if the img was in cpu itself it doesn't raise error

        plt.title(f'class : {label}, index : {index}')
        plt.show()


dataset = MyDataset(
    cleanAndLoadMetadata(),
    trans.Compose([
        trans.Resize((224, 224)),  # resize the tensor to 255 255 (width and height)
        trans.RandomHorizontalFlip(),  # randomly flip horizontally
        # for training the same image in different ways to improve performance

        trans.Normalize([0.5, 0.5, 0.5],  # to convert 0-1 to [-1, 1], formula (mean)/std
                        [0.5, 0.5, 0.5])
    ])
)

trainCount = int(.7 * len(dataset))
validationCount = int(.2 * len(dataset))
testCount = len(dataset) - trainCount - validationCount
print(f'trainSet count : {trainCount},\n validationSet count : {validationCount},\n testSet count : {testCount}')

batchSize = 32
numWorkers = 4

trainDataSet, validationDataSet, testingDataSet = random_split(
    dataset,
    (trainCount, validationCount, testCount)
)

trainDL = DataLoader(
    trainDataSet,
    batch_size=batchSize,
    shuffle=True,
    num_workers=numWorkers
)

validationDL = DataLoader(
    validationDataSet,
    batch_size=batchSize,
    shuffle=True,
    num_workers=numWorkers
)

testDL = DataLoader(
    testingDataSet,
    batch_size=batchSize,
    shuffle=True,
    num_workers=numWorkers
)


dls = {
    'train': trainDL,
    'validation': validationDL,
    'test': testDL
}
print(f'DataLoaders : {dls}')
