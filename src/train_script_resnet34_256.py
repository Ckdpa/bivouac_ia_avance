
import os
import random
import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler
from PIL import Image
import numpy as np

random.seed(5)

class LabelDataset(Dataset):
    def __init__(self, filename_list, mask_as_int, transform = None):
        self.mask_as_int = mask_as_int
        if transform != None:
            self.transform = transform
        else:
            self.transform = transforms.Compose([transforms.ToTensor()])
        self.image_filenames = [f for f in filename_list if f.endswith('.png')]
        self.label_filenames = [f for f in filename_list if f.endswith('.npy')]
        self.image_filenames.sort()
        self.label_filenames.sort()

    def __len__(self):
        return len(self.image_filenames)

    def __getitem__(self, idx):
        image = Image.open(self.image_filenames[idx])
        if self.mask_as_int:
            label = np.load(self.label_filenames[idx]).max()
        else :
            label = np.load(self.label_filenames[idx])
        label = torch.tensor(label)
        if (self.transform):
            image = self.transform(image)
        return image, label
    
    def remove_zeros(self, erase_percentage):
        to_remove = []
        labels = []
        for idx in range(len(self.label_filenames)):
            if np.load(self.label_filenames[idx]).max() == 0:
                if random.random() > 1 - erase_percentage:
                    to_remove.append(self.label_filenames[idx])
        for file in to_remove:
            self.label_filenames.remove(file)
            self.image_filenames.remove(file[:-4]+'.png')
    

def list_data_files(root_dir):
    files = [os.path.join(root_dir, f) for f in os.listdir(root_dir)]
    files.sort()
    return files

def train_test_val_split_list(files):
    train_end = round(len(files) * 0.7) + (round(len(files) * 0.7) % 2)
    test_end = round(len(files) * 0.15) + (round(len(files) * 0.15) % 2)
    return files[:train_end], files[train_end:train_end + test_end], files[train_end + test_end:]


def get_weights(data):
    d = np.array([d for _, d in data])
    class_sample_count = np.unique(d, return_counts=True)[1]
    weight = 1. / class_sample_count
    arr = []
    for i in d:
        arr.append(i)
    arr = np.array(arr)
    samples_weights = weight[arr]                                   
    return samples_weights  
    
def getOverSampler(data, mask_as_int = True,batch_size=16, remove_zeros = True, transform = None):
    if transform == None:
        transform = transforms.Compose([
            transforms.RandomRotation(90, fill=250),
            transforms.RandomResizedCrop(256, scale =(.9, 1.1)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(),
            transforms.ColorJitter(hue=.02, saturation= [.2, 1.6], brightness=[0.8, 1.2]),
            transforms.GaussianBlur(kernel_size= 3),
            transforms.Resize((224,224)),
            transforms.ToTensor()
        ])
    else:
        transform = transforms

    dataset = LabelDataset(data, mask_as_int, transform=transform)
    if remove_zeros:
        dataset.remove_zeros(0.95)
    # Compute the class weights
    samples_weights = get_weights(dataset)
    
    # Create the sampler                                                    

    sampler = WeightedRandomSampler(
        weights=samples_weights,
        num_samples=len(dataset),
        replacement=True)

    dataloader = DataLoader(
        dataset, 
        sampler=sampler,
        batch_size=batch_size,
        drop_last=True,
        num_workers=16)

    return dataloader

def forward_dataset(model, dataloader, training):
    running_loss = 0.0
    correct = 0
    total = 0
    # Compute all the steps
    for _, (data, target) in enumerate(dataloader):
        data, target = data.to(device), target.long().to(device)
        total += target.size(0)
        if training:
            optimizer.zero_grad()
        outputs = model(data)
        loss = criterion(outputs, target.long())
        if training:
            loss.backward()
            optimizer.step()
        running_loss += loss.item()
        _,pred = torch.max(outputs, dim=1)
        correct += torch.sum(pred==target).item()
    if total == 0:
        print('ERROR total target is 0')
    accuracy = correct/total
    return accuracy, running_loss/len(dataloader)

train_files, test_files, val_files = train_test_val_split_list(list_data_files('./process_image_output_256/'))
print("Found files of size (train, test, val):", len(train_files), len(test_files), len(val_files)) 

mask_as_int = True
batch_size = 512 # Multiply by 4 as we run on 4 GPUs
train_dataloader = getOverSampler(train_files, mask_as_int, batch_size)
val_dataloader = getOverSampler(val_files, mask_as_int, batch_size, remove_zeros = False)
test_dataloader = DataLoader(LabelDataset(test_files, mask_as_int), batch_size=batch_size, shuffle=True, num_workers=16)

# Define the model
model = models.resnet34()
model.fc = nn.Linear(model.fc.in_features, 3)

# USE MULTI GPU
model = torch.nn.parallel.DataParallel(model)

# LOAD MODEL ON DEVICE
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
x = model.to(device)

# LOSS, OPTIMIZER
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)


# MODEL PERF
def accuracy(out, labels):
    _,pred = torch.max(out, dim=1)
    return torch.sum(pred==labels).item()

#TRAIN LOOP
n_epochs = 300
val_loss_min = np.Inf
val_loss = val_acc = train_loss = train_acc = early_stop = 0
early_stop_limit = 15
model.train()
for epoch in range(1, n_epochs+1):
    if early_stop > early_stop_limit:
        print(f'Early Stopping detected after {early_stop_limit} epochs without improvements on validation loss. Retrieving previous best model')
        #model.load_state_dict(torch.load('resnet_tcga.pt'))
        break
    print(f'Epoch {epoch}/{n_epochs}')
    # Train for an epoch
    model.train()
    train_acc, train_loss = forward_dataset(model, train_dataloader, True)
    print(f'train loss: {(train_loss):.4f}, train acc: {(train_acc):.4f}')
    
    # Validation step
    with torch.no_grad():
        model.eval()
        val_acc, val_loss = forward_dataset(model, val_dataloader, False)
        print(f'validation loss: {(val_loss):.4f}, validation acc: {(val_acc):.4f}')
        if val_loss < val_loss_min:
            early_stop = 0
            val_loss_min = val_loss
            torch.save(model.state_dict(), './output/resnet_34_256_tcga.pt')
            print('validation loss improved, saving model')
        else:
            early_stop += 1
    print()

