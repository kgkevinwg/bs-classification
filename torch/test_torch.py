import torch
from model_fc_torch import EnsembleModel
import numpy as np
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
from torchvision import transforms
from torchvision import datasets, transforms
from torch.utils import data
import torch.nn.functional as F
import os
from tqdm import tqdm

mels_data_root = './data/output/figures_split_old/mels'
mfcc_data_root = './data/output/figures_split_old/mfcc'
CHECKPOINT_DIR = './checkpoints/best.pth'
SPLIT=4


class MultiImageInput(data.Dataset):
    def __init__(self, *datasets):
        self.datasets = datasets

    def __getitem__(self, item):
        return tuple(d[item] for d in self.datasets)

    def __len__(self):
        return min(len(d) for d in self.datasets)

def load_dataset(batch_size=4):
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    transform = transforms.Compose([
        transforms.Resize(75, 128),
        transforms.ToTensor(),
        normalize
    ])


    test_loader = data.DataLoader(
        MultiImageInput(datasets.ImageFolder(os.path.join(mels_data_root, "test"), transform=transform),
                        datasets.ImageFolder(os.path.join(mfcc_data_root, "test"), transform=transform)),
        shuffle=False, batch_size=batch_size)

    return test_loader

def generate_labels(loader):
    loader = tqdm(loader, desc='Generate Labels', ncols=0)
    label_list = list()
    for mels, mfcc in loader:
        y1 = mels[1]
        y1 = y1.numpy().squeeze().item()
        label_list.append(y1)
    return label_list

def find_classes(path):
    classes = os.listdir(path)
    classes.sort()
    class_to_idx = {classes[i]: i for i in range(len(classes))}
    return classes, class_to_idx


if __name__ == "__main__":
    test_loader = load_dataset(batch_size=1)
    y_true = generate_labels(test_loader)
    ensembleModel = EnsembleModel(split=SPLIT, num_classes=7)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    checkpoint = torch.load(CHECKPOINT_DIR)
    ensembleModel.load_state_dict(checkpoint['model'])
    ensembleModel.eval()
    y_pred = list()
    with torch.no_grad():
        loader = tqdm(test_loader, desc='Test', ncols=0)
        label_list = list()
        ensembleModel.to(device)
        for mels, mfcc in loader:
            x1 = mels[0]
            x2 = mfcc[0]
            y1 = mels[1]
            y2 = mfcc[1]

            x1 = x1.to(device)
            x2 = x2.to(device)
            y = y1.to(device)
            output = ensembleModel(x1, x2)
            
            y_pred.append(output.cpu().numpy().squeeze())

    print(np.array(y_true).shape)
    y_pred = np.array(y_pred).argmax(axis=1).tolist()
    
    test_path = os.path.join(mels_data_root, "test")
    classes = find_classes(test_path)

    print(confusion_matrix(y_true, y_pred))

