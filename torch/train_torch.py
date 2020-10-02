import numpy as np
import torch
from torchvision import datasets, transforms
from torch import nn
from torch.utils import data
import torch.optim as optim
import torch.nn.functional as F
import os
from tqdm import tqdm, trange
from efficientnet_pytorch.efficientnet.metrics import Accuracy, Average
from model_fc_torch import EnsembleModel

mels_data_root = './data/output/figures_split/mels'
mfcc_data_root = './data/output/figures_split/mfcc'
OUTPUT_DIR = './checkpoints/torch'
LEARNING_RATE = 1e-5
EPOCHS = 50
SPLIT=4


class MultiImageInput(data.Dataset):
    def __init__(self, *datasets):
        self.datasets = datasets

    def __getitem__(self, item):
        return tuple(d[item] for d in self.datasets)

    def __len__(self):
        return min(len(d) for d in self.datasets)





class Trainer():
    def __init__(self, model: nn.Module, optimizer: optim.Optimizer, train_loader: data.DataLoader,
                 valid_loader: data.DataLoader, device: torch.device,
                 num_epochs: int, output_dir: str):
        self.model = model
        self.optimizer = optimizer
        # self.scheduler = scheduler
        self.train_loader = train_loader
        self.valid_loader = valid_loader
        self.device = device
        self.num_epochs = num_epochs
        self.output_dir = output_dir

        self.epoch = 1
        self.best_acc = 0

    def fit(self):
        epochs = trange(self.epoch, self.num_epochs + 1, desc='Epoch', ncols=0)
        for self.epoch in epochs:
            # self.scheduler.step()

            train_loss, train_acc = self.train()
            valid_loss, valid_acc = self.evaluate()

            self.save_checkpoint(os.path.join(self.output_dir, 'eff4b_split{}_epoch{}_validloss{}_checkpoint.pth'.format(SPLIT, self.epoch, valid_loss)))
            if valid_acc > self.best_acc:
                self.best_acc = valid_acc.value
                self.save_checkpoint(os.path.join(self.output_dir, 'best.pth'))

            epochs.set_postfix_str(f'train loss: {train_loss}, train acc: {train_acc}, '
                                   f'valid loss: {valid_loss}, valid acc: {valid_acc}, '
                                   f'best valid acc: {self.best_acc:.2f}')

    def train(self):
        self.model.train()

        train_loss = Average()
        train_acc = Accuracy()

        train_loader = tqdm(self.train_loader, ncols=0, desc='Train')
        for mels, mfcc in train_loader:
            x1 = mels[0]
            x2 = mfcc[0]
            y1 = mels[1]
            y2 = mfcc[1]
            if not torch.all(torch.eq(y1, y2)).numpy():
                print("y1 y2 not valid")
                exit()
            x1 = x1.to(self.device)
            x2 = x2.to(self.device)
            y = y1.to(self.device)

            output = self.model(x1, x2)
            loss = F.cross_entropy(output, y)

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            train_loss.update(loss.item(), number=x1.size(0))
            train_acc.update(output, y)

            train_loader.set_postfix_str(f'train loss: {train_loss}, train acc: {train_acc}.')

        return train_loss, train_acc

    def evaluate(self):
        self.model.eval()

        valid_loss = Average()
        valid_acc = Accuracy()

        with torch.no_grad():
            valid_loader = tqdm(self.valid_loader, desc='Validate', ncols=0)
            for x, y in valid_loader:
                x = x.to(self.device)
                y = y.to(self.device)

                output = self.model(x)
                loss = F.cross_entropy(output, y)

                valid_loss.update(loss.item(), number=x.size(0))
                valid_acc.update(output, y)

                valid_loader.set_postfix_str(f'valid loss: {valid_loss}, valid acc: {valid_acc}.')

        return valid_loss, valid_acc

    def save_checkpoint(self, f):
        self.model.eval()

        checkpoint = {
            'model': self.model.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            # 'scheduler': self.scheduler.state_dict(),
            'epoch': self.epoch,
            'best_acc': self.best_acc
        }

        dirname = os.path.dirname(f)
        if dirname:
            os.makedirs(dirname, exist_ok=True)

        torch.save(checkpoint, f)

    def resume(self, f):
        checkpoint = torch.load(f, map_location=self.device)

        self.model.load_state_dict(checkpoint['model'])
        self.optimizer.load_state_dict(checkpoint['optimizer'])
        # self.scheduler.load_state_dict(checkpoint['scheduler'])

        self.epoch = checkpoint['epoch'] + 1
        self.best_acc = checkpoint['best_acc']

def load_dataset(batch_size=4):
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    transform = transforms.Compose([
        transforms.Resize(75, 128),
        transforms.ToTensor(),
        normalize
    ])

    train_loader = data.DataLoader(
        MultiImageInput(datasets.ImageFolder(os.path.join(mels_data_root, "train"), transform=transform),
                        datasets.ImageFolder(os.path.join(mfcc_data_root, "train"), transform=transform)),
        shuffle=False, batch_size=batch_size)
    val_loader = data.DataLoader(
        MultiImageInput(datasets.ImageFolder(os.path.join(mels_data_root, "val"), transform=transform),
                        datasets.ImageFolder(os.path.join(mfcc_data_root, "val"), transform=transform)),
        shuffle=False, batch_size=batch_size)
    test_loader = data.DataLoader(
        MultiImageInput(datasets.ImageFolder(os.path.join(mels_data_root, "test"), transform=transform),
                        datasets.ImageFolder(os.path.join(mfcc_data_root, "test"), transform=transform)),
        shuffle=False, batch_size=batch_size)

    return train_loader, val_loader, test_loader

def main():
    train_loader, val_loader, test_loader = load_dataset(batch_size=5)
    ensembleModel = EnsembleModel(split=SPLIT, num_classes=7)
    optimizer_SGD = torch.optim.SGD(ensembleModel.parameters(), lr=LEARNING_RATE)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print("Device: ", device)
    trainer = Trainer(ensembleModel, optimizer_SGD, train_loader, val_loader, device=device, num_epochs=EPOCHS, output_dir=OUTPUT_DIR)
    trainer.fit()




if __name__ == '__main__':
    main()