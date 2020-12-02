import numpy as np
import torch
from torchvision import datasets, transforms, models
from torch import nn
from torch.utils import data
import torch.optim as optim
import torch.nn.functional as F
import os
from tqdm import tqdm, trange
from efficientnet_pytorch.efficientnet.metrics import Accuracy, Average
from efficientnet_pytorch.efficientnet.models import efficientnet_orig

mels_data_root = './data/output/figures_split/mels'
mfcc_data_root = './data/output/figures_split/mfcc'
OUTPUT_DIR = './checkpoints/figures_weight_shuffle_scratch'
TRAIN_ACC_HISTORY = 'eff4_scratch_train_acc_history.txt'
TRAIN_LOSS_HISTORY = 'eff4_scratch_train_loss_history.txt'
VAL_ACC_HISTORY = 'eff4_scratch_val_acc_history.txt'
VAL_LOSS_HISTORY = 'eff4_scratch_val_loss_history.txt'
LEARNING_RATE = 1e-4
EPOCHS = 100
SPLIT=4


class ImageInput(data.Dataset):
    def __init__(self, datasets):
        self.datasets = datasets

    def __getitem__(self, index):
        x1 = self.datasets[index]
        return x1

    def __len__(self):
        return len(self.datasets)





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
        
        n_samples = [288, 1424, 2032, 2080, 2048, 1664, 864]
        normedWeights = [1 - (x / sum(n_samples)) for x in n_samples]
        self.normedWeights = torch.FloatTensor(normedWeights).to(device)

    def fit(self):
        epochs = trange(self.epoch, self.num_epochs + 1, desc='Epoch', ncols=0)
        
        for self.epoch in epochs:
            # self.scheduler.step()

            train_loss, train_acc = self.train()
            valid_loss, valid_acc = self.evaluate()
            with open(VAL_ACC_HISTORY, 'a') as fva:
                fva.write(str(valid_acc.value)+"\n")
            with open(VAL_LOSS_HISTORY, 'a') as fvl:
                fvl.write(str(valid_loss.value)+"\n")
            with open(TRAIN_ACC_HISTORY, 'a') as fta:
                fta.write(str(train_acc.value)+"\n")
            with open(TRAIN_LOSS_HISTORY, 'a') as ftl:
                ftl.write(str(train_loss.value)+"\n")
            

            self.save_checkpoint(os.path.join(self.output_dir, 'eff4_scratch_epoch{}_validloss{}_checkpoint.pth'.format(self.epoch, valid_loss)))
            if valid_acc > self.best_acc:
                self.best_acc = valid_acc.value
                self.save_checkpoint(os.path.join(self.output_dir, 'best.pth'))

            epochs.set_postfix_str(f'train loss: {train_loss}, train acc: {train_acc}, '
                                f'valid loss: {valid_loss}, valid acc: {valid_acc}, '
                                f'best valid acc: {self.best_acc:.2f}')

    def train(self):
        self.model.train()
        self.model.to(self.device)
        train_loss = Average()
        train_acc = Accuracy()

        train_loader = tqdm(self.train_loader, ncols=0, desc='Train')
        for mels in train_loader:
            x1 = mels[0]
            y1 = mels[1]
          
            x1 = x1.to(self.device)
            y = y1.to(self.device)

            output = self.model(x1)
            loss = F.cross_entropy(output, y, weight=self.normedWeights)

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            train_loss.update(loss.item(), number=x1.size(0))
            train_acc.update(output, y)

            train_loader.set_postfix_str(f'train loss: {train_loss}, train acc: {train_acc}.')

        return train_loss, train_acc

    def evaluate(self):
        self.model.eval()
        self.model.to(self.device)
        valid_loss = Average()
        valid_acc = Accuracy()

        with torch.no_grad():
            valid_loader = tqdm(self.valid_loader, desc='Validate', ncols=0)
            for mels in valid_loader:
                x1 = mels[0]
                y1 = mels[1]
             
                x1 = x1.to(self.device)
                y = y1.to(self.device)

                output = self.model(x1)
                loss = F.cross_entropy(output, y, weight=self.normedWeights)
                valid_loss.update(loss.item(), number=x1.size(0))
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
        transforms.Resize((75, 128)),
        transforms.ToTensor(),
        normalize
    ])

    train_loader = data.DataLoader(
        ImageInput(datasets.ImageFolder(os.path.join(mels_data_root, "train"), transform=transform)),
        shuffle=True, batch_size=batch_size)
  
    val_loader = data.DataLoader(
        ImageInput(datasets.ImageFolder(os.path.join(mels_data_root, "val"), transform=transform)),
        shuffle=True, batch_size=batch_size)

    test_loader = data.DataLoader(
        ImageInput(datasets.ImageFolder(os.path.join(mels_data_root, "test"), transform=transform)),
        shuffle=False, batch_size=batch_size)

    return train_loader, val_loader, test_loader

def main():
    train_loader, val_loader, test_loader = load_dataset(batch_size=64)
    model = efficientnet_orig.efficientnet_b4(pretrained=False, num_classes=7)
    optimizer_SGD = torch.optim.SGD(model.parameters(), lr=LEARNING_RATE)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print("Device: ", device)
    trainer = Trainer(model, optimizer_SGD, train_loader, val_loader, device=device, num_epochs=EPOCHS, output_dir=OUTPUT_DIR)
    trainer.fit()




if __name__ == '__main__':
    main()