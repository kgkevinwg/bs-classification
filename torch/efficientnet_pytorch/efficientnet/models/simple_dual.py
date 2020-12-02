import torch
import torch.nn as nn


class DualInputNetwork(nn.Module):
    def build_block(self, in_channels, out_channels, kernel_size, stride):
        return nn.Sequential(
                nn.BatchNorm2d(num_features = in_channels),
                nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, stride=stride),
                nn.MaxPool2d(kernel_size=(2, 2))
                )
            
    def build_superblock(self):
        block_1 = self.build_block(3, 32, (5,5), (1,2))
        block_2 = self.build_block(32, 32, (5, 5), (1, 1))
        block_3 = self.build_block(32, 64, (5, 5), (1, 1))
        block_4 = self.build_block(64, 128, (3, 3), (1, 1))
        return nn.Sequential(
            block_1,
            block_2,
            block_3,
            block_4,
        )
    
    def __init__(self, n_classes):
        super(DualInputNetwork, self).__init__()
        self.mels_block = self.build_superblock()
        self.mfcc_block = self.build_superblock()
        self.fc_block = nn.Sequential(
            nn.BatchNorm2d(num_features= 256),
            nn.Flatten(start_dim = 1),
            nn.Dropout(.5),
            nn.Linear(in_features= 43264, out_features=5408),
            nn.Dropout(.5),
            nn.Linear(in_features=5408, out_features= n_classes)
        )
        
    def forward(self, mels, mfcc):
        mels = self.mels_block(mels)
        mfcc = self.mfcc_block(mfcc)
        concatenated = torch.cat((mels, mfcc), dim=1)
        x = self.fc_block(concatenated)
        return x
        
if __name__ == '__main__':
    DualInputNetwork(n_classes=7)