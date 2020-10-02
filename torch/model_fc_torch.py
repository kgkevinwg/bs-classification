from efficientnet_pytorch.efficientnet.models import efficientnet
import torch
from torch import nn

class EnsembleModel(nn.Module):
    def __init__(self,split, num_classes=7):
        super(EnsembleModel, self).__init__()
        model_start_A, model_start_B, model_end = efficientnet.efficientnet_b4_split(split=split, num_classes=num_classes)
        self.model_start_A = model_start_A
        self.model_start_B = model_start_B
        self.model_end = model_end

    def forward(self, mels, mfcc):
        mels = self.model_start_A(mels)
        mfcc = self.model_start_B(mfcc)
        add = torch.add(mels, mfcc)
        x = self.model_end(add)
        return x




if __name__ == '__main__':
    ensembleModel = EnsembleModel(split=4, num_classes=7)
