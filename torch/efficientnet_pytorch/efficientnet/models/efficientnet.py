import math

import mlconfig
import torch
from torch import nn

from .utils import load_state_dict_from_url

model_urls = {
    'efficientnet_b0': 'https://www.dropbox.com/s/9wigibun8n260qm/efficientnet-b0-4cfa50.pth?dl=1',
    'efficientnet_b1': 'https://www.dropbox.com/s/6745ear79b1ltkh/efficientnet-b1-ef6aa7.pth?dl=1',
    'efficientnet_b2': 'https://www.dropbox.com/s/0dhtv1t5wkjg0iy/efficientnet-b2-7c98aa.pth?dl=1',
    'efficientnet_b3': 'https://www.dropbox.com/s/5uqok5gd33fom5p/efficientnet-b3-bdc7f4.pth?dl=1',
    'efficientnet_b4': 'https://www.dropbox.com/s/y2nqt750lixs8kc/efficientnet-b4-3e4967.pth?dl=1',
    'efficientnet_b4_A': 'https://www.dropbox.com/s/y2nqt750lixs8kc/efficientnet-b4-3e4967.pth?dl=1',
    'efficientnet_b4_B': 'https://www.dropbox.com/s/y2nqt750lixs8kc/efficientnet-b4-3e4967.pth?dl=1',
    'efficientnet_b5': 'https://www.dropbox.com/s/qxonlu3q02v9i47/efficientnet-b5-4c7978.pth?dl=1',
    'efficientnet_b6': None,
    'efficientnet_b7': None,
}

params = {
    'efficientnet_b0': (1.0, 1.0, 224, 0.2),
    'efficientnet_b1': (1.0, 1.1, 240, 0.2),
    'efficientnet_b2': (1.1, 1.2, 260, 0.3),
    'efficientnet_b3': (1.2, 1.4, 300, 0.3),
    'efficientnet_b4': (1.4, 1.8, 380, 0.4),
    'efficientnet_b5': (1.6, 2.2, 456, 0.4),
    'efficientnet_b6': (1.8, 2.6, 528, 0.5),
    'efficientnet_b7': (2.0, 3.1, 600, 0.5),
}

split_layer_number = {
    4: 17,
    5: 23,
    6: 31,
}


class Swish(nn.Module):

    def __init__(self, *args, **kwargs):
        super(Swish, self).__init__()

    def forward(self, x):
        return x * torch.sigmoid(x)


class ConvBNReLU(nn.Sequential):

    def __init__(self, in_planes, out_planes, kernel_size, stride=1, groups=1):
        padding = self._get_padding(kernel_size, stride)
        super(ConvBNReLU, self).__init__(
            nn.ZeroPad2d(padding),
            nn.Conv2d(in_planes, out_planes, kernel_size, stride, padding=0, groups=groups, bias=False),
            nn.BatchNorm2d(out_planes),
            Swish(),
        )

    def _get_padding(self, kernel_size, stride):
        p = max(kernel_size - stride, 0)
        return [p // 2, p - p // 2, p // 2, p - p // 2]


class SqueezeExcitation(nn.Module):

    def __init__(self, in_planes, reduced_dim):
        super(SqueezeExcitation, self).__init__()
        self.se = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_planes, reduced_dim, 1),
            Swish(),
            nn.Conv2d(reduced_dim, in_planes, 1),
            nn.Sigmoid(),
        )

    def forward(self, x):
        return x * self.se(x)


class MBConvBlock(nn.Module):

    def __init__(self,
                 in_planes,
                 out_planes,
                 expand_ratio,
                 kernel_size,
                 stride,
                 reduction_ratio=4,
                 drop_connect_rate=0.2):
        super(MBConvBlock, self).__init__()
        self.drop_connect_rate = drop_connect_rate
        self.use_residual = in_planes == out_planes and stride == 1
        assert stride in [1, 2]
        assert kernel_size in [3, 5]

        hidden_dim = in_planes * expand_ratio
        reduced_dim = max(1, int(in_planes / reduction_ratio))

        layers = []
        # pw
        if in_planes != hidden_dim:
            layers += [ConvBNReLU(in_planes, hidden_dim, 1)]

        layers += [
            # dw
            ConvBNReLU(hidden_dim, hidden_dim, kernel_size, stride=stride, groups=hidden_dim),
            # se
            SqueezeExcitation(hidden_dim, reduced_dim),
            # pw-linear
            nn.Conv2d(hidden_dim, out_planes, 1, bias=False),
            nn.BatchNorm2d(out_planes),
        ]

        self.conv = nn.Sequential(*layers)

    def _drop_connect(self, x):
        if not self.training:
            return x
        keep_prob = 1.0 - self.drop_connect_rate
        batch_size = x.size(0)
        random_tensor = keep_prob
        random_tensor += torch.rand(batch_size, 1, 1, 1, device=x.device)
        binary_tensor = random_tensor.floor()
        return x.div(keep_prob) * binary_tensor

    def forward(self, x):
        if self.use_residual:
            return x + self._drop_connect(self.conv(x))
        else:
            return self.conv(x)


def _make_divisible(value, divisor=8):
    new_value = max(divisor, int(value + divisor / 2) // divisor * divisor)
    if new_value < 0.9 * value:
        new_value += divisor
    return new_value


def _round_filters(filters, width_mult):
    if width_mult == 1.0:
        return filters
    return int(_make_divisible(filters * width_mult))


def _round_repeats(repeats, depth_mult):
    if depth_mult == 1.0:
        return repeats
    return int(math.ceil(depth_mult * repeats))


@mlconfig.register
class EfficientNet(nn.Module):

    def __init__(self, width_mult=1.0, depth_mult=1.0, dropout_rate=0.2, num_classes=1000, mode="", split=0, include_top=False):
        super(EfficientNet, self).__init__()
        self.include_top = include_top
        self.mode = mode

        orig_settings = [
            # t,  c, n, s, k
            [1, 16, 1, 1, 3],  # MBConv1_3x3, SE, 112 -> 112
            [6, 24, 2, 2, 3],  # MBConv6_3x3, SE, 112 ->  56
            [6, 40, 2, 2, 5],  # MBConv6_5x5, SE,  56 ->  28
            [6, 80, 3, 2, 3],  # MBConv6_3x3, SE,  28 ->  14
            [6, 112, 3, 1, 5],  # MBConv6_5x5, SE,  14 ->  14
            [6, 192, 4, 2, 5],  # MBConv6_5x5, SE,  14 ->   7
            [6, 320, 1, 1, 3]  # MBConv6_3x3, SE,   7 ->   7
        ]

        # yapf: disable
        if mode == "start":
            settings = orig_settings[:split]
        elif mode == 'end':
            settings = orig_settings[split:]
        else:
            settings = orig_settings



        # yapf: enable

        out_channels = _round_filters(32, width_mult)
        features = list()
        if mode == 'start':
            features = [ConvBNReLU(3, out_channels, 3, stride=2)]
            in_channels = out_channels
        else:
            in_channels = _round_filters(orig_settings[split-1][1], width_mult)


        for t, c, n, s, k in settings:
            out_channels = _round_filters(c, width_mult)
            repeats = _round_repeats(n, depth_mult)
            for i in range(repeats):
                stride = s if i == 0 else 1
                features += [MBConvBlock(in_channels, out_channels, expand_ratio=t, stride=stride, kernel_size=k)]
                in_channels = out_channels

        if mode == "end" or mode == "":
            self.last_channels = _round_filters(1280, width_mult)
            print("[INFO] Last channels: ", self.last_channels)
            features += [ConvBNReLU(in_channels, self.last_channels, 1)]

        self.features = nn.Sequential(*features)
        if self.include_top:
            self.classifier = nn.Sequential(
                nn.Dropout(dropout_rate),
                nn.Linear(self.last_channels, num_classes),
            )

        # weight initialization
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                fan_out = m.weight.size(0)
                init_range = 1.0 / math.sqrt(fan_out)
                nn.init.uniform_(m.weight, -init_range, init_range)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x):
        x = self.features(x)
        if self.mode == "start":
            return x
        x = x.mean([2, 3])
        if self.include_top:
            x = self.classifier(x)
        return x


def _efficientnet(arch, pretrained, progress, mode="", split=0, num_classes = 7, freeze_A = False, **kwargs):
    width_mult, depth_mult, _, dropout_rate = params[arch]
    # model = EfficientNet(width_mult, depth_mult, dropout_rate)
    # print(model)
    if mode != "":
        model_start_A = EfficientNet(width_mult, depth_mult, dropout_rate, mode="start", split=split)
        model_start_B = EfficientNet(width_mult, depth_mult, dropout_rate, mode="start", split=split)
        model_end = EfficientNet(width_mult, depth_mult, dropout_rate, mode="end", split=split, include_top=True, num_classes=7)

        if pretrained:
            print('[INFO] pretrained network used')
            model_state = model_start_A.state_dict().keys()
            # print("model_start_state state: ", model_state)

            # print("\n\n")
            state_dict = load_state_dict_from_url(model_urls[arch], progress=progress)
            start_state_dict = dict()
            end_state_dict = dict()
            end_state_dict_corrected = dict()
            # print("weight state: ", state_dict.keys())
            for key in list(state_dict.keys()):
                if key not in model_state:
                    end_state_dict[key] = state_dict[key]
                else:
                    start_state_dict[key] = state_dict[key]

            if 'num_classes' in kwargs and kwargs['num_classes'] != 1000:
                del state_dict['classifier.1.weight']
                del state_dict['classifier.1.bias']

            model_start_A.load_state_dict(start_state_dict)
            model_start_B.load_state_dict(start_state_dict)
            # print("start_state_dict: ", start_state_dict.keys())
            if freeze_A:
                print("[INFO] FREEZING INPUT BLOCk")
                for param in model_start_A.parameters():
                    param.requires_grad = False
                for param in model_start_B.parameters():
                    param.requires_grad = False

            model_end_state = model_end.state_dict().keys()
            # print("model_end_state: ", model_end_state)

            for key in list(end_state_dict.keys()):
                if key.startswith('features'):
                    split_number_start = split_layer_number[split]
                    key_split = key.split(".")
                    key_number = key_split[1]
                    new_key_number = int(key_number) - split_number_start
                    key_split[1] = str(new_key_number)
                    join_new_key = ".".join(key_split)
                    # print("join_new_key : ", join_new_key)
                    end_state_dict_corrected[join_new_key] = end_state_dict[key]

            for key in list(end_state_dict_corrected.keys()):
                if key not in model_end_state or not key.startswith('features'):
                    print(end_state_dict_corrected[key])

            model_end.load_state_dict(end_state_dict_corrected, strict=False)
        else:
            print('[INFO] scratch network used')

        return model_start_A, model_start_B, model_end
            
    else:
        print("[INFO] Single input eff used")
        width_mult, depth_mult, _, dropout_rate = params[arch]

        model = EfficientNet(width_mult, depth_mult, dropout_rate, **kwargs)
        if pretrained:
            state_dict = load_state_dict_from_url(model_urls[arch], progress=progress)
            if num_classes != 1000:
                del state_dict['classifier.1.weight']
                del state_dict['classifier.1.bias']

        model.load_state_dict(state_dict, strict=False)
        return model





@mlconfig.register
def efficientnet_b0(pretrained=False, progress=True, **kwargs):
    return _efficientnet('efficientnet_b0', pretrained, progress, **kwargs)


@mlconfig.register
def efficientnet_b1(pretrained=False, progress=True, **kwargs):
    return _efficientnet('efficientnet_b1', pretrained, progress, **kwargs)


@mlconfig.register
def efficientnet_b2(pretrained=False, progress=True, **kwargs):
    return _efficientnet('efficientnet_b2', pretrained, progress, **kwargs)


@mlconfig.register
def efficientnet_b3(pretrained=False, progress=True, **kwargs):
    return _efficientnet('efficientnet_b3', pretrained, progress, **kwargs)


@mlconfig.register
def efficientnet_b4(pretrained=False, progress=True, num_classes=7, freeze_A= False, **kwargs):
    return _efficientnet('efficientnet_b4', pretrained, progress, num_classes=num_classes, freeze_A = freeze_A, **kwargs)

@mlconfig.register
def efficientnet_b4_split(pretrained=True, progress=True, split=4, num_classes=7, freeze_A= False, **kwargs):
    model_start_A, model_start_B, model_end = _efficientnet("efficientnet_b4", pretrained, progress, mode="start", split=split, num_classes=num_classes, freeze_A = freeze_A, **kwargs)
    # model_B = _efficientnet("efficientnet_b4", pretrained, progress, mode="end", split=split, **kwargs)
    return model_start_A, model_start_B, model_end

@mlconfig.register
def efficientnet_b5(pretrained=False, progress=True, **kwargs):
    return _efficientnet('efficientnet_b5', pretrained, progress, **kwargs)


@mlconfig.register
def efficientnet_b6(pretrained=False, progress=True, **kwargs):
    return _efficientnet('efficientnet_b6', pretrained, progress, **kwargs)


@mlconfig.register
def efficientnet_b7(pretrained=False, progress=True, **kwargs):
    return _efficientnet('efficientnet_b7', pretrained, progress, **kwargs)
