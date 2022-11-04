import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.modules.batchnorm import _BatchNorm


class Identity(nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__()
    def forward(self, x):
        return x
    
class _CN(_BatchNorm):
    #def __init__(self, num_features, eps=1e-5, momentum=0.1, affine=False):
    def __init__(self, target, eps = 1e-5, momentum = 0.1, affine=True):
        num_features = target.num_features
        super(_CN, self).__init__(num_features, eps, momentum, affine=True)
        self.running_mean = target.running_mean
        self.running_var = target.running_var
        
        self.weight = target.weight
        self.bias = target.bias

        self.N = num_features
        self.setG()
        #self.avg = nn.AdaptiveAvgPool2d((1,1))

    def setG(self):
        pass
    def _check_input_dim(self, input):
        pass

    def forward(self, input):
        #self._check_input_dim(input)
        out_gn = F.group_norm(input, self.G, None, None, self.eps)
        out = F.batch_norm(out_gn, self.running_mean, self.running_var, self.weight, self.bias,
                self.training, self.momentum, self.eps)
        return out

class CN(_CN):
    def __init__(self, num_features, eps = 1e-5, momentum = 0.1, G=8, affine=True):
        super(_CN, self).__init__(num_features, eps, momentum, affine=True)
        self.G = G

    def forward(self, input):
        out_gn = F.group_norm(input, self.G, None, None, self.eps)
        out = F.batch_norm(out_gn, self.running_mean, self.running_var, self.weight, self.bias,
                self.training, self.momentum, self.eps)
        return out
    
class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, padding, stride=1, norm='none'):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, in_channels, kernel_size, padding=padding, stride=stride)
        self.norm1 = get_norm(norm)(in_channels)
        self.conv2 = nn.Conv2d(in_channels, out_channels, kernel_size, padding=1)
        self.norm2 = get_norm(norm)(out_channels)
        self.proj = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride) if in_channels != out_channels else Identity()
    def forward(self, x):
        z = F.leaky_relu(self.norm1(self.conv1(x)))
        z = self.norm2(self.conv2(z))
        z += self.proj(x) 
        return F.leaky_relu(z)               

class BottleNeckBlock(nn.Module):
    def __init__(self, in_channels, mid_channels, out_channels, kernel_size, padding, stride=1, norm='none'):
        super(BottleNeckBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, mid_channels, kernel_size=1, stride=stride)
        self.norm1 = get_norm(norm)(mid_channels)
        self.conv2 = nn.Conv2d(mid_channels, mid_channels, kernel_size=kernel_size, padding=padding)
        self.norm2 = get_norm(norm)(mid_channels)
        self.conv3 = nn.Conv2d(mid_channels, out_channels, kernel_size=1)
        self.norm3 = get_norm(norm)(out_channels)
        self.proj = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride) if in_channels != out_channels else Identity()
    def forward(self, x):
        z = F.leaky_relu(self.norm1(self.conv1(x)))
        z = F.leaky_relu(self.norm2(self.conv2(z)))
        z = self.conv3(z)
        z += self.proj(x)
        return F.leaky_relu(self.norm3(z))

class ResNet(nn.Module):
    def __init__(self, args, bottleneck_ratio=4):
        super(ResNet, self).__init__()
        model_candidate = ['resnet_18', 'resnet_34', 'resnet_50', 'resnet_101', 'resnet_152']
        assert args.model in model_candidate, "Wrong Model Configuration"
        # Apply GAP right after last block
        self.block_cfg = {'resnet_18':[64]*2 + [128]*2 + [256]*2 + [512]*2, # output shape : 1x1x512
                          'resnet_34':[64]*3 + [128]*4 + [256]*6 + [512]*3, # output shape : 1x1x512
                          'resnet_50':[256]*3 + [512]*4 + [1024]*6 + [2048]*3, # output shape : 1x1x2048
                          'resnet_101':[256]*3 + [512]*4 + [1024]*23 + [2048]*3, # output shape : 1x1x2048
                          'resnet_152':[256]*3 + [512]*8 + [1024]*36 + [2048]*3} # output shape : 1x1x2048
        self.extractor = self._get_network(args, self.block_cfg[args.model], bottleneck_ratio)
        self.avgpool = nn.AdaptiveMaxPool2d(1)
        self.num_classes = args.num_classes
        self.embedding = nn.Embedding(args.num_classes, 512)
        # self.fc1 = nn.Linear(self.block_cfg[args.model][-1], 512)
        # self.fc2 = nn.Linear(512, 128)
        self.fc3 = nn.Linear(512, 1)
    def _get_network(self, args, block_cfg, bottleneck_ratio):
        prev_channel = 64
        networks = list()
        networks.append(nn.Conv2d(3, 64, 3, 1, 1))
        for channel in block_cfg:
            stride = 2 if prev_channel != channel and prev_channel != 3 else 1
            if args.model == 'resnet_50' or args.model == 'resnet_101' or args.model == 'resnet_152':
                networks.append(BottleNeckBlock(prev_channel, channel // bottleneck_ratio, channel, 3, padding=1, stride=stride, norm=args.norm))
            else:
                networks.append(ResidualBlock(prev_channel, channel, 3, padding=1, stride=stride, norm=args.norm))
            prev_channel = channel
        return nn.Sequential(*networks)
    def forward(self, x, y):
        bs = x.size(0)
        z = self.extractor(x)
        z = self.avgpool(z)
        z = z.view(bs, -1)
        rep = z
        y_z = self.embedding(y)
        y_z = F.softmax(y_z, dim=-1) * y_z.shape[-1]
        # z = self.fc1(z)
        z = z[:,None,:].expand_as(y_z)
        z = z * y_z
        
        # z = self.fc2(z)
        return self.fc3(z).view(bs, -1), rep

class EBM_Beginning(nn.Module):
    def __init__(self, args):
        super(EBM_Beginning, self).__init__()
        
        self.num_class = args.num_classes
        self.embedding = nn.Embedding(self.num_class, 3)
        self.conv1 = nn.Conv2d(3, 32, kernel_size = 3, padding = 1, bias = True)
        self.conv2 = nn.Conv2d(32, 32, kernel_size = 3, padding = 0, bias = True)
        self.conv3 = nn.Conv2d(32, 64, kernel_size = 3, padding = 1, bias = True)
        self.conv4 = nn.Conv2d(64, 64, kernel_size = 3, padding = 0, bias = True)

        self.maxpool1 = nn.MaxPool2d(2, stride = 2)
        self.maxpool2 = nn.MaxPool2d(2, stride = 2)
        
        self.fc1 = nn.Linear(1600, 1024)
        self.fc2 = nn.Linear(1024, 1)
        
    def forward(self, x, y):
        bs = x.shape[0]
        y = self.embedding(y)
        y = F.softmax(y, dim = -1)
        y = y*y.shape[-1]
        y = y.view(bs,3,1,1)
        y = y.expand_as(x)
        
        x = x*y
        
        x = self.conv1(x)
        x = F.relu(x)
        x = self.conv2(x)
        x = F.relu(x)
        x = self.maxpool1(x)
        x = self.conv3(x)
        x = F.relu(x)
        x = self.conv4(x)
        x = F.relu(x)
        x = self.maxpool2(x)
        
        x = x.view(bs, -1)
        
        x = self.fc1(x)
        x = F.relu(x)
        x = self.fc2(x)
        
        x = x.view(bs, -1)
        
        return x
        
class EBM_Middle(nn.Module):
    def __init__(self, args):
        super(EBM_Middle, self).__init__()
                
        self.num_class = args.num_classes
        self.embedding = nn.Embedding(self.num_class, 32)
        self.conv1 = nn.Conv2d(3, 32, kernel_size = 3, padding = 1, bias = True)
        self.conv2 = nn.Conv2d(32, 32, kernel_size = 3, padding = 0, bias = True)
        self.conv3 = nn.Conv2d(32, 64, kernel_size = 3, padding=1, bias = True)
        self.conv4 = nn.Conv2d(64, 64, kernel_size = 3, padding = 0, bias = True)

        self.maxpool1 = nn.MaxPool2d(2, stride = 2)
        self.maxpool2 = nn.MaxPool2d(2, stride = 2)
        
        self.fc1 = nn.Linear(2304, 1024)
        self.fc2 = nn.Linear(1024, 1)
        
    def forward(self, x, y):
        bs = x.shape[0]
        x = self.conv1(x)
        x = F.relu(x)
        x = self.conv2(x)
        x = F.relu(x)
        
        y = self.embedding(y)
        y = F.softmax(y, dim = -1)
        y = y*y.shape[-1]
        y = y.view(bs,32,1,1)
        y = y.expand_as(x)
        
        x = x*y
        
        x = self.maxpool1(x)
        x = self.conv3(x)
        x = F.relu(x)
        x = self.conv4(x)
        x = F.relu(x)
        x = self.maxpool2(x)
        
        x = x.view(bs, -1)
        
        x = self.fc1(x)
        x = F.relu(x)
        x = self.fc2(x)
                
        return x
    
class EBM_End(nn.Module):
    def __init__(self, args):
        super(EBM_End, self).__init__()
        
        self.num_class = args.num_classes
        self.embedding = nn.Embedding(self.num_class, 1024)
        self.conv1 = nn.Conv2d(3, 32, kernel_size = 3, padding = 1, bias = True)
        self.conv2 = nn.Conv2d(32, 32, kernel_size = 3, padding = 0, bias = True)
        self.conv3 = nn.Conv2d(32, 64, kernel_size = 3, padding=1, bias = True)
        self.conv4 = nn.Conv2d(64, 64, kernel_size = 3, padding = 0, bias = True)

        self.maxpool1 = nn.MaxPool2d(2, stride = 2)
        self.maxpool2 = nn.MaxPool2d(2, stride = 2)
        
        self.fc1 = nn.Linear(1600, 1024)
        self.fc2 = nn.Linear(1024, 1)
        
    def forward(self, x, y):
        bs = x.shape[0]
        x = self.conv1(x) # |32x28x28|
        x = F.relu(x)
        x = self.conv2(x) # |32x26x26|
        x = F.relu(x)
        x = self.maxpool1(x) # |32x13x13|
        x = self.conv3(x) # |64x13x13|
        x = F.relu(x)
        x = self.conv4(x) # |64x11x11|
        x = F.relu(x)
        x = self.maxpool2(x) # |64x5x5|
        
        x = x.view(bs, -1)
        x = self.fc1(x)
        
        y = self.embedding(y)
        y = F.softmax(y, dim = -1)
        y = y*y.shape[-1]
        
        x = x*y
        
        x = x.view(bs, -1)
        
        x = self.fc2(x)
        
        return x

class EBM_End_cifar(nn.Module):
    def __init__(self, args):
        super(EBM_End_cifar, self).__init__()
        
        self.num_class = args.num_classes
        self.embedding = nn.Embedding(self.num_class, 1024)
        self.conv1 = ResidualBlock(3, 32, kernel_size = 3, padding = 1) 
        self.conv2 = ResidualBlock(32, 32, kernel_size = 3, padding = 1) 
        self.conv3 = ResidualBlock(32, 64, kernel_size = 3, padding = 1) 
        self.conv4 = ResidualBlock(64, 64, kernel_size = 3, padding = 1) 
        self.conv5 = ResidualBlock(64, 128, kernel_size = 3, padding = 1, residual=False)
        self.conv6 = ResidualBlock(128, 128, kernel_size = 3, padding = 1)
        self.conv7 = ResidualBlock(128, 256, kernel_size = 3, padding = 1)
        self.conv8 = ResidualBlock(256, 256, kernel_size = 3, padding = 1, stride = 2, residual=False)
        self.conv9 = ResidualBlock(256, 512, kernel_size = 3, padding = 1, stride = 2, residual=False)

        self.maxpool1 = nn.MaxPool2d(2, stride = 2)
        self.maxpool2 = nn.MaxPool2d(2, stride = 2)
        self.maxpool3 = nn.MaxPool2d(2, stride = 2)
        self.maxpool4 = nn.MaxPool2d(2, stride = 2)
        self.fc1 = nn.Linear(512, 1024)
        self.fc2 = nn.Linear(1024, 1)
        
    def forward(self, x, y, rep_arr=None):
        bs = x.shape[0]
        x = self.conv1(x) # |32 x 128 x 128|
        x = F.relu(x)
        x = self.conv2(x) # |32 x 128 x 128|
        x = F.relu(x)
        x = self.maxpool1(x) # |32 x 64 x 64|
        x = self.conv3(x) # |64 x 64 x 64|
        x = F.relu(x)
        x = self.conv4(x) # |64 x 64 x 64|
        x = F.relu(x)
        x = self.maxpool2(x) # |64 x 32 x 32|
        x = self.conv5(x) # |128 x 32 x 32|
        x = F.relu(x)
        x = self.conv6(x) # |128 x 32 x 32|
        x = F.relu(x)
        x = self.maxpool3(x) # |128 x 16 x 16|
        x = self.conv7(x) # |256 x 16 x 16|
        x = F.relu(x)
        x = self.conv8(x) # |256 x 8 x 8|
        x = F.relu(x)
        x = self.conv9(x)
        x = F.relu(x) # |512 x 4 x 4|
        
        x = x.view(bs, -1)
        x = self.fc1(x)
        
        y = self.embedding(y)
        y = F.softmax(y, dim = -1)
        y = y*y.shape[-1]
        
        x = x*y
        
        x = x.view(bs, -1) # |b x 1024|
        rep = x
        x = self.fc2(x)
        
        return x, rep

class SBC_cifar(nn.Module):
    def __init__(self, args):
        super(SBC_cifar, self).__init__()
        
        self.num_class = args.num_classes
        self.conv1 = ResidualBlock(3, 32, kernel_size = 3, padding = 1) 
        self.conv2 = ResidualBlock(32, 32, kernel_size = 3, padding = 1) 
        self.conv3 = ResidualBlock(32, 64, kernel_size = 3, padding = 1) 
        self.conv4 = ResidualBlock(64, 64, kernel_size = 3, padding = 1) 
        self.conv5 = ResidualBlock(64, 128, kernel_size = 3, padding = 1, residual=False)
        self.conv6 = ResidualBlock(128, 128, kernel_size = 3, padding = 1)
        self.conv7 = ResidualBlock(128, 256, kernel_size = 3, padding = 1)
        self.conv8 = ResidualBlock(256, 310, kernel_size = 3, padding = 1, stride = 2, residual=False)
        self.conv9 = ResidualBlock(310, 512, kernel_size = 3, padding = 1, stride = 2, residual=False)

        self.maxpool1 = nn.MaxPool2d(2, stride = 2)
        self.maxpool2 = nn.MaxPool2d(2, stride = 2)
        self.maxpool3 = nn.MaxPool2d(2, stride = 2)
        self.maxpool4 = nn.MaxPool2d(2, stride = 2)
        self.fc1 = nn.Linear(512, 256)
        self.fc2 = nn.Linear(256, 10)
        
    def forward(self, x, rep_arr=None):
        bs = x.shape[0]
        x = self.conv1(x) # |32 x 128 x 128|
        x = F.relu(x)
        x = self.conv2(x) # |32 x 128 x 128|
        x = F.relu(x)
        x = self.maxpool1(x) # |32 x 64 x 64|
        x = self.conv3(x) # |64 x 64 x 64|
        x = F.relu(x)
        x = self.conv4(x) # |64 x 64 x 64|
        x = F.relu(x)
        x = self.maxpool2(x) # |64 x 32 x 32|
        x = self.conv5(x) # |128 x 32 x 32|
        x = F.relu(x)
        x = self.conv6(x) # |128 x 32 x 32|
        x = F.relu(x)
        x = self.maxpool3(x) # |128 x 16 x 16|
        x = self.conv7(x) # |256 x 16 x 16|
        x = F.relu(x)
        x = self.conv8(x) # |256 x 8 x 8|
        x = F.relu(x)
        x = self.conv9(x)
        x = F.relu(x) # |512 x 4 x 4|
        
        x = x.view(bs, -1)
        x = self.fc1(x)
        
        x = x.view(bs, -1) # |b x 1024|
        rep = x
        x = self.fc2(x)
        
        return x, rep

class EBM_End_Oxford(nn.Module):
    def __init__(self, args):
        super(EBM_End_Oxford, self).__init__()
        
        self.num_class = args.num_classes
        self.embedding = nn.Embedding(self.num_class, 1024)
        self.conv1 = ResidualBlock(3, 32, kernel_size = 3, padding = 1) 
        self.conv2 = ResidualBlock(32, 32, kernel_size = 3, padding = 1) 
        self.conv3 = ResidualBlock(32, 64, kernel_size = 3, padding = 1) 
        self.conv4 = ResidualBlock(64, 64, kernel_size = 3, padding = 1) 
        self.conv5 = ResidualBlock(64, 128, kernel_size = 3, padding = 1, residual=False)
        self.conv6 = ResidualBlock(128, 128, kernel_size = 3, padding = 1)
        self.conv7 = ResidualBlock(128, 256, kernel_size = 3, padding = 1)
        self.conv8 = ResidualBlock(256, 256, kernel_size = 3, padding = 1, stride = 2, residual=False)
        self.conv9 = ResidualBlock(256, 512, kernel_size = 3, padding = 1, stride = 2, residual=False)

        self.maxpool1 = nn.MaxPool2d(2, stride = 2)
        self.maxpool2 = nn.MaxPool2d(2, stride = 2)
        self.maxpool3 = nn.MaxPool2d(2, stride = 2)
        self.maxpool4 = nn.MaxPool2d(2, stride = 2)
        self.fc1 = nn.Linear(2048, 1024)
        self.fc2 = nn.Linear(1024, 1)
        
    def forward(self, x, y, rep_arr=None):
        bs = x.shape[0]
        x = self.conv1(x) # |32 x 128 x 128|
        x = F.relu(x)
        x = self.conv2(x) # |32 x 128 x 128|
        x = F.relu(x)
        x = self.maxpool1(x) # |32 x 64 x 64|
        x = self.conv3(x) # |64 x 64 x 64|
        x = F.relu(x)
        x = self.conv4(x) # |64 x 64 x 64|
        x = F.relu(x)
        x = self.maxpool2(x) # |64 x 32 x 32|
        x = self.conv5(x) # |128 x 32 x 32|
        x = F.relu(x)
        x = self.conv6(x) # |128 x 32 x 32|
        x = F.relu(x)
        x = self.maxpool3(x) # |128 x 16 x 16|
        x = self.conv7(x) # |256 x 16 x 16|
        x = F.relu(x)
        x = self.conv8(x) # |256 x 8 x 8|
        x = F.relu(x)
        x = self.conv9(x)
        x = F.relu(x) # |512 x 4 x 4|
        x = self.maxpool4(x) # |512 x 2 x 2|
        
        x = x.view(bs, -1)
        x = self.fc1(x)
        
        y = self.embedding(y)
        y = F.softmax(y, dim = -1)
        y = y*y.shape[-1]
        
        x = x*y
        
        x = x.view(bs, -1) # |b x 1024|
        rep = x
        x = self.fc2(x)
        
        return x, rep

class EBM_MNIST(nn.Module):
    def __init__(self, args):
        super(EBM_MNIST, self).__init__()
        self.num_classes = args.num_classes
        self.FC_1 = nn.Linear(784, 1000)
        self.relu_1 = nn.ReLU(inplace=True)
        self.embedding = nn.Embedding(self.num_classes, 1000)
        self.relu_2 = nn.ReLU(inplace=True)
        self.out = nn.Linear(1000, 1)
        
    def forward(self, x, y):
        x = x.view(-1, 784)
        z = self.FC_1(x)
        z = self.relu_1(z)
        y = self.embedding(y)
        z = z * F.normalize(y, p=2, dim=-1) + z
        z = self.relu_2(z)
        return self.out(z)

class SBC_Splitted_MNIST(nn.Module):
    def __init__(self, args):
        super(SBC_Splitted_MNIST, self).__init__()
        self.num_classes = args.num_classes
        self.FC_1 = nn.Linear(784, 1000)
        self.relu_1 = nn.ReLU(inplace=True)
        self.FC_2 = nn.Linear(1000, 1000)
        self.relu_2 = nn.ReLU(inplace=True)
        self.out = nn.Linear(1000, 10)
    def forward(self, x):
        x = x.view(-1, 784)
        z = self.FC_1(x)
        z = self.relu_1(z)
        z = self.FC_2(z)
        z = self.relu_2(z)
        return F.softmax(self.out(z))

class SBC_Permuted_MNIST(nn.Module):
    def __init__(self, args):
        super(SBC_Permuted_MNIST, self).__init__()
        self.num_classes = args.num_classes
        self.FC_1 = nn.Linear(784, 1000)
        self.relu_1 = nn.ReLU(inplace=True)
        self.FC_2 = nn.Linear(1000, 1000)
        self.relu_2 = nn.ReLU(inplace=True)
        self.out = nn.Linear(1000, 100)
    def forward(self, x):
        x = x.view(-1, 784)
        z = self.FC_1(x)
        z = self.relu_1(z)
        z = self.FC_2(z)
        z = self.relu_2(z)
        return F.softmax(self.out(z))

def get_norm(norm):
    if norm == 'batchnorm':
        return nn.BatchNorm2d
    elif norm == 'continualnorm':
        return CN
    else:
        return Identity


def get_model(model_name):
    if model_name == 'beginning':
        return EBM_Beginning
    elif model_name == 'middle':
        return EBM_Middle
    elif model_name == 'end':
        return EBM_End
    elif model_name == 'ebm_mnist':
        return EBM_MNIST
    elif model_name == 'sbc_splitted_mnist':
        return SBC_Splitted_MNIST
    elif model_name == 'sbc_permuted_mnist':
        return SBC_Permuted_MNIST
    elif model_name=='end_oxford':
        return EBM_End_Oxford
    elif model_name == 'ebm_cifar':
        return EBM_End_cifar
    elif model_name == 'sbc_cifar':
        return SBC_cifar
    elif model_name.split('_')[0] == 'resnet':
        return ResNet
    else:
        raise 