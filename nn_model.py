import torch
from torch import nn
from torch.nn import functional as F
from torch.autograd import Variable
from functions import ReverseLayerF

class BasicBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1):
        super(BasicBlock, self).__init__()
        self.conv = nn.Conv1d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding, bias=False)
        self.bn = nn.BatchNorm1d(out_channels)
        self.pool = nn.MaxPool1d(2,stride=2)
        
    def forward(self, x, pool = True):
        out = self.conv(x)
        out = self.bn(out)
        out = self.pool(out)
        out = F.relu(out)
        return out

class encoder_1(nn.Module):
    def __init__(self, in_channels,  mode = 'training_mode'):
        super(encoder_1, self).__init__()
        self.extractor = nn.Sequential()
        self.extractor.add_module('bn_layer',nn.BatchNorm1d(in_channels))
        self.extractor.add_module('conv_1',BasicBlock(in_channels, 16, kernel_size=64, stride=16, padding=24))
        self.extractor.add_module('drop_out', nn.Dropout(0.5))
        self.extractor.add_module('conv_2', BasicBlock(16, 32))
        self.extractor.add_module('drop_out', nn.Dropout(0.4))        
        self.extractor.add_module('conv_3', BasicBlock(32, 64))
        self.extractor.add_module('drop_out', nn.Dropout(0.3))

    def forward(self, x):
        features = self.extractor(x)
        features = features.view(-1, 1024)

        return features
    
class encoder_2(nn.Module):
    def __init__(self, in_channels):
        super(encoder_2, self).__init__()
        self.extractor = nn.Sequential()
        self.extractor.add_module('bn_layer',nn.BatchNorm1d(in_channels))
        self.extractor.add_module('conv_1',BasicBlock(in_channels, 16, kernel_size=64, stride=16, padding=24))
        self.extractor.add_module('drop_out', nn.Dropout(0.5))
        self.extractor.add_module('conv_2', BasicBlock(16, 32))
        self.extractor.add_module('res_1', res_block(32, 32, 3))
        self.extractor.add_module('drop_out', nn.Dropout(0.4))
        self.extractor.add_module('conv_3', BasicBlock(32, 64))
        self.extractor.add_module('res_2', res_block(64, 64, 3))
        self.extractor.add_module('conv_4', BasicBlock(64, 64))
        self.extractor.add_module('res_3', res_block(64, 64, 3))
        self.extractor.add_module('flatten', nn.Flatten())
        # self.extractor.add_module('linear', nn.Linear(512, 128))
        # self.extractor.add_module('conv_4', BasicBlock(64, 64, padding = 'same'))
        # self.extractor.add_module('drop_out', nn.Dropout(0.3))
        
    def forward(self, x):
        features = self.extractor(x)
        # features = F.relu(features)
        return features


class class_classifer(nn.Module):
    def __init__(self, n_class,  mode = 'training_mode'):
        super(class_classifer, self).__init__()
        self.classifier = nn.Sequential()
        self.classifier.add_module('cls_2', nn.Linear(512, 4))
        # self.classifier.add_module('cls_bn_1', nn.BatchNorm1d(32))
        # self.classifier.add_module('cls_relu_2', nn.ReLU(True))
        # self.classifier.add_module('drop_out', nn.Dropout(0.2))
        # self.classifier.add_module('cls_3', nn.Linear(32, n_class))

    def forward(self, x):

        cls = self.classifier(x)

        return cls    

class domain_classifer(nn.Module):
    def __init__(self):
        super(domain_classifer, self).__init__()
        self.domain_classifier = nn.Sequential()
        self.domain_classifier.add_module('d_cls_1', nn.Linear(512, 32))
        self.domain_classifier.add_module('d_cls_bn_1', nn.BatchNorm1d(32))
        self.domain_classifier.add_module('d_cls_relu_2', nn.ReLU(True))
        self.domain_classifier.add_module('drop_out', nn.Dropout(0.2))
        self.domain_classifier.add_module('d_cls_2', nn.Linear(32, 2))
        
    def forward(self, x, alpha = 0):

        reverse_feature = ReverseLayerF.apply(x, alpha)
        domain_output = self.domain_classifier(reverse_feature)

        return domain_output


class res_block(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size):
        super(res_block, self).__init__()
        self.conv_1 = nn.Conv1d(in_channels, in_channels, kernel_size = kernel_size, stride=1, padding= 'same', bias=False)
        self.conv_2 = nn.Conv1d(in_channels, out_channels, kernel_size = kernel_size, stride=1, padding= 'same', bias=False)
        self.bn_1 = nn.BatchNorm1d(in_channels)
        self.bn_2 = nn.BatchNorm1d(out_channels)

    def forward(self, x):

        output = self.conv_1(x)
        outout = F.relu(output)
        output = self.bn_1(output)
        output = self.conv_2(output)
        outout = F.relu(output)
        output = self.bn_2(output)
        output = output + x

        return output



class block(nn.Module):
    def __init__(self, input_channel, output_channel, kernel_szie = 5, stride = 1, padding = 'same'):
        super(block, self).__init__()
        self.block = nn.Sequential()
        
        #self.block.add_module('conv', nn.Conv1d(input_channel, output_channel, kernel_size=kernel_szie, stride=stride, padding= padding, bias=False))
        #self.block.add_module('bn', nn.BatchNorm1d(output_channel))
        self.block.add_module('res_block', res_block(output_channel, output_channel, kernel_szie))
        self.block.add_module('pool', nn.MaxPool1d(2, stride = 2))
        
    def forward(self, x):
        
        output = self.block(x)


        return output



class res_sig(nn.Module):
    def __init__(self, mode = 'training_mode'):
        super(res_sig, self).__init__()
        self.res_sig = nn.Sequential()
        self.res_sig.add_module('conv_1', nn.Conv1d(1, 10, kernel_size=10, stride=1, padding= 'same', bias=False))
        
        self.flatten = nn.Flatten()
        self.fc_layer = nn.Linear(640, 4)
        self.block_1 = block(10, 10, 10, 1)
        self.block_2 = block(10, 10, 10, 1)
        self.block_3 = block(32, 64, 3, 1)
        self.pool = nn.MaxPool1d(2, stride = 2)

    def forward(self, x):
        
        output = self.res_sig(x)
        output = self.pool(output)
        output = self.block_1(output)
        output = self.block_2(output)
        #output = self.block_3(output)
        output = self.flatten(output)
        output = self.fc_layer(output)
        

        return output
    
