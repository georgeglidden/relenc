# all code scooped directly from https://github.com/mpatacchiola/self-supervised-relational-reasoning/
import torchvision
from PIL import Image
class MultiCIFAR10(torchvision.datasets.CIFAR10):

    def __init__(self, K, **kwds):
        super().__init__(**kwds)
        self.K = K

    def __getitem__(self, index):
        img, target = self.data[index], self.targets[index]
        #print(img.shape)
        pic = Image.fromarray(img)
        img_list = list()
        if self.transform is not None:
            for _ in range(self.K):
                img_transformed = self.transform(pic.copy())
                img_list.append(img_transformed)
        else:
            img_list = img
        return img_list, target

from torchvision import transforms
normalize = transforms.Normalize(mean=[0.491, 0.482, 0.447],
    std=[0.247,0.243,0.262])
color_jitter = transforms.ColorJitter(brightness=0.8, contrast=0.8,
    saturation=0.8, hue=0.2)
rnd_color_jitter = transforms.RandomApply([color_jitter], p=0.8)
rnd_gray = transforms.RandomGrayscale(p=0.2)
rnd_rcrop = transforms.RandomResizedCrop(size=32, scale=(0.08, 1.0),
    interpolation=transforms.InterpolationMode.BILINEAR)
rnd_hflip = transforms.RandomHorizontalFlip(p=0.5)
train_transform = transforms.Compose([rnd_rcrop, rnd_hflip, rnd_color_jitter,
    rnd_gray, transforms.ToTensor(), normalize])

import torch
class RelationalReasoning(torch.nn.Module):

    def __init__(self, backbone):
        super(RelationalReasoning, self).__init__()
        feature_size = 64*2
        self.backbone = backbone
        self.relation_head = torch.nn.Sequential(
            nn.Linear(feature_size, 256),
            nn.BatchNorm1d(256),
            nn.LeakyReLU(),
            nn.Linear(256, 1)
        )

    def aggregate(self, features, K):
        relation_pairs_list = list()
        targets_list = list()
        size = int(features.shape[0] / K)
        shifts_counter = 1
        for index_1 in range(0, size*K, size):
            for index_2 in range(index_1+size, size*K, size):
                pos_pair = torch.cat([
                    features[index_1:index_1+size],
                    features[index_2:index_2+size]], 1)
                neg_pair = torch.cat([
                    features[index_1:index_1+size],
                    torch.roll(features[index_2:index_2+size],
                        shifts=shifts_counter, dims=0)], 1)
                relation_pairs_list.append(pos_pair)
                relation_pairs_list.append(neg_pair)
                targets_list.append(torch.ones(size, dtype=torch.float32))
                targets_list.append(torch.zeros(size, dtype=torch.float32))
                shifts_counter += 1
                if shifts_counter >= size:
                    shifts_counter = 1
        relation_pairs = torch.cat(relation_pairs_list, 0)
        targets = torch.cat(targets_list, 0)
        return relation_pairs, targets

    def train(self, epochs, train_loader):
        optimizer = torch.optim.Adam([
            {'params': self.backbone.parameters()},
            {'params': self.relation_head.parameters()}
        ])
        BCE = torch.nn.BCEWithLogitsLoss()
        self.backbone.train()
        self.relation_head.train()
        for E in range(epochs):
            for i, (data_augmented,_) in enumerate(train_loader):
                K = len(data_augmented)
                x = torch.cat(data_augmented, 0)
                optimizer.zero_grad()

                features = self.backbone(x)
                relation_pairs, targets = self.aggregate(features, K)
                score = self.relation_head(relation_pairs).squeeze()

                loss = BCE(score, targets)
                loss.backward()
                optimizer.step()

                predicted = torch.round(torch.sigmoid(score))
                correct = predicted.eq(targets.view_as(predicted)).sum()
                accuracy = (100.0 * correct / float(len(targets)))

                if (i%100 == 0):
                    print("epoch {} {:0.3f}% loss {:.5f} acc {:.2f}".format(E, 100*(16*i/len(train_loader.dataset)), loss.item(), accuracy.item()))

import torch
import torch.nn as nn
import collections
import math
class Conv4(torch.nn.Module):
    def __init__(self, flatten=True):
        super(Conv4, self).__init__()
        self.feature_size = 64
        self.name = "conv4"

        self.layer1 = nn.Sequential(collections.OrderedDict([
          ('conv',    nn.Conv2d(3, 8, kernel_size=3, stride=1, padding=1, bias=False)),
          ('bn',      nn.BatchNorm2d(8)),
          ('relu',    nn.ReLU()),
          ('avgpool', nn.AvgPool2d(kernel_size=2, stride=2))
        ]))

        self.layer2 = nn.Sequential(collections.OrderedDict([
          ('conv',    nn.Conv2d(8, 16, kernel_size=3, stride=1, padding=1, bias=False)),
          ('bn',      nn.BatchNorm2d(16)),
          ('relu',    nn.ReLU()),
          ('avgpool', nn.AvgPool2d(kernel_size=2, stride=2))
        ]))

        self.layer3 = nn.Sequential(collections.OrderedDict([
          ('conv',    nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1, bias=False)),
          ('bn',      nn.BatchNorm2d(32)),
          ('relu',    nn.ReLU()),
          ('avgpool', nn.AvgPool2d(kernel_size=2, stride=2))
        ]))

        self.layer4 = nn.Sequential(collections.OrderedDict([
          ('conv',    nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1, bias=False)),
          ('bn',      nn.BatchNorm2d(64)),
          ('relu',    nn.ReLU()),
          #('avgpool', nn.AvgPool2d(kernel_size=4))
          ('glbpool', nn.AdaptiveAvgPool2d(1))
        ]))

        self.is_flatten = flatten
        self.flatten = nn.Flatten()

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def forward(self, x):
        h = self.layer1(x)
        h = self.layer2(h)
        h = self.layer3(h)
        h = self.layer4(h)
        if(self.is_flatten): h = self.flatten(h)
        return h

def main():
    backbone = Conv4()
    model = RelationalReasoning(backbone)
    train_set = MultiCIFAR10(K=4,
        root='data',
        train=True,
        transform=train_transform,
        download=True)
    train_loader = torch.utils.data.DataLoader(train_set,
        batch_size=16,
        shuffle=True)
    model.train(epochs=1, train_loader=train_loader)
    torch.save(model.backbone.state_dict(), 'enc.tar')
    torch.save(model.relation_head.state_dict(), 'rel.tar')

if __name__ == "__main__":
    main()
