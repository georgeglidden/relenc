# this code is copied (with minor modifications) from [1] and the
# corresponding publication on relational reasoning by Patacchiola and Storkey.
# [1] https://github.com/mpatacchiola/self-supervised-relational-reasoning/
import json
import math
import collections
import torch
from torch import nn

class Encoder(nn.Module):

    def __init__(self, feature_size=64, flatten=True):
        super(Encoder, self).__init__()
        self.feature_size = feature_size
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

class RelationHead(nn.Module):
    def __init__(self, feature_size=64):
        super(RelationHead, self).__init__()
        input_size = feature_size * 2
        self.layer1 = nn.Sequential(
            nn.Linear(input_size, 256),
            nn.BatchNorm1d(256),
            nn.LeakyReLU()
        )
        self.out = nn.Linear(256, 1)

    def forward(self, x):
        h = self.layer1(x)
        h = self.out(h)
        return h

class RelationHeadCone(nn.Module):
    def __init__(self, feature_size=64):
        super(RelationHeadCone, self).__init__()
        input_size = feature_size * 2
        # build layers
        i = 1
        self.layers = []
        while input_size // (2 ** i) > 1:
            self.layers.append(nn.Sequential(
                nn.Linear(input_size // (2 ** (i-1)), input_size // (2 ** i)),
                nn.BatchNorm1d(input_size // (2 ** i)),
                nn.LeakyReLU()
            ))
            i += 1
        # map to similarity score
        self.out = nn.Sequential(
            nn.Linear(input_size // (2 ** (i-1)), 1)
        )

    def forward(self, x):
        for L in self.layers:
            x = L(x)
        return self.out(x)

class RelationHead2(nn.Module):
    def __init__(self, feature_size=64):
        super(RelationHead2, self).__init__()
        input_size = feature_size * 2
        self.layer1 = nn.Sequential(
            nn.Linear(input_size, 256),
            nn.BatchNorm1d(256),
            nn.LeakyReLU()
        )
        self.layer2 = nn.Sequential(
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.LeakyReLU()
        )
        self.out = nn.Linear(128, 1)

    def forward(self, x):
        h = self.layer1(x)
        h = self.layer2(h)
        h = self.out(h)
        return h

class RelationHead3(nn.Module):
    def __init__(self, feature_size=64):
        super(RelationHead3, self).__init__()
        input_size = feature_size * 2
        self.layer1 = nn.Sequential(
            nn.Linear(input_size, 256),
            nn.BatchNorm1d(256),
            nn.LeakyReLU()
        )
        self.layer2 = nn.Sequential(
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.LeakyReLU()
        )
        self.layer3 = nn.Sequential(
            nn.Linear(128, 64),
            nn.BatchNorm1d(64),
            nn.LeakyReLU()
        )
        self.out = nn.Linear(64, 1)

    def forward(self, x):
        h = self.layer1(x)
        h = self.layer2(h)
        h = self.layer3(h)
        h = self.out(h)
        return h

class RelationHeadLong(nn.Module):
    def __init__(self, feature_size=64):
        super(RelationHeadLong, self).__init__()
        input_size = feature_size * 2
        self.layer1 = nn.Sequential(
            nn.Linear(input_size, 256),
            nn.BatchNorm1d(256),
            nn.LeakyReLU()
        )
        self.layer2 = nn.Sequential(
            nn.Linear(256, 256),
            nn.BatchNorm1d(256),
            nn.LeakyReLU()
        )
        self.layer3 = nn.Sequential(
            nn.Linear(256, 64),
            nn.BatchNorm1d(64),
            nn.LeakyReLU()
        )
        self.layer4 = nn.Sequential(
            nn.Linear(64, 64),
            nn.BatchNorm1d(64),
            nn.LeakyReLU()
        )
        self.layer5 = nn.Sequential(
            nn.Linear(64, 16),
            nn.BatchNorm1d(16),
            nn.LeakyReLU()
        )
        self.layer6 = nn.Sequential(
            nn.Linear(16, 16),
            nn.BatchNorm1d(16),
            nn.LeakyReLU()
        )
        self.layer7 = nn.Sequential(
            nn.Linear(16, 4),
            nn.BatchNorm1d(4),
            nn.LeakyReLU()
        )
        self.out = nn.Linear(4, 1)

    def forward(self, x):
        h = self.layer1(x)
        h = self.layer2(h)
        h = self.layer3(h)
        h = self.layer4(h)
        h = self.layer5(h)
        h = self.layer6(h)
        h = self.layer7(h)
        h = self.out(h)
        return h

def aggregate(features, K):
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
                if (shifts_counter >= size):
                    shifts_counter = 1
        relation_pairs = torch.cat(relation_pairs_list, 0)
        targets = torch.cat(targets_list, 0)
        return relation_pairs, targets

# implements Patacchiola & Storkey's Relational Reasoning model
class RelEnc(nn.Module):
    def __init__(self, encoder = None, relation_head = None, feature_size=64):
        super(RelEnc, self).__init__()
        if encoder == None:
            encoder = Encoder(feature_size=feature_size)
        if relation_head == None:
            relation_head = RelationHead(feature_size=feature_size)
        self.encoder = encoder
        self.relation_head = relation_head

    def train(self, epochs, k, m, train_loader, log_file="train_session.json", nb_records=100, verbose=False):
        log = dict()
        log["epochs"] = epochs
        optimizer = torch.optim.Adam([
            {'params': self.encoder.parameters()},
            {'params': self.relation_head.parameters()}
        ])
        log["optimizer"] = str(optimizer)
        loss_fn = torch.nn.BCEWithLogitsLoss()
        log["loss"] = {
            "function": str(loss_fn),
            "average": 0.0
        }
        self.encoder.train()
        self.relation_head.train()
        avg_loss = 0
        for E in range(epochs):
            sum_loss = 0.0
            log["loss"][E] = []
            if verbose:
                print("epoch", E)
            for i, (data_augmented,_) in enumerate(train_loader):
                K = len(data_augmented)
                x = torch.cat(data_augmented, 0)
                optimizer.zero_grad()

                features = self.encoder(x)
                relation_pairs, targets = aggregate(features, K)
                score = self.relation_head(relation_pairs).squeeze()
                loss = loss_fn(score, targets)
                loss.backward()
                optimizer.step()

                sum_loss += loss.detach()
                nb_batches = len(train_loader.dataset) // k^2
                if ((i % (nb_batches // nb_records)) == 0):
                    log["loss"][E].append(loss.detach().item())
                    if verbose:
                        print(int(100 * i / nb_batches), i, loss.detach().item())
            avg_loss += sum_loss / (epochs * nb_batches)
            sum_loss = 0
        log["loss"]["average"] = avg_loss.item()
        return log
