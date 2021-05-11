import dev

import sys
import os

import torch
import torch.onnx as onnx
import torchvision.models as models

def load_model_class(cls, weights_path):
    model = cls()
    model_weights = torch.load(weights_path)
    model.load_state_dict(model_weights)
    return model

def load_encoder(job_dir):
    weights_path = os.path.join(job_dir, "enc.tar")
    return load_model_class(dev.Encoder, weights_path)

def load_relation_head(job_dir):
    weights_path = os.path.join(job_dir, "rel.tar")
    return load_model_class(dev.RelationHead, weights_path)

def main():
    job_dir = sys.argv[1]
    print("performing a check on", job_dir)
    enc = load_encoder(job_dir)
    rel = load_relation_head(job_dir)
    print("loaded models")
    trainable_model = dev.RelationalEncoder(encoder=enc,relation_head=rel)
    print("initialized trainable model")
    e = 1
    m = 2
    k = 2
    cifar10 = dev.MultiCIFAR10(k,
        root='data',
        download=True,
        train=True,
        transform=dev.train_transform)
    train_loader = dev.DataLoader(cifar10,
        batch_size = m,
        shuffle=True)
    print(f"training for m={m} k={k} over {e} epoch")
    trainable_model.train(e, m, k, train_loader, verbose=True)

if __name__ == "__main__":
    main()
