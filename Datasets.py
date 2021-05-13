import torch
from torch.utils.data import TensorDataset
from torchvision import datasets, transforms
import numpy as np


def MnistLabel(class_num):
    raw_dataset = datasets.MNIST(root="./data", train=True, download=False,
                                 transform=transforms.Compose([
                                     transforms.ToTensor(),
                                 ]))
    class_tot = [0] * 10
    data = []
    labels = []
    positive_tot = 0
    tot = 0
    perm = np.random.permutation(raw_dataset.__len__())
    for i in range(raw_dataset.__len__()):
        datum, label = raw_dataset.__getitem__(perm[i])
        if class_tot[label] < class_num:
            data.append(datum.numpy())
            labels.append(label)
            class_tot[label] += 1
            tot += 1
            if tot >= 10 * class_num:
                break
    tensor_data = torch.FloatTensor(data)
    tensor_labels = torch.LongTensor(labels)
    return TensorDataset(tensor_data, tensor_labels)
    # return TensorDataset(torch.FloatTensor(np.array(data)), torch.LongTensor(np.array(labels)))


def MnistUnlabel():
    raw_dataset = datasets.MNIST(root="./data", train=True, download=False,
                                 transform=transforms.Compose([
                                     transforms.ToTensor(),
                                 ]))
    return raw_dataset


def MnistTest():
    return datasets.MNIST(root="./data", train=False, download=False,
                          transform=transforms.Compose([
                              transforms.ToTensor(),
                          ]))


if __name__ == '__main__':
    mnist_path = "./data"
