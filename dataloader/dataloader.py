"""
@author: Shubin Liu
@contact: liushb7@gmail.com
"""

import os
import numpy as np

import torch
from torch.utils.data import Dataset
from torchvision import transforms

class BalancedBatchSampler(torch.utils.data.sampler.BatchSampler):
    """
    BatchSampler - from a MNIST-like dataset, samples n_samples for each of the n_classes.
    Returns batches of size n_classes * (batch_size // n_classes)
    Taken from https://github.com/criteo-research/pytorch-ada/blob/master/adalib/ada/datasets/sampler.py
    """

    def __init__(self, labels, batch_size):
        print("length of labels: ", len(labels))
        self.classes = sorted(set(labels.numpy()))
        print(self.classes)

        n_classes = len(self.classes)
        self._n_samples = batch_size // n_classes
        self._n_remain = batch_size % n_classes
        if self._n_samples == 0:
            raise ValueError(f"batch_size should be bigger than the number of classes, got {batch_size}")

        self._class_iters = [
            InfiniteSliceIterator(np.where(labels == class_)[0], class_=class_) for class_ in self.classes
        ]

        # batch_size = self._n_samples * n_classes
        self.n_dataset = len(labels)
        self._n_batches = int(np.round(self.n_dataset // batch_size))
        print("number of batches: ", self._n_batches)
        if self._n_batches == 0:
            raise ValueError(f"Dataset is not big enough to generate batches with size {batch_size}")
        print("K=", n_classes, "nk=", self._n_samples)
        print("Batch size = ", batch_size)

    def __iter__(self):
        for _ in range(self._n_batches):
            indices = []
            add_class = set(np.random.choice(self.classes, self._n_remain, replace=False))
            for class_iter in self._class_iters:
                if class_iter.class_ in add_class:
                    add_samples = 1
                else:
                    add_samples = 0
                indices.extend(class_iter.get(self._n_samples + add_samples))

            np.random.shuffle(indices)
            yield indices

        for class_iter in self._class_iters:
            class_iter.reset()

    def __len__(self):
        return self._n_batches


class InfiniteSliceIterator:
    def __init__(self, array, class_):
        assert type(array) is np.ndarray
        self.array = array
        self.i = 0
        self.class_ = class_

    def reset(self):
        self.i = 0

    def get(self, n):
        len_ = len(self.array)
        # not enough element in 'array'
        if len_ < n:
            print(f"there are really few items in class {self.class_}")
            self.reset()
            np.random.shuffle(self.array)
            mul = n // len_
            rest = n - mul * len_
            return np.concatenate((np.tile(self.array, mul), self.array[:rest]))

        # not enough element in array's tail
        if len_ - self.i < n:
            self.reset()

        if self.i == 0:
            np.random.shuffle(self.array)
        i = self.i
        self.i += n
        return self.array[i : self.i]



class Load_Dataset(Dataset):
    def __init__(self, dataset):
        super(Load_Dataset, self).__init__()

        X_train = dataset["samples"]
        y_train = dataset["labels"]

        if isinstance(X_train, np.ndarray):
            X_train = torch.from_numpy(X_train)
            y_train = torch.from_numpy(y_train).long()

        # (channel, length)
        if len(X_train.shape) < 3:
            X_train = X_train.unsqueeze(2)
        
        if X_train.shape.index(min(X_train.shape[1], X_train.shape[2])) != 1:  # make sure the Channels in second dim
            X_train = X_train.permute(0, 2, 1)
            # X_train = X_train.permute(0, 2, 1).contiguous()  # ensure the memory is continuous

        self.x_data = X_train
        self.y_data = y_train

        # X_train: (num_samples, num_channels, seq_len)
        self.num_channels = X_train.shape[1]

        # normalize
        data_mean = torch.mean(X_train, dim=(0, 2))
        data_std = torch.std(X_train, dim=(0, 2))
        self.transform = transforms.Normalize(mean=data_mean, std=data_std)
        self.len = X_train.shape[0]

    def __getitem__(self, index):
        if self.transform is not None:
            output = self.transform(self.x_data[index].view(self.num_channels, -1, 1))
            self.x_data[index] = output.view(self.x_data[index].shape)

        return self.x_data[index].float(), self.y_data[index].long()

    def __len__(self):
        return self.len


def data_generator(data_path, domain_id, args, is_src=False):
    """
    Args:
        data_path (str): Path to dataset folder.
        domain_id (str): ID of the domain (e.g., '1', '2' for UCIHAR).
        args (Namespace): Arguments containing batch_size, num_workers, etc.
        is_src (bool): Whether this generator is for the Source domain. 
                       If True, uses BalancedBatchSampler.
    """
    # loading path
    train_dataset = torch.load(os.path.join(data_path, "train_" + domain_id + ".pt"))
    test_dataset = torch.load(os.path.join(data_path, "test_" + domain_id + ".pt"))

    # Loading datasets
    train_dataset = Load_Dataset(train_dataset)
    test_dataset = Load_Dataset(test_dataset)
    batch_size = args.bs

    use_balanced = args.use_balanced_sampler

    if is_src and use_balanced == 1:
        sampler = BalancedBatchSampler(train_dataset.y_data, batch_size=batch_size)
        train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_sampler=sampler, num_workers=args.num_workers)
    else:
        print("Using standard random sampling for the source domain.")
        train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=batch_size,
                                                   shuffle=True, drop_last=True, num_workers=args.num_workers)
        # train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=batch_size,
        #                                            shuffle=False, drop_last=True, num_workers=args.num_workers)
        
    test_loader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=batch_size,
                                              shuffle=False, drop_last=True, num_workers=args.num_workers)
    return train_loader, test_loader
