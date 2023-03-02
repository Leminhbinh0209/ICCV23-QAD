from __future__ import print_function

import os
from plistlib import load
import socket
import numpy as np
import random
import pickle
from io import BytesIO
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import datasets
from torchvision import transforms
from albumentations import JpegCompression



def get_dataloader(data_info,
                    dataset_name,
                    normalization,
                    image_size,
                    batch_size,
                    num_workers,
                    return_pair, 
                    aug = None, 
                    intra=False, 
                    test_random_compression=False):
    if aug is None:
        tensor_transform_train = transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalization
        ])
        # remove the random flipping
        tensor_transform_val = transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            normalization
        ])
        print("Train set creating ")
        train_aug = None

        train_dataset = FaceDataset(dataframe=data_info['train'],
                                    aug_transform=train_aug,
                                    tensor_transform=tensor_transform_val if return_pair else tensor_transform_train,
                                    return_pair=return_pair,
                                    mode='train', random_compress=False)
        print("Val and test set creating ")
        val_dataset = FaceDataset(dataframe=data_info['val'],
                                tensor_transform=tensor_transform_val,
                                return_pair=False,  # Validation and test set do not require to shuffle
                                mode='val', random_compress=False)

        test_dataset = FaceDataset(dataframe=data_info['test'],
                                aug_transform=None,
                                tensor_transform=tensor_transform_val,
                                return_pair=False,
                                mode='test', 
                                random_compress=test_random_compression)
   

    train_loader = DataLoader(train_dataset, batch_size=batch_size,
                              shuffle=True, num_workers=num_workers, pin_memory=False)
    val_loader = DataLoader(val_dataset, batch_size=batch_size*1,
                            shuffle=False, num_workers=num_workers, pin_memory=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size*1,
                             shuffle=False, num_workers=num_workers, pin_memory=False)
    return train_loader, val_loader, test_loader


class FaceDataset(Dataset):
    """
    Face data loader. It will work as nomal if ```include_compress = false```
    args:
        dataframe: dictionary: {raw: [], c23: [], c40: [] ,... , labels: []}
        return_pair: boolean, return images by pair with their quality
    """

    def __init__(self, dataframe,
                 aug_transform=None,
                 tensor_transform=None,
                 return_pair=False,
                 mode='train',
                 random_compress=False):
        self.common_size = 128
        assert "labels" in dataframe, "No labels found in the datasframe"
        self.dataframe = dataframe
        self.labels = dataframe["labels"]
        self.aug_transform = aug_transform
        self.tensor_transform = tensor_transform
        self.mode = mode
        self.return_pair = return_pair

        if self.return_pair:
            print("Include the compress version in the dataset loader")
            self.data_quality = [k for k in dataframe if k != "labels"]

        else:
            print("Shuffle the data and return any type ...")
            # flatten all directory
            self.img_paths = sum([dataframe[k]
                                 for k in dataframe if k != "labels"], [])

            self.labels *= (len(self.img_paths)//len(self.labels))
            assert len(self.img_paths) == len(
                self.labels), "# Data dir not equal # labels"
            print(
                f"Total number of {self.mode} samples: \t\t{len(self.labels)}")
        self.random_compress = random_compress
        if self.mode == "test":
            if random_compress:
                print("Include random compression")
                self.aug = np.load("./dataset/random_compress_idx.npy")

    def __getitem__(self, index):
        output = dict()
        if self.random_compress:
            self.aug_transform = JpegCompression(quality_lower=self.aug[index], quality_upper=self.aug[index], always_apply=True, p=1.0)
        if self.return_pair:
            rand_flip = np.random.rand()
            # increase the randomness but nothing
            random.shuffle(self.data_quality)
            for k in self.data_quality:  # walk through all data quality version
                img_path_comp = self.dataframe[k][index]
                img_comp = Image.open(img_path_comp)
                img_comp = img_comp.resize(
                    (self.common_size, self.common_size))
                if (self.mode=='train') and( rand_flip < 0.5):  # Apply random flipping
                    img_comp = img_comp.transpose(Image.FLIP_LEFT_RIGHT)
                if self.aug_transform is not None:
                    img_comp =  Image.fromarray(
                        self.aug_transform(image=np.asarray(img_comp))["image"]
                    )
                if self.tensor_transform is not None:
                    img_comp = self.tensor_transform(img_comp)
                output[k] = img_comp
            output["labels"] = self.labels[index]
        else:
            img_path = self.img_paths[index]
            label = self.labels[index]
            img = Image.open(img_path)
            img = img.resize((self.common_size, self.common_size))
            if self.aug_transform is not None:
        
                img = Image.fromarray(
                        self.aug_transform(image=np.asarray(img))["image"]
                    )
          
            if self.tensor_transform is not None:
                img = self.tensor_transform(img)
            output = {"img": img, "labels": label}
        return output

    def __len__(self):
        return len(self.labels)



if __name__ == '__main__':
    print("NO dummy test")
