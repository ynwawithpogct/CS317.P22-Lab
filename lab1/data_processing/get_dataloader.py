import torch
import torchvision.datasets as datasets
from torch.utils.data import DataLoader, random_split, Subset
from sklearn.model_selection import KFold
from .transforms import get_transform
import numpy as np
import os

def get_dataloader(
    data_dir = "data/dataset", 
    test_ratio = 0.2, 
    val_ratio = ('split', 0.25), 
    batch_size = 32,
    printting = True,
    transform_list = ['RandomFlip', 'ColorJitter', 'RandomDistort', 'RandomNoise'],
    num_workers=1
):
    initial_transform = get_transform()

    Dataset = datasets.ImageFolder(root=data_dir, transform=initial_transform)

    if printting:
        num_classes = len(Dataset.classes)
        print(f"Number of classes: {num_classes}")

        num_samples = len(Dataset)
        print(f"Number of samples: {num_samples}")

    train_val_size = int((1-test_ratio) * len(Dataset))  
    test_size = len(Dataset) - train_val_size 

    if printting:
        print(f"Number of training and validation samples: {train_val_size}")
        print(f"Number of testing samples: {test_size}")

    train_val_idx, test_idx = random_split(range(len(Dataset)), [train_val_size, test_size])

    test_dataset = Subset(Dataset, test_idx)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    train_transform =  get_transform(transform_list)

    if val_ratio[0] == 'split':
        train_size = int((1-val_ratio[1]) * train_val_size)  
        val_size = train_val_size - train_size

        if printting:
            print(f"Number of training samples: {train_size}")
            print(f"Number of validation samples: {val_size}")

        train_idx, val_idx = random_split(range(len(train_val_idx)), [train_size, val_size])

        train_dataset = Subset(Subset(datasets.ImageFolder(root=data_dir, transform=train_transform), train_val_idx), train_idx)
        val_dataset = Subset(Subset(Dataset, train_val_idx), val_idx)
            
        targets = torch.tensor([s[1] for s in train_dataset.samples])
        class_sample_count = np.array([sum(targets == i).item() for i in range(num_classes)])
        class_weights = 1. / class_sample_count
        samples_weight = torch.tensor([class_weights[t] for t in targets], dtype=torch.double)

        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, sampler=samples_weight, num_workers=num_workers)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    else:
        hash_seed = os.environ.get("PYTHONHASHSEED", None)
        kf = KFold(n_splits=val_ratio[1], shuffle=True, random_state=hash_seed)

        train_loader, val_loader =  [], []
        for fold, (train_idx, val_idx) in enumerate(kf.split(train_val_idx)):
            if printting:
                print(f"====== Fold {fold+1} / {val_ratio[1]} ======")
            
            train_dataset = Subset(Subset(datasets.ImageFolder(root=data_dir, transform=train_transform), train_val_idx), train_idx)
            val_dataset = Subset(Subset(Dataset, train_val_idx), val_idx)
            
            targets = torch.tensor([s[1] for s in train_dataset.samples])
            class_sample_count = np.array([sum(targets == i).item() for i in range(num_classes)])
            class_weights = 1. / class_sample_count
            samples_weight = torch.tensor([class_weights[t] for t in targets], dtype=torch.double)

            if printting:
                print(f"Number of training samples: {len(train_dataset)}")
                print(f"Number of validation samples: {len(val_dataset)}")

            train_loader.append(DataLoader(train_dataset, batch_size=batch_size, shuffle=True, sampler=samples_weight, num_workers=num_workers))
            val_loader.append(DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers))
            
    return train_loader, val_loader, test_loader