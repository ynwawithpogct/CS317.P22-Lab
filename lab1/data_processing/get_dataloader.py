import torch
import torchvision.datasets as datasets
from torch.utils.data import DataLoader, Subset
from sklearn.model_selection import KFold, train_test_split
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
    num_workers=0
):
    num_workers=num_workers if num_workers>=0 else os.cpu_count()
    
    initial_transform = get_transform()

    Dataset = datasets.ImageFolder(root=data_dir, transform=None)

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

    hash_seed = int(os.environ.get("PYTHONHASHSEED", None))
    train_val_idx, test_idx = train_test_split(range(len(Dataset)), test_size=test_ratio,random_state=hash_seed)

    if printting:
        print(f"Number of training and validation samples idx: {len(train_val_idx)}-{type(train_val_idx)}-{type(train_val_idx[0])}")
        print(f"Number of testing samples idx: {len(test_idx)}-{type(test_idx)}-{type(test_idx[0])}")

    test_dataset = Subset(Dataset, test_idx)
    
    Dataset.transform = initial_transform
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    train_transform =  get_transform(transform_list)

    if val_ratio[0] == 'split':
        train_size = int((1-val_ratio[1]) * train_val_size)  
        val_size = train_val_size - train_size

        if printting:
            print(f"Number of training samples: {train_size}")
            print(f"Number of validation samples: {val_size}")

        hash_seed = int(os.environ.get("PYTHONHASHSEED", None))
        train_idx, val_idx = train_test_split(range(len(train_val_idx)), test_size=val_ratio[1],random_state=hash_seed)
        
        if printting:
            print(f"Number of training samples idx: {len(train_idx)}-{type(train_idx)}-{type(train_idx[0])}")
            print(f"Number of validation samples idx: {(len(val_idx))}-{type(val_idx)}-{type(val_idx[0])}")

        idx_train = [train_val_idx[i] for i in train_idx]
        idx_val = [train_val_idx[i] for i in val_idx]
               
        if printting:
            print(f"Idx number of training samples: {len(idx_train)}-{type(idx_train)}-{type(idx_train[0])}")
            print(f"Idx number of validation samples: {(len(idx_val))}-{type(idx_val)}-{type(idx_val[0])}")
            
        Dataset.transform = train_transform
        train_dataset = Subset(Dataset, idx_train)
        Dataset.transform = initial_transform
        val_dataset = Subset(Dataset, idx_val)
        
        # if printting:
        #     print(f"train_idx: {type(train_dataset.indices)} - train_val_idx: {type(train_dataset.dataset.indices)}")
            
        # original_samples = train_dataset.dataset.dataset.samples
        # effective_indices = [train_val_idx[i] for i in train_idx]
        targets = torch.tensor([s[1] for s in train_dataset.dataset.samples])
        class_sample_count = np.array([sum(targets == i).item() for i in range(num_classes)])
        class_weights = 1. / class_sample_count
        samples_weight = torch.tensor([class_weights[t] for t in targets], dtype=torch.double)

        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=False, sampler=samples_weight, num_workers=num_workers) # shuffle=True or sampler=samples_weight
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    else:
        hash_seed = int(os.environ.get("PYTHONHASHSEED", None))
        kf = KFold(n_splits=val_ratio[1], shuffle=True, random_state=hash_seed)

        train_loader, val_loader =  [], []
        for fold, (train_idx, val_idx) in enumerate(kf.split(train_val_idx)):
            if printting:
                print(f"====== Fold {fold+1} / {val_ratio[1]} ======")
                    
            if printting:
                print(f"Number of training samples idx: {len(train_idx)}-{type(train_idx)}-{type(train_idx[0])}")
                print(f"Number of validation samples idx: {(len(val_idx))}-{type(val_idx)}-{type(val_idx[0])}")
            
            idx_train = [train_val_idx[i] for i in train_idx]
            idx_val = [train_val_idx[i] for i in val_idx]
                   
            if printting:
                print(f"Idx number of training samples: {len(idx_train)}-{type(idx_train)}-{type(idx_train[0])}")
                print(f"Idx number of validation samples: {(len(idx_val))}-{type(idx_val)}-{type(idx_val[0])}")
            
            Dataset.transform = train_transform
            train_dataset = Subset(Dataset, idx_train)
            Dataset.transform = initial_transform
            val_dataset = Subset(Dataset, idx_val)
            
            # original_samples = train_dataset.dataset.dataset.samples
            # effective_indices = [train_val_idx[i] for i in train_idx]
            targets = torch.tensor([s[1] for s in train_dataset.dataset.samples])
            class_sample_count = np.array([sum(targets == i).item() for i in range(num_classes)])
            class_weights = 1. / class_sample_count
            samples_weight = torch.tensor([class_weights[t] for t in targets], dtype=torch.double)

            if printting:
                print(f"Number of training samples: {len(train_dataset)}")
                # print(f"train_idx: {type(train_dataset.indices)} - train_val_idx: {type(train_dataset.dataset.indices)}")
                print(f"Number of validation samples: {len(val_dataset)}")

            train_loader.append(DataLoader(train_dataset, batch_size=batch_size, shuffle=False, sampler=samples_weight, num_workers=num_workers)) # shuffle=True or sampler=samples_weight
            val_loader.append(DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers))
            
    return train_loader, val_loader, test_loader