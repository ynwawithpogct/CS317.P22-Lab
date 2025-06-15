from torchvision.transforms import functional as F
import torchvision.transforms as transforms
from torchvision.transforms import v2

class TensorToSquare:
    def __init__(self, fill=0):
        super().__init__()
        self.fill = fill

    def __call__(self, pic):
        w, h = pic.size
        max_wh = max(w, h)
        return F.pad(pic, ((max_wh - w) // 2, (max_wh - h) // 2, (max_wh - w + 1) // 2, (max_wh - h + 1) // 2), fill=self.fill)

    def __repr__(self):
        return self.__class__.__name__ + '()'

class PassThrough:
    def __call__(self, pic):
        return pic

    def __repr__(self):
        return self.__class__.__name__ + '()'
    
transform_dict = {
    'RandomHorizontalFlip': transforms.RandomHorizontalFlip(p=0.5),
    'RandomVerticalFlip': transforms.RandomVerticalFlip(p=0.5),
    'RandomOppositeFlip': transforms.RandomApply(
        [
            transforms.RandomHorizontalFlip(p=1.0),
            transforms.RandomVerticalFlip(p=1.0),
        ], 
        p=0.5
    ),
    'RandomFlip': transforms.RandomChoice([
        transforms.RandomHorizontalFlip(p=1.0),
        transforms.RandomVerticalFlip(p=1.0),
        transforms.Compose([
            transforms.RandomHorizontalFlip(p=1.0),
            transforms.RandomVerticalFlip(p=1.0),
        ]),
        PassThrough(),
    ]),
    'ColorJitter': transforms.RandomApply(
        [
            transforms.ColorJitter(
                brightness=0.4, 
                contrast=0.4, 
                saturation=0.4, 
                hue=0.1
                )
        ], 
        p=0.5
    ),
    'RandomPerspective': transforms.RandomPerspective(p=0.5),
    'RandomRotation': transforms.RandomApply(
        [transforms.RandomRotation(degrees=45)],
        p=0.5
    ),
    'RandomAffine': transforms.RandomApply(
        [
            transforms.RandomAffine(degrees=15, translate=(0.1,0.1))
        ],
        p=0.5
    ),
    'RandomDistort': transforms.RandomChoice([
        transforms.RandomPerspective(p=1.0),
        transforms.RandomRotation(degrees=45),
        transforms.RandomAffine(degrees=15, translate=(0.1,0.1)),
        PassThrough(),
    ]),
    'RandomErasing': transforms.RandomErasing(p=0.5, value=0, scale=(0.01, 0.1), ratio=(0.3, 3.3)),
    'GaussianBlur': transforms.RandomApply(
        [
            transforms.GaussianBlur(kernel_size=5, sigma=(0.1,2.0))
        ],
        p=0.5
    ),
    'GaussianNoise': transforms.RandomApply(
        [
            v2.GaussianNoise(sigma=0.1)
        ],
        p=0.5
    ),
    'RandomNoise': transforms.RandomChoice([
        transforms.RandomErasing(p=1.0, value=0, scale=(0.01, 0.1), ratio=(0.3, 3.3)),
        transforms.GaussianBlur(kernel_size=5, sigma=(0.1,2.0)),
        transforms.v2.GaussianNoise(sigma=0.1),
        PassThrough(),
    ]),
}

def get_transform(transform_list=None):
    if transform_list is None:
        return transforms.Compose([
            TensorToSquare(),
            transforms.Resize((224,224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
    else:
        transforms_Compose =  [
            TensorToSquare(),
            transforms.Resize((224,224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
        transforms_Compose[3:3] =  [transform_dict[t] for t in transform_list]
        return transforms.Compose(transforms_Compose)
