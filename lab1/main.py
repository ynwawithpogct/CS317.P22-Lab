from download_data import download_animals10
from set_seed import set_seed
from data_processing import get_dataloader
from model import get_model
from train import train
from evaluate import evaluate

download_animals10()
set_seed(21520398)
train_loader, val_loader, test_loader = get_dataloader(data_dir="data/dataset", test_ratio=0.2, val_ratio=('split', 0.25))
model = get_model(model_name='resnet50')
model = train(
    experiment_name='resnet50-animals10-split-v1', 
    model=model, train_loader=train_loader, val_loader=val_loader, 
    save_path='save/resnet50-animals10-split-v1.pth'
)
result = evaluate(
    model=model, test_loader=test_loader, 
    save_path='save/resnet50-animals10-split-v1.pkl'
)
print(result)