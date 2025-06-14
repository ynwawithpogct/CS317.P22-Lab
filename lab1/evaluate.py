import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import copy
import pickle


def evaluate(model, test_loader, save_path='save/evaluate.pkl'):
    ngpu = torch.cuda.device_count()
    device = torch.device("cuda:0" if (torch.cuda.is_available() and ngpu > 0) else "cpu")
             
    temp_model = copy.deepcopy(model)
    temp_model = temp_model.to(device)
    
    # Handle multi-GPU if desired
    if (device.type == 'cuda') and (ngpu > 1):
        temp_model = nn.DataParallel(temp_model, list(range(ngpu)))
     
    temp_model.eval()
    all_preds, all_labels = [], []
    with torch.no_grad():
        for x_test, y_test in test_loader:
            preds = temp_model(x_test)
            preds = preds.argmax(dim=1)
            all_preds.extend(preds.tolist())
            all_labels.extend(y_test.tolist())
        
    result = {     
        'f1': f1_score(all_labels, all_preds),
        'accuracy': accuracy_score(all_labels, all_preds),
        'precision': precision_score(all_labels, all_preds),
        'recall': recall_score(all_labels, all_preds)
    }
    
    if save_path is not None:
        with open(save_path, "wb") as f:
            pickle.dump(result, f)
    return result