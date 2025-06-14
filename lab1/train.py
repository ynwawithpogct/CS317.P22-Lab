import torch
import torch.nn as nn
import mlflow
import mlflow.pytorch
from sklearn.metrics import f1_score
import copy


def train(
    experiment_name,
    model, 
    train_loader, val_loader, 
    num_epochs = 20, 
    learning_rate = 0.01, weight_decay = 0.001, momentum = 0.9,
    save_path = 'save/model_weights.pth',
):
    ngpu = torch.cuda.device_count()
    device = torch.device("cuda:0" if (torch.cuda.is_available() and ngpu > 0) else "cpu")
     
    mlflow.set_experiment(experiment_name)
    
    if isinstance(train_loader, list):
        best_model = None
        best_f1 = 0.0
        for fold, (loader_train, loader_val) in enumerate(zip(train_loader, val_loader)):
            with mlflow.start_run(run_name=f"Fold-{fold}"):
                mlflow.log_params({
                    'train_type': 'cv',
                    'num_epochs': num_epochs,
                    'fold': fold,
                    'learning_rate': learning_rate,
                    'weight_decay': weight_decay,
                    'momentum': momentum,
                    'model_name': model.__class__.__name__,
                    'batch_size': loader_train.batch_size
                })
                best_model_fold = None
                best_f1_fold = 0.0
                
                temp_model = copy.deepcopy(model)
                temp_model = temp_model.to(device)
                
                # Handle multi-GPU if desired
                if (device.type == 'cuda') and (ngpu > 1):
                    temp_model = nn.DataParallel(temp_model, list(range(ngpu)))
                
                # Loss and optimizer
                criterion = nn.CrossEntropyLoss()
                optimizer = torch.optim.SGD(temp_model.parameters(), lr=learning_rate, weight_decay = weight_decay, momentum = momentum) 
    
                for epoch in range(num_epochs):
                    temp_model.train()
                    all_preds, all_labels = [], []
                    train_loss = 0
                    for x_batch, y_batch in loader_train:
                        optimizer.zero_grad()
                        outputs = temp_model(x_batch)
                        loss = criterion(outputs, y_batch)
                        loss.backward()
                        optimizer.step()
                        all_preds.extend(outputs.tolist())
                        all_labels.extend(y_batch.tolist())
                        
                        train_loss += loss.item()
                    
                    epoch_loss = train_loss / len(loader_train)   
                    train_f1 = f1_score(all_labels, all_preds) 
                        
                    temp_model.eval()
                    all_preds, all_labels = [], []
                    val_loss = 0
                    with torch.no_grad():
                        for x_val, y_val in loader_val:
                            preds = temp_model(x_val)
                            preds = preds.argmax(dim=1)
                            loss = criterion(preds, y_val)
                            all_preds.extend(preds.tolist())
                            all_labels.extend(y_val.tolist())
                        
                            val_loss += loss.item()
                            
                    val_epoch_loss = val_loss / len(loader_val)   
                    val_f1 = f1_score(all_labels, all_preds)
                
                    mlflow.log_metrics(
                        {
                            "train_loss": epoch_loss,
                            "val_loss": val_epoch_loss,
                            "train_f1": train_f1,
                            "val_f1": val_f1,
                        },
                        step=epoch,
                    )
                    
                    if val_f1 > best_f1_fold:
                        best_f1_fold = val_f1
                        best_model_fold = copy.deepcopy(temp_model)
                
                mlflow.log_metric("best_f1", best_f1_fold)
                mlflow.pytorch.log_model(best_model_fold, "model")

                print(f"✅ F1 fold {fold}: {best_f1_fold:.4f}")
                
                if best_f1_fold > best_f1:
                    best_model = copy.deepcopy(best_model_fold)
                    best_f1 = best_f1_fold
                    
        with mlflow.start_run(run_name="Train"):
            mlflow.log_params({
                'train_type': 'cv',
                'num_epochs': num_epochs,
                'learning_rate': learning_rate,
                'weight_decay': weight_decay,
                'momentum': momentum,
                'model_name': model.__class__.__name__,
                'batch_size': train_loader[0].batch_size
            })
            mlflow.log_metric("best_f1", best_f1)
            mlflow.pytorch.log_model(best_model, "model")
            
            print(f"✅ Model: {best_model:.4f}")
            
            torch.save(best_model.state_dict(), save_path)
            
        return best_model
            
    else:
        best_model = None
        best_f1 = 0.0
        with mlflow.start_run(run_name="Train"):
            mlflow.log_params({
                'train_type': 'split',
                'num_epochs': num_epochs,
                'learning_rate': learning_rate,
                'weight_decay': weight_decay,
                'momentum': momentum,
                'model_name': model.__class__.__name__,
                'batch_size': train_loader.batch_size
            })
            
            temp_model = copy.deepcopy(model)
            temp_model = temp_model.to(device)
            
            # Handle multi-GPU if desired
            if (device.type == 'cuda') and (ngpu > 1):
                temp_model = nn.DataParallel(temp_model, list(range(ngpu)))
                
            # Loss and optimizer
            criterion = nn.CrossEntropyLoss()
            optimizer = torch.optim.SGD(temp_model.parameters(), lr=learning_rate, weight_decay = weight_decay, momentum = momentum) 

            for epoch in range(num_epochs):
                temp_model.train()
                all_preds, all_labels = [], []
                train_loss = 0
                for x_batch, y_batch in train_loader:
                    optimizer.zero_grad()
                    outputs = temp_model(x_batch)
                    loss = criterion(outputs, y_batch)
                    loss.backward()
                    optimizer.step()
                    all_preds.extend(outputs.tolist())
                    all_labels.extend(y_batch.tolist())
                    
                    train_loss += loss.item()
                
                epoch_loss = train_loss / len(train_loader)   
                train_f1 = f1_score(all_labels, all_preds) 
                    
                temp_model.eval()
                all_preds, all_labels = [], []
                val_loss = 0
                with torch.no_grad():
                    for x_val, y_val in val_loader:
                        preds = temp_model(x_val)
                        preds = preds.argmax(dim=1)
                        loss = criterion(preds, y_val)
                        all_preds.extend(preds.tolist())
                        all_labels.extend(y_val.tolist())
                    
                        val_loss += loss.item()
                        
                val_epoch_loss = val_loss / len(val_loader)   
                val_f1 = f1_score(all_labels, all_preds)
            
                mlflow.log_metrics(
                    {
                        "train_loss": epoch_loss,
                        "val_loss": val_epoch_loss,
                        "train_f1": train_f1,
                        "val_f1": val_f1,
                    },
                    step=epoch,
                )
                
                if val_f1 > best_f1:
                    best_model = copy.deepcopy(temp_model)
                    best_f1 = val_f1
                
            mlflow.log_metric("best_f1", best_model)
            mlflow.pytorch.log_model(best_f1, "model")

            print(f"✅ Model: {best_model:.4f}")
            
            torch.save(best_model.state_dict(), save_path)
            
        return best_model
    