import torchvision.models as models
import torch.nn as nn


def get_model(model_name, num_class=10):
    if model_name == 'resnet50':
        resnet = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)

        for name, param in resnet.named_parameters():
            if "layer4" not in name and "fc" not in name:
                param.requires_grad = False

        in_features = resnet.fc.in_features
        resnet.fc = nn.Linear(in_features, num_class)
        return resnet
    
    if model_name == 'convnext_tiny':
        convnext_tiny = models.convnext_tiny(weights=models.ConvNeXt_Tiny_Weights.DEFAULT)

        for name, param in convnext_tiny.named_parameters():
            if any(stage in name for stage in ['features.0', 'features.1', 'features.2', 'features.3', 'features.4', 'features.5']):
                param.requires_grad = False

        in_features = convnext_tiny.classifier[2].in_features 
        convnext_tiny.classifier[2] = nn.Linear(in_features, 10)
        return convnext_tiny
    
    raise NotImplementedError(f"{model_name} has not been implemented yet!")
