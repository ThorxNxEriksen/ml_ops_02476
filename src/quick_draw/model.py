import timm
import torch.nn as nn

class QuickDrawModel(nn.Module):
    """"
    A model for Quick, Draw! dataset using EfficientNet-lite0 as the base model.
    It replaces the classifier layer to match the number of classes in our example.
    """
    def __init__(self, num_classes=10):  # Adjust num_classes 
        super(QuickDrawModel, self).__init__()

        # Load a pre-trained timm model (EfficientNet-lite0)
        self.base_model = timm.create_model('tf_efficientnet_lite0', pretrained=True, in_chans=1)  # in_chans=1 for grayscale input
        
        # Replace the classifier to match the number of classes in Quick, Draw!
        in_features = self.base_model.classifier.in_features
        self.base_model.classifier = nn.Linear(in_features, num_classes)

        # Freeze the base model parameters (all layers except the classifier)
        for param in self.base_model.parameters():
            param.requires_grad = False

        # Unfreeze the classifier layer so it can be trained
        for param in self.base_model.classifier.parameters():
            param.requires_grad = True


    def forward(self, x):
        return self.base_model(x)
    
