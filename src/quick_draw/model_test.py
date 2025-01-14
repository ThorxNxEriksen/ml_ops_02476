import timm
import torch.nn as nn

class QuickDrawModel(nn.Module):
    def __init__(self, num_classes=10):  # Adjust num_classes 
        super(QuickDrawModel, self).__init__()

        # Load a pre-trained timm model (EfficientNet-lite0)
        self.base_model = timm.create_model('tf_efficientnet_lite0', pretrained=True, in_chans=1)  # in_chans=1 for grayscale input
        
        # Replace the classifier to match the number of classes in Quick, Draw!
        in_features = self.base_model.classifier.in_features
        self.base_model.classifier = nn.Linear(in_features, num_classes)

    def forward(self, x):
        return self.base_model(x)
    
