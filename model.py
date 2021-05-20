import torch
import torch.nn as nn
import timm

class MyModel(nn.Module):
    def __init__(self, num_classes=10):
        super().__init__()
        model = timm.create_model('efficientnet_b0', pretrained=True, in_chans=3, num_classes=1)
        num_features = model.num_features
        self.extractor_features = nn.Sequential(*(list(model.children())[:-1]))
        self.fc = nn.Linear(num_features, num_classes)
        self.dropout = nn.Dropout(0.5)

    def forward(self, x):
        x = self.extractor_features(x)
        x = self.dropout(x)
        x = self.fc(x)
        return x

if __name__=='__main__':
    model = MyModel()
    print(model)
