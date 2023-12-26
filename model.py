from torch import nn
import torch

# 自建模型
class VGG16(nn.Module):
    def __init__(self):
        super(VGG16, self).__init__()
        self.model = nn.Sequential(

            # feature extraction
            # block1
            nn.Conv2d(3, 64, 3, 1, 1), nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, 3, 1, 1), nn.ReLU(inplace=True),
            nn.MaxPool2d(2, stride=2),

            # block2
            nn.Conv2d(64, 128, 3, 1, 1), nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, 3, 1, 1), nn.ReLU(inplace=True),
            nn.MaxPool2d(2, stride=2),

            # block3
            nn.Conv2d(128, 256, 3, 1, 1), nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, 3, 1, 1), nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, 3, 1, 1), nn.ReLU(inplace=True),
            nn.MaxPool2d(2, stride=2),

            # block4
            nn.Conv2d(256, 512, 3, 1, 1), nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, 3, 1, 1), nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, 3, 1, 1), nn.ReLU(inplace=True),
            nn.MaxPool2d(2, stride=2),

            # block5
            nn.Conv2d(512, 512, 3, 1, 1), nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, 3, 1, 1), nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, 3, 1, 1), nn.ReLU(inplace=True),
            nn.MaxPool2d(2, stride=2),

            # classifier
            nn.Flatten(),
            nn.Linear(7 * 7 * 512, 4096), nn.ReLU(inplace=True), nn.Dropout(p=0.5, inplace=False),
            nn.Linear(4096, 4096), nn.ReLU(inplace=True), nn.Dropout(p=0.5, inplace=False),
            nn.Linear(4096, 6),
            nn.Softmax(1)
        )

    def forward(self, _x):
        _x = self.model(_x)
        return _x


# validate the self-built VGG16 model
if __name__ == '__main__':
    model = VGG16()
    x = torch.ones((10, 3, 224, 224))
    y = model(x)
    print(y)
