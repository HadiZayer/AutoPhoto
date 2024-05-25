import torch
import torch.nn as nn
from torch.autograd import Variable
import torchvision.models as models
from robust_cnns.models_lpf import vgg16



class FullVggCompositionNet(nn.Module):
    def __init__(self, pretrained=True, isFreeze=False, LinearSize1=1024, LinearSize2=512):
        super(FullVggCompositionNet, self).__init__()

        model = models.vgg16(pretrained=pretrained)
        self.features = model.features

        self.classifier = nn.Sequential(
            nn.Linear(512 * 7 * 7, LinearSize1),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(LinearSize1, LinearSize2),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(LinearSize2, 1),
        )

        if isFreeze:
            for param in self.features.parameters():
                if isFreeze:
                    param.requires_grad = False
                else:
                    param.requires_grad = True

    def forward(self, x):
        x = self.features(x)
        x = x.reshape(x.size(0), -1)
        x = self.classifier(x)
        return x

    def robustify(self):
        downsampled_vgg = vgg16(pretrained=False)
        layers = []
        offset = 0
        for i, layer in enumerate(self.features):
            if isinstance(layer, torch.nn.MaxPool2d):
                layers.append(downsampled_vgg.features[i + offset])
                layers.append(downsampled_vgg.features[(i+1) + offset])
                offset += 1
            else:
                layers.append(layer)

        self.features = torch.nn.Sequential(*layers)

        assert len(downsampled_vgg.features) == len(self.features)
        print(self.features)

    def get_name(self):
        return self.__class__.__name__


if __name__ == '__main__':
    test_input = torch.randn([1, 3, 224, 224])
    compNet = FullVggCompositionNet()
    test_input = Variable(test_input)
    output = compNet(test_input)
    print("DEBUG")




