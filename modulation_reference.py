"""PyTorch port of Harper et al. 2023 create_model defaults, for 12-class comparison.

Architecture adapted from caharper (MIT, see third_party/harper2023_LICENSE.txt).
Pinned source: 902cca45c3bf04b5da9f78030372b5a03f730f42,
experiments/radioml2018.01a/{models,layers}.py and amc_neural_networks/layers.py.
This is a port and task adaptation, not a reproduction of the paper's accuracy.
"""
import torch
from torch import nn

REFERENCE = {
    'paper': 'https://doi.org/10.3390/electronics12183962',
    'repository': 'https://github.com/caharper/Automatic-Modulation-Classification-with-Deep-Neural-Networks',
    'commit': '902cca45c3bf04b5da9f78030372b5a03f730f42',
    'models_blob': '88e321d974219486a4cca9509b651003159f791c',
    'se_blob': '3b554be6900f9b4faac184ee704c324df5f97b2e',
    'variance_blob': '43813dd9c236ddfd9a8ec0913d3154368f763a49',
    'source_options': dict(use_se=True, use_act=True, residual=False,
                           dilate=True, self_attention=False),
    'adaptations': ['PyTorch instead of TensorFlow; channels-first input adapter',
                    '4096 samples and 12 classes instead of 1024 and 24',
                    'logits instead of softmax, for CrossEntropyLoss',
                    'our fixed training protocol, not original optimizer/callback settings'],
}


class SqueezeExcitation(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.reduce = nn.Linear(channels, channels // 2, bias=False)
        self.expand = nn.Linear(channels // 2, channels, bias=False)

    def forward(self, x):
        weights = torch.sigmoid(self.expand(torch.relu(self.reduce(x.mean(dim=2)))))
        return x * weights.unsqueeze(2)


class StatisticsPooling(nn.Module):
    def forward(self, x):
        # tf.math.reduce_variance is population variance, not unbiased variance.
        return torch.cat((x.mean(dim=2), x.var(dim=2, correction=0)), dim=1)


class Harper2023(nn.Module):
    def __init__(self, classes=12):
        super().__init__()
        channels = [2, 32, 48, 64, 72, 84, 96, 108]
        kernels = [7, 5, 7, 5, 3, 3, 3]
        dilations = [1, 2, 3, 2, 2, 2, 1]
        layers = []
        for i, (kernel, dilation) in enumerate(zip(kernels, dilations)):
            layers += [nn.Conv1d(channels[i], channels[i+1], kernel,
                        dilation=dilation, padding=dilation*(kernel-1)//2), nn.ReLU()]
            if i < 6:
                layers.append(SqueezeExcitation(channels[i+1]))
        self.features = nn.Sequential(*layers)
        self.pool = StatisticsPooling()
        self.head = nn.Sequential(nn.Linear(216, 128), nn.SELU(),
                                  nn.Linear(128, 128), nn.SELU(), nn.Linear(128, classes))
        # Match the source's initializer families, not framework default initializers.
        for module in self.modules():
            if isinstance(module, (nn.Conv1d, nn.Linear)):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
        for module in self.modules():
            if isinstance(module, SqueezeExcitation):
                # Keras HeNormal uses a corrected truncated normal; reproduce that law.
                for linear in (module.reduce, module.expand):
                    std = (2 / linear.in_features) ** .5 / .8796256610342398
                    nn.init.trunc_normal_(linear.weight, std=std, a=-2*std, b=2*std)

    def forward(self, x):
        # Same NCHW raw-IQ interface as the original baseline; height is one.
        return self.head(self.pool(self.features(x.squeeze(2))))
