"""Explicit experimental SE gain; historical model implementation stays unchanged."""
from torch import nn
from modulation_reference import Harper2023, SqueezeExcitation


class ScaledSqueezeExcitation(SqueezeExcitation):
    def __init__(self, original, gain):
        nn.Module.__init__(self)
        if gain not in (1., 2.): raise ValueError('Expected SE gain 1 or 2')
        self.reduce = original.reduce
        self.expand = original.expand
        self.gain = float(gain)

    def forward(self, x):
        return super().forward(x) * self.gain


def make_gain_model(gain=1., tensorflow_selu=True):
    model = Harper2023(tensorflow_selu=tensorflow_selu)
    for index, module in enumerate(model.features):
        if isinstance(module, SqueezeExcitation):
            model.features[index] = ScaledSqueezeExcitation(module, gain)
    return model


def set_gain(model, gain):
    if gain not in (1., 2.): raise ValueError('Expected SE gain 1 or 2')
    modules = [m for m in model.modules() if isinstance(m, ScaledSqueezeExcitation)]
    if len(modules) != 6: raise ValueError('Expected six experimental SE blocks')
    for module in modules: module.gain = float(gain)


def restore_gain_checkpoint(checkpoint):
    if checkpoint.get('architecture') != 'Harper2023-experimental-se-gain':
        raise ValueError('Not an explicitly tagged SE gain checkpoint')
    activation = checkpoint.get('activation')
    if activation not in ('tf-compatible', 'historical'):
        raise ValueError('Unknown checkpoint activation')
    model = make_gain_model(checkpoint['se_gain'], tensorflow_selu=activation == 'tf-compatible')
    model.load_state_dict(checkpoint['model'])
    return model
