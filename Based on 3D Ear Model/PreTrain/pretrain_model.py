import torch.nn as nn
from feature_extraction import FeatureExtraction
from mlp import MLP

class PretrainModel(nn.Module):
    def __init__(self):
        super(PretrainModel, self).__init__()
        self.feature_extraction = FeatureExtraction(n_input_channels=1)
        self.mlp = MLP()

    def forward(self, mesh):
        features = self.feature_extraction(mesh)
        features_params = self.mlp(features)
        return features_params
