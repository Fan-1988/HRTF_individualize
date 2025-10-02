import torch
import torch.nn as nn

class CosineSimilarityLoss(nn.Module):
    def __init__(self, eps=1e-8):
        super(CosineSimilarityLoss, self).__init__()
        self.cos_sim = nn.CosineSimilarity(dim=1, eps=eps)

    def forward(self, input1, input2):
        # 计算余弦相似度
        cos_sim = self.cos_sim(input1, input2)
        # 返回 1 减去余弦相似度（因为我们希望最小化 1 - cos_sim）
        return 1 - cos_sim.mean()
