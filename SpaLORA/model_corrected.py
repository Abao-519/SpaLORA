"""SpaLORA model with explicit, per-observation modality attention."""

import torch
import torch.nn as nn


class GraphLinear(nn.Module):
    def __init__(self, in_features: int, out_features: int):
        super().__init__()
        self.weight = nn.Parameter(torch.empty(in_features, out_features))
        nn.init.xavier_uniform_(self.weight)

    def forward(self, features: torch.Tensor, adjacency: torch.Tensor) -> torch.Tensor:
        return torch.sparse.mm(adjacency, torch.mm(features, self.weight))


class AttentionCorrected(nn.Module):
    def __init__(self, in_features: int, out_features: int):
        super().__init__()
        self.w_omega = nn.Parameter(torch.empty(in_features, out_features))
        self.u_omega = nn.Parameter(torch.empty(out_features, 1))
        nn.init.xavier_uniform_(self.w_omega)
        nn.init.xavier_uniform_(self.u_omega)

    def forward(self, first: torch.Tensor, second: torch.Tensor):
        stacked = torch.stack((first, second), dim=1)
        scores = torch.matmul(torch.tanh(torch.matmul(stacked, self.w_omega)), self.u_omega).squeeze(-1)
        alpha = torch.softmax(scores, dim=1)
        combined = torch.sum(stacked * alpha.unsqueeze(-1), dim=1)
        return combined, alpha


class EncoderOverallCorrected(nn.Module):
    def __init__(self, in1: int, out1: int, in2: int, out2: int):
        super().__init__()
        self.encoder1 = GraphLinear(in1, out1)
        self.decoder1 = GraphLinear(out1, in1)
        self.encoder2 = GraphLinear(in2, out2)
        self.decoder2 = GraphLinear(out2, in2)
        self.attention1 = AttentionCorrected(out1, out1)
        self.attention2 = AttentionCorrected(out2, out2)
        self.cross_attention = AttentionCorrected(out1, out2)

    def forward(self, features1, features2, spatial1, feature1, spatial2, feature2):
        latent_spatial1 = self.encoder1(features1, spatial1)
        latent_spatial2 = self.encoder2(features2, spatial2)
        latent_feature1 = self.encoder1(features1, feature1)
        latent_feature2 = self.encoder2(features2, feature2)
        latent1, alpha1 = self.attention1(latent_spatial1, latent_feature1)
        latent2, alpha2 = self.attention2(latent_spatial2, latent_feature2)
        combined, alpha = self.cross_attention(latent1, latent2)
        recon1 = self.decoder1(combined, spatial1)
        recon2 = self.decoder2(combined, spatial2)
        latent1_across = self.encoder2(self.decoder2(latent1, spatial2), spatial2)
        latent2_across = self.encoder1(self.decoder1(latent2, spatial1), spatial1)
        return {
            "emb_latent_omics1": latent1,
            "emb_latent_omics2": latent2,
            "emb_latent_combined": combined,
            "emb_recon_omics1": recon1,
            "emb_recon_omics2": recon2,
            "emb_latent_omics1_across_recon": latent1_across,
            "emb_latent_omics2_across_recon": latent2_across,
            "alpha_omics1": alpha1,
            "alpha_omics2": alpha2,
            "alpha": alpha,
        }
