import io
import png

import numpy as np
import timm
import torch
import torch.nn.functional as F
import torchvision
from torch import nn

from .MedViT import MedViT


__all__ = [
    "VinillaCNN",
    "DenseNet",
    "VGG",
    "EfficientNet",
    "ResNet",
    "LinearClassifier",
    "NonLinearClassifier",
    "VisionTransformer",
    "SwinTransformer",
    "MedVisiontransformer",
    "TimmModel",
]

class TimmModel(nn.Module):
    def __init__(self, model_name: str, num_classes, pretrained=True):
        super().__init__()
        self.network = timm.create_model(model_name, pretrained=True, num_classes=num_classes)
        # self.output = nn.Linear(
        #     in_features=self.network.embed_dim, out_features=num_classes
        # )
        # nn.init.kaiming_uniform_(self.output.weight)

    def forward(self, x):
        x = self.network(x)
        # x = self.output(x)
        return x
    
class SelfAttention(nn.Module):
    def __init__(self, cnn_dim=1407, trans_dim=768, num_heads=8):
        super().__init__()
        self.hidden_dim = trans_dim
        self.num_heads = num_heads

        self.cnn_proj = nn.Conv2d(cnn_dim, self.hidden_dim, kernel_size=1)
        self.trans_proj = nn.Conv2d(trans_dim, self.hidden_dim, kernel_size=1)

        self.norm1 = nn.LayerNorm(self.hidden_dim)

        self.attn = nn.MultiheadAttention(embed_dim=self.hidden_dim, num_heads=num_heads, batch_first=True)

        self.norm2 = nn.LayerNorm(self.hidden_dim)

    def forward(self, cnn_feat, transformer_feat):
        B, _, H, W = cnn_feat.shape

        cnn_feat_proj = self.cnn_proj(cnn_feat) # [B, T, H, W]
        trans_feat_proj = self.trans_proj(transformer_feat) # [B, T, H', W']
    
        cnn_tokens = cnn_feat_proj.flatten(2).transpose(1, 2)     # [B, H * W, T]
        trans_tokens = trans_feat_proj.flatten(2).transpose(1, 2) # [B, H' * W', T]

        tokens = torch.cat([cnn_tokens, trans_tokens], dim=1)

        tokens = self.norm1(tokens)

        # CNN attends to Transformer
        attn_out, _ = self.attn(query=tokens, key=tokens, value=tokens) # [B, H * W, T]

        tokens = self.norm2(tokens + attn_out) 

        fused = tokens[:, :cnn_tokens.size(1), :]  # [B, N1, C]
        fused = fused.transpose(1, 2).view(B, self.hidden_dim, H, W)

        return fused

class CrossAttention(nn.Module):
    def __init__(self, cnn_dim=1407, trans_dim=768, num_heads=8):
        super().__init__()
        self.hidden_dim = trans_dim
        self.num_heads = num_heads

        self.cnn_proj = nn.Conv2d(cnn_dim, self.hidden_dim, kernel_size=1)
        self.trans_proj = nn.Conv2d(trans_dim, self.hidden_dim, kernel_size=1)

        self.attn = nn.MultiheadAttention(embed_dim=self.hidden_dim, num_heads=num_heads, batch_first=True)

    def forward(self, cnn_feat, transformer_feat):
        B, _, H, W = cnn_feat.shape

        cnn_feat_proj = self.cnn_proj(cnn_feat) # [B, T, H, W]
        trans_feat_proj = self.trans_proj(transformer_feat) # [B, T, H', W']
    
        cnn_tokens = cnn_feat_proj.flatten(2).transpose(1, 2)     # [B, H * W, T]
        trans_tokens = trans_feat_proj.flatten(2).transpose(1, 2) # [B, H' * W', T]

        # CNN attends to Transformer
        attn_output, _ = self.attn(query=cnn_tokens, key=trans_tokens, value=trans_tokens) # [B, H * W, T]

        # residual connection
        fused = attn_output + cnn_tokens

        fused = fused.transpose(1, 2).view(B, self.hidden_dim, H, W) # [B, T, H, W]

        return fused

class Hybrid2Model(nn.Module):
    def __init__(self, num_classes, hidden_layer_sizes):
        super().__init__()
        self.transformer = timm.create_model("tiny_vit_21m_224.dist_in22k_ft_in1k", pretrained=True, num_classes=0)
        trans_prev_dim = 576
        self.cnn = getattr(torchvision.models, f"efficientnet_b0")(weights="DEFAULT")
        cnn_prev_dim = self.cnn.classifier[-1].in_features
        self.cnn.classifier = nn.Identity()
        # concat
        # prev_dim = transformer_prev_dim + cnn_prev_dim
        prev_dim = cnn_prev_dim
        self.layer_norm = nn.LayerNorm(prev_dim)
        layers = []
        dropout = 0.2
        self.attention = SelfAttention(cnn_dim=trans_prev_dim, trans_dim=cnn_prev_dim)
        # for size in hidden_layer_sizes:
        #     fc = nn.Linear(prev_dim, size)
        #     nn.init.kaiming_uniform_(fc.weight)
        #     layers.append(fc)
        #     layers.append(nn.ReLU())
        #     layers.append(nn.BatchNorm1d(size))
        #     layers.append(nn.Dropout(dropout))
        #     prev_dim = size
        # self.mlp = nn.Sequential(*layers) if layers else nn.Identity()
        # self.dropout = nn.Dropout(dropout)
        # Classification head
        self.classifier = nn.Sequential(
            
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            # nn.BatchNorm1d(prev_dim),
            # nn.Dropout(dropout),
            nn.Linear(prev_dim, 128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, num_classes)
        )

    def forward(self, x):
        # CNN forward
        cnn_feat = self.cnn.features(x)  # (B, 1408, 9, 9)
        # ViT forward
        trans_feat = self.transformer.forward_features(x)
        
        fused_feat = self.attention(trans_feat, cnn_feat)  # (B, 1)

        # MLP + output
        # x = self.mlp(fused_feat)
        return self.classifier(fused_feat)

class Hybrid1Model(nn.Module):
    def __init__(self, num_classes, hidden_layer_sizes):
        super().__init__()
        self.transformer = timm.create_model("tiny_vit_21m_224.dist_in22k_ft_in1k", pretrained=True, num_classes=0)
        trans_prev_dim = 576
        self.cnn = getattr(torchvision.models, f"efficientnet_b0")(weights="DEFAULT")
        cnn_prev_dim = self.cnn.classifier[-1].in_features
        self.cnn.classifier = nn.Identity()
        # concat
        # prev_dim = transformer_prev_dim + cnn_prev_dim
        prev_dim = cnn_prev_dim
        self.layer_norm = nn.LayerNorm(prev_dim)
        layers = []
        dropout = 0.2
        self.attention = CrossAttention(cnn_dim=trans_prev_dim, trans_dim=cnn_prev_dim)
        # for size in hidden_layer_sizes:
        #     fc = nn.Linear(prev_dim, size)
        #     nn.init.kaiming_uniform_(fc.weight)
        #     layers.append(fc)
        #     layers.append(nn.ReLU())
        #     layers.append(nn.BatchNorm1d(size))
        #     layers.append(nn.Dropout(dropout))
        #     prev_dim = size
        # self.mlp = nn.Sequential(*layers) if layers else nn.Identity()
        # self.dropout = nn.Dropout(dropout)
        # Classification head
        self.classifier = nn.Sequential(
            
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            # nn.BatchNorm1d(prev_dim),
            # nn.Dropout(dropout),
            nn.Linear(prev_dim, 128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, num_classes)
        )

    def forward(self, x):
        # CNN forward
        cnn_feat = self.cnn.features(x)  # (B, 1408, 9, 9)
        # ViT forward
        trans_feat = self.transformer.forward_features(x)
        
        fused_feat = self.attention(trans_feat, cnn_feat)  # (B, 1)

        # MLP + output
        # x = self.mlp(fused_feat)
        return self.classifier(fused_feat)

class HybridModel(nn.Module):
    def __init__(self, num_classes, hidden_layer_sizes):
        super().__init__()
        self.transformer = timm.create_model("tiny_vit_21m_224.dist_in22k_ft_in1k", pretrained=True, num_classes=0)
        trans_prev_dim = 576
        self.cnn = getattr(torchvision.models, f"efficientnet_b0")(weights="DEFAULT")
        cnn_prev_dim = self.cnn.classifier[-1].in_features
        self.cnn.classifier = nn.Identity()
        # concat
        # prev_dim = transformer_prev_dim + cnn_prev_dim
        prev_dim = trans_prev_dim
        self.layer_norm = nn.LayerNorm(prev_dim)
        layers = []
        dropout = 0.2
        self.attention = CrossAttention(cnn_dim=cnn_prev_dim, trans_dim=trans_prev_dim)
        # for size in hidden_layer_sizes:
        #     fc = nn.Linear(prev_dim, size)
        #     nn.init.kaiming_uniform_(fc.weight)
        #     layers.append(fc)
        #     layers.append(nn.ReLU())
        #     layers.append(nn.BatchNorm1d(size))
        #     layers.append(nn.Dropout(dropout))
        #     prev_dim = size
        # self.mlp = nn.Sequential(*layers) if layers else nn.Identity()
        # self.dropout = nn.Dropout(dropout)
        # Classification head
        self.classifier = nn.Sequential(
            
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            # nn.BatchNorm1d(prev_dim),
            # nn.Dropout(dropout),
            nn.Linear(prev_dim, 128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, num_classes)
        )

    def forward(self, x):
        # CNN forward
        cnn_feat = self.cnn.features(x)  # (B, 1408, 9, 9)
        # ViT forward
        trans_feat = self.transformer.forward_features(x)
        
        fused_feat = self.attention(cnn_feat, trans_feat)  # (B, 1)

        # MLP + output
        # x = self.mlp(fused_feat)
        return self.classifier(fused_feat)

class MedVisiontransformer(nn.Module):
    def __init__(self, num_classes, suffix: str = "small", weights="DEFAULT"):
        super().__init__()
        assert suffix in ["small", "base", "large"]
        # import pdb; pdb.set_trace()
        self.network = getattr(MedViT, f"MedViT_{suffix}")(num_classes=num_classes)

    def forward(self, x):
        x = self.network(x)
        return x

class LinearClassifier(nn.Module):
    def __init__(self, embedding_size, num_classes):
        super().__init__()
        self.token_num = 32
        self.embeddings_size = 768
        # self.dropout = nn.Dropout(0.5)
        # self.normalizer = nn.BatchNorm1d(in_features)
        self.fc = nn.Linear(in_features=self.embeddings_size, out_features=num_classes)
        nn.init.kaiming_uniform_(self.fc.weight)

    def forward(self, x):
        x = x.view(x.size(0), self.token_num, self.embeddings_size)
        x = x.mean(dim=1)
        x = self.fc(x)
        return x
    
class NonLinearClassifier(nn.Module):
    def __init__(self, hidden_layer_sizes, num_classes):
        super().__init__()
        self.token_num = 32
        self.embeddings_size = 768

        # hidden_layer_sizes = [1024, 512]
        layers = []
        dropout = 0.2
        prev_dim = self.embeddings_size
        for size in hidden_layer_sizes:
            fc = nn.Linear(prev_dim, size)
            nn.init.kaiming_uniform_(fc.weight)
            layers.append(fc)
            layers.append(nn.ReLU())
            layers.append(nn.BatchNorm1d(size))
            layers.append(nn.Dropout(dropout))
            prev_dim = size
        self.mlp = nn.Sequential(*layers) if layers else nn.Identity()
        self.output = nn.Linear(in_features=prev_dim, out_features=num_classes)
        nn.init.kaiming_uniform_(self.output.weight)

    def forward(self, x):
        x = x.view(x.size(0), self.token_num, self.embeddings_size)
        x = x.mean(dim=1)
        x = F.relu(x)
        x = self.mlp(x)
        x = self.output(x)
        return x

class SwinTransformer(nn.Module):
    def __init__(self, num_classes, hidden_layer_sizes, suffix: str = "b_16", weights="DEFAULT"):
        super().__init__()
        assert suffix in ["t", "s", "b", "v2_t", "v2_s", "v2_b"]
        self.network = getattr(torchvision.models, f"swin_{suffix}")(weights=weights)

        prev_dim = self.network.head.in_features
        self.network.head = nn.Identity()

        layers = []
        dropout = 0.2
        for size in hidden_layer_sizes:
            fc = nn.Linear(prev_dim, size)
            nn.init.kaiming_uniform_(fc.weight)
            layers.append(fc)
            layers.append(nn.ReLU())
            layers.append(nn.BatchNorm1d(size))
            layers.append(nn.Dropout(dropout))
            prev_dim = size
        self.mlp = nn.Sequential(*layers) if layers else nn.Identity()
        self.output = nn.Linear(in_features=prev_dim, out_features=num_classes)
        nn.init.kaiming_uniform_(self.output.weight)

    def forward(self, x):
        x = self.network(x)
        x = self.mlp(x)
        x = self.output(x)
        return x

class VisionTransformer(nn.Module):
    def __init__(self, num_classes, hidden_layer_sizes, suffix: str = "b_16", weights="DEFAULT"):
        super().__init__()
        assert suffix in ["b_16", "b_32"]
        self.network = getattr(torchvision.models, f"vit_{suffix}")(weights=weights)

        prev_dim = self.network.heads.head.in_features
        self.network.heads = nn.Identity()

        layers = []
        dropout = 0.2
        for size in hidden_layer_sizes:
            fc = nn.Linear(prev_dim, size)
            nn.init.kaiming_uniform_(fc.weight)
            layers.append(fc)
            layers.append(nn.ReLU())
            layers.append(nn.BatchNorm1d(size))
            layers.append(nn.Dropout(dropout))
            prev_dim = size
        self.mlp = nn.Sequential(*layers) if layers else nn.Identity()
        self.output = nn.Linear(in_features=prev_dim, out_features=num_classes)
        nn.init.kaiming_uniform_(self.output.weight)

    def forward(self, x):
        x = self.network(x)
        x = self.mlp(x)
        x = self.output(x)
        return x

class VinillaCNN(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.conv1 = nn.Conv2d(
            in_channels=3, out_channels=32, kernel_size=3, stride=1, padding=1
        )
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.conv2 = nn.Conv2d(
            in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=1
        )
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.conv3 = nn.Conv2d(
            in_channels=64, out_channels=128, kernel_size=3, stride=1, padding=1
        )
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.fc1 = nn.Linear(in_features=128 * 28 * 28, out_features=512)
        self.fc2 = nn.Linear(in_features=512, out_features=num_classes)

        self.dropout = nn.Dropout(0.5)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = self.pool1(x)

        x = F.relu(self.conv2(x))
        x = self.pool2(x)

        x = F.relu(self.conv3(x))
        x = self.pool3(x)

        x = x.view(-1, 128 * 28 * 28)

        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x


class DenseNet(nn.Module):
    def __init__(self, num_classes, hidden_layer_sizes, suffix: int | str = 121, weights="DEFAULT"):
        super().__init__()
        assert suffix in ["121", "161", "169", "201"]
        self.network = getattr(torchvision.models, f"densenet{suffix}")(weights=weights)

        prev_dim = self.network.classifier.in_features
        self.network.classifier = nn.Identity()

        layers = []
        dropout = 0.4
        for size in hidden_layer_sizes:
            fc = nn.Linear(prev_dim, size)
            nn.init.kaiming_uniform_(fc.weight)
            layers.append(fc)
            layers.append(nn.ReLU())
            layers.append(nn.BatchNorm1d(size))
            layers.append(nn.Dropout(dropout))
            prev_dim = size
        self.mlp = nn.Sequential(*layers) if layers else nn.Identity()
        self.dropout = nn.Dropout(dropout)
        self.output = nn.Linear(in_features=prev_dim, out_features=num_classes)
        nn.init.kaiming_uniform_(self.output.weight)

    def forward(self, x):
        x = self.network(x)
        x = self.mlp(x)
        x = self.dropout(x)
        x = self.output(x)
        return x


class VGG(nn.Module):
    def __init__(self, num_classes, hidden_layer_sizes, suffix: str = "11", weights="DEFAULT"):
        super().__init__()
        assert suffix in ["11", "11_bn", "13", "13_bn", "16", "16_bn", "19", "19_bn"]
        self.network = getattr(torchvision.models, f"vgg{suffix}")(weights=weights)
        fc = nn.Linear(
            self.network.classifier[-1].in_features, num_classes
        )
        nn.init.kaiming_uniform_(fc.weight)

        self.network.classifier[-1] = fc

    def forward(self, x):
        x = self.network(x)
        return x


class ResNet(nn.Module):
    def __init__(self, num_classes, hidden_layer_sizes, suffix: str = "18", weights="DEFAULT"):
        super().__init__()
        assert suffix in ["18", "34", "50", "101", "152"]
        self.network = getattr(torchvision.models, f"resnet{suffix}")(weights=weights)
        prev_dim = self.network.fc.in_features
        self.network.fc = nn.Identity()

        layers = []
        dropout = 0.2
        for size in hidden_layer_sizes:
            fc = nn.Linear(prev_dim, size)
            nn.init.kaiming_uniform_(fc.weight)
            layers.append(fc)
            layers.append(nn.ReLU())
            layers.append(nn.BatchNorm1d(size))
            layers.append(nn.Dropout(dropout))
            prev_dim = size
        self.mlp = nn.Sequential(*layers) if layers else nn.Identity()
        self.output = nn.Linear(in_features=prev_dim, out_features=num_classes)
        nn.init.kaiming_uniform_(self.output.weight)


    def forward(self, x):
        x = self.network(x)
        x = self.mlp(x)
        x = self.output(x)
        return x


class EfficientNet(nn.Module):
    def __init__(self, num_classes, hidden_layer_sizes, suffix: str = "b0", weights="DEFAULT"):
        super().__init__()
        assert suffix in ["b0", "b1", "b2", "b3", "b4", "b5", "b6", "b7"]
        self.network = getattr(torchvision.models, f"efficientnet_{suffix}")(
            weights=weights
        )
        prev_dim = self.network.classifier[-1].in_features
        self.network.classifier = nn.Identity()

        layers = []
        dropout = 0.2
        for size in hidden_layer_sizes:
            fc = nn.Linear(prev_dim, size)
            nn.init.kaiming_uniform_(fc.weight)
            layers.append(fc)
            layers.append(nn.ReLU())
            layers.append(nn.BatchNorm1d(size))
            layers.append(nn.Dropout(dropout))
            prev_dim = size
        self.mlp = nn.Sequential(*layers) if layers else nn.Identity()
        self.dropout = nn.Dropout(dropout)
        self.output = nn.Linear(in_features=prev_dim, out_features=num_classes)
        nn.init.kaiming_uniform_(self.output.weight)

    def forward(self, x):
        x = self.network(x)
        x = self.mlp(x)
        x = self.dropout(x)
        x = self.output(x)

        return x
    
class EfficientNetV2(nn.Module):
    def __init__(self, num_classes, hidden_layer_sizes, suffix: str = "m", weights="DEFAULT"):
        super().__init__()
        assert suffix in ["s", "m", "l"]
        self.network = getattr(torchvision.models, f"efficientnet_v2_{suffix}")(
            weights=weights
        )
        prev_dim = self.network.classifier[-1].in_features
        self.network.classifier = nn.Identity()

        layers = []
        dropout = 0.2
        for size in hidden_layer_sizes:
            fc = nn.Linear(prev_dim, size)
            nn.init.kaiming_uniform_(fc.weight)
            layers.append(fc)
            layers.append(nn.ReLU())
            layers.append(nn.BatchNorm1d(size))
            layers.append(nn.Dropout(dropout))
            prev_dim = size
        self.mlp = nn.Sequential(*layers) if layers else nn.Identity()
        self.output = nn.Linear(in_features=prev_dim, out_features=num_classes)
        nn.init.kaiming_uniform_(self.output.weight)

    def forward(self, x):
        x = self.network(x)
        x = self.mlp(x)
        x = self.output(x)

        return x
