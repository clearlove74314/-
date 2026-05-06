import torch
import torch.nn as nn
import torch.nn.functional as F

# ===================== SE通道注意力模块 =====================
class SEBlock(nn.Module):
    def __init__(self, channel, reduction=16):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1)
        return x * y.expand_as(x)

# ===================== 多尺度残差卷积块 =====================
class MultiScaleResBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv3 = nn.Conv1d(in_channels, out_channels//3, kernel_size=3, padding=1)
        self.conv5 = nn.Conv1d(in_channels, out_channels//3, kernel_size=5, padding=2)
        self.conv9 = nn.Conv1d(in_channels, out_channels//3 + (out_channels%3), kernel_size=9, padding=4)
        self.bn = nn.BatchNorm1d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.se = SEBlock(out_channels)
        self.downsample = nn.Conv1d(in_channels, out_channels, kernel_size=1) if in_channels != out_channels else nn.Identity()

    def forward(self, x):
        residual = self.downsample(x)
        x3 = self.conv3(x)
        x5 = self.conv5(x)
        x9 = self.conv9(x)
        out = torch.cat([x3, x5, x9], dim=1)
        out = self.bn(out)
        out = self.relu(out)
        out = self.se(out)
        out += residual
        out = self.relu(out)
        return out

# ===================== 特征流主干网络 =====================
class FeatureStream(nn.Module):
    def __init__(self, in_channels=1, out_dim=256, input_length=256):
        super().__init__()
        self.backbone = nn.Sequential(
            MultiScaleResBlock(in_channels, 64),
            MultiScaleResBlock(64, 128),
            MultiScaleResBlock(128, 256),
            nn.AdaptiveAvgPool1d(1)
        )
        self.fc = nn.Linear(256, out_dim)

    def forward(self, x):
        if len(x.shape) == 4:
            x = x.mean(dim=2)
        out = self.backbone(x)
        out = out.flatten(1)
        out = self.fc(out)
        return out

# ===================== 语义流模块 =====================
class SemanticStream(nn.Module):
    def __init__(self, num_classes=5, embed_dim=128, out_dim=256, label_smooth=0.05):
        super().__init__()
        self.label_smooth = label_smooth
        self.class_embedding = nn.Embedding(num_classes, embed_dim)
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, out_dim),
            nn.ReLU(inplace=True),
            nn.Linear(out_dim, out_dim)
        )

    def forward(self, labels):
        if self.training:
            one_hot = F.one_hot(labels, num_classes=self.class_embedding.num_embeddings).float()
            smooth_label = one_hot * (1 - self.label_smooth) + self.label_smooth / self.class_embedding.num_embeddings
            embed = torch.matmul(smooth_label, self.class_embedding.weight)
        else:
            embed = self.class_embedding(labels)
        out = self.mlp(embed)
        return out

# ===================== 交叉注意力交互模块 =====================
class CrossAttentionModule(nn.Module):
    def __init__(self, dim=256, num_heads=8, dropout=0.1):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5

        self.q_proj = nn.Linear(dim, dim)
        self.k_proj = nn.Linear(dim, dim)
        self.v_proj = nn.Linear(dim, dim)
        self.out_proj = nn.Linear(dim, dim)

        self.norm = nn.LayerNorm(dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, semantic_feat, feature_feat):
        residual = semantic_feat
        batch_size = semantic_feat.shape[0]

        q = self.q_proj(semantic_feat).view(batch_size, self.num_heads, 1, self.head_dim)
        k = self.k_proj(feature_feat).view(batch_size, self.num_heads, 1, self.head_dim)
        v = self.v_proj(feature_feat).view(batch_size, self.num_heads, 1, self.head_dim)

        attn_scores = (q @ k.transpose(-2, -1)) * self.scale
        attn_weights = F.softmax(attn_scores, dim=-1)
        attn_weights = self.dropout(attn_weights)

        attn_out = (attn_weights @ v).transpose(1, 2).contiguous().view(batch_size, self.dim)
        out = self.out_proj(attn_out)

        out = self.norm(out + residual)
        enhanced_feat = out + feature_feat
        return enhanced_feat

# ===================== 余弦β调度与扩散核心模块 =====================
class DiffusionDecoder(nn.Module):
    def __init__(self, timesteps=200, feature_dim=256, signal_length=1024):
        super().__init__()
        self.timesteps = timesteps
        self.feature_dim = feature_dim

        self.betas = self.get_cosine_beta_schedule(timesteps)
        self.alphas = 1. - self.betas
        self.alphas_cumprod = torch.cumprod(self.alphas, dim=0)
        self.sqrt_alphas_cumprod = torch.sqrt(self.alphas_cumprod)
        self.sqrt_one_minus_alphas_cumprod = torch.sqrt(1. - self.alphas_cumprod)

        self.denoise_net = nn.Sequential(
            nn.Linear(feature_dim + 1, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.1),
            nn.Linear(512, 1024),
            nn.ReLU(inplace=True),
            nn.Linear(1024, signal_length)
        )

    def get_cosine_beta_schedule(self, timesteps, s=0.008):
        steps = timesteps + 1
        x = torch.linspace(0, timesteps, steps)
        alphas_cumprod = torch.cos(((x / timesteps) + s) / (1 + s) * torch.pi * 0.5) ** 2
        alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
        betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
        return torch.clip(betas, 0.0001, 0.9999)

    def forward_add_noise(self, x_start, t, noise=None):
        if noise is None:
            noise = torch.randn_like(x_start)
        sqrt_alphas_cumprod_t = self.sqrt_alphas_cumprod[t].view(-1, 1).to(x_start.device)
        sqrt_one_minus_alphas_cumprod_t = self.sqrt_one_minus_alphas_cumprod[t].view(-1, 1).to(x_start.device)
        x_noisy = sqrt_alphas_cumprod_t * x_start + sqrt_one_minus_alphas_cumprod_t * noise
        return x_noisy, noise

    def reverse_denoise(self, enhanced_feat, device):
        x = torch.randn(enhanced_feat.shape[0], 1024).to(device)
        for t in reversed(range(0, self.timesteps)):
            t_tensor = torch.full((enhanced_feat.shape[0],), t, dtype=torch.long).to(device)
            model_input = torch.cat([enhanced_feat, t_tensor.unsqueeze(1).float()], dim=1)
            pred_noise = self.denoise_net(model_input)

            beta_t = self.betas[t].to(device)
            alpha_t = self.alphas[t].to(device)
            alpha_cumprod_t = self.alphas_cumprod[t].to(device)

            if t > 0:
                noise = torch.randn_like(x)
            else:
                noise = torch.zeros_like(x)

            x = 1 / torch.sqrt(alpha_t) * (
                x - ((1 - alpha_t) / torch.sqrt(1 - alpha_cumprod_t)) * pred_noise
            ) + torch.sqrt(beta_t) * noise
        return x

# ===================== 完整DSDM模型整合 =====================
class DSDM(nn.Module):
    def __init__(self, num_classes=5, timesteps=200):
        super().__init__()
        self.feature_stream = FeatureStream()
        self.semantic_stream = SemanticStream(num_classes=num_classes)
        self.cross_attention = CrossAttentionModule()
        self.diffusion_decoder = DiffusionDecoder(timesteps=timesteps)

    def forward(self, x, labels, t):
        feature_feat = self.feature_stream(x)
        semantic_feat = self.semantic_stream(labels)
        enhanced_feat = self.cross_attention(semantic_feat, feature_feat)

        x_start = x.mean(dim=2).flatten(1)
        x_noisy, noise = self.diffusion_decoder.forward_add_noise(x_start, t)

        model_input = torch.cat([enhanced_feat, t.unsqueeze(1).float()], dim=1)
        pred_noise = self.diffusion_decoder.denoise_net(model_input)

        return pred_noise, noise, enhanced_feat

    def generate_samples(self, x, labels, device):
        feature_feat = self.feature_stream(x)
        semantic_feat = self.semantic_stream(labels)
        enhanced_feat = self.cross_attention(semantic_feat, feature_feat)
        generated_samples = self.diffusion_decoder.reverse_denoise(enhanced_feat, device)
        return generated_samples

# 训练配置
TRAIN_CONFIG = {
    "epochs": 300,
    "batch_size": 32,
    "lr": 1e-4,
    "weight_decay": 1e-2,
    "timesteps": 200,
    "num_classes": 5,
    "device": "cuda" if torch.cuda.is_available() else "cpu"
}

# 损失函数
def dsdm_loss(pred_noise, real_noise, enhanced_feat, labels, cls_weight=0.1):
    rec_loss = F.mse_loss(pred_noise, real_noise)
    cls_head = nn.Linear(enhanced_feat.shape[1], labels.max().item()+1).to(enhanced_feat.device)
    cls_pred = cls_head(enhanced_feat)
    cls_loss = F.cross_entropy(cls_pred, labels)
    total_loss = rec_loss + cls_weight * cls_loss
    return total_loss, rec_loss, cls_loss
