import torch
import torch.nn as nn
from transformers import AutoModel
from config import EmotionConfig


class MultimodalFusionModel(nn.Module):
    def __init__(self, num_classes=3):
        super(MultimodalFusionModel, self).__init__()
        self.config = EmotionConfig()
        self.num_classes = num_classes

        # 音频特征维度
        self.audio_feature_dim = 7  # 根据预训练模型的输出调整

        # 文本特征维度
        self.text_feature_dim = 7  # 根据预训练模型的输出调整

        # 视觉特征维度
        self.visual_feature_dim = 7  # 根据预训练模型的输出调整

        # 融合策略
        if self.config.FUSION_STRATEGY == "weighted_average":
            self.audio_weight = nn.Parameter(torch.tensor(0.33))
            self.text_weight = nn.Parameter(torch.tensor(0.33))
            self.visual_weight = nn.Parameter(torch.tensor(0.34))
        elif self.config.FUSION_STRATEGY == "concatenation":
            self.fc = nn.Linear(
                self.audio_feature_dim + self.text_feature_dim + self.visual_feature_dim,
                self.num_classes
            )
        elif self.config.FUSION_STRATEGY == "attention":
            self.attention = nn.MultiheadAttention(
                embed_dim=self.audio_feature_dim,
                num_heads=1,
                batch_first=True
            )
            self.fc = nn.Linear(self.audio_feature_dim, self.num_classes)

        self.sigmoid = nn.Sigmoid()

    def forward(self, audio_features, text_features, visual_features):
        if self.config.FUSION_STRATEGY == "weighted_average":
            # 加权平均融合
            fused_features = (
                    self.audio_weight * audio_features +
                    self.text_weight * text_features +
                    self.visual_weight * visual_features
            )
            output = self.sigmoid(fused_features)

        elif self.config.FUSION_STRATEGY == "concatenation":
            # 拼接融合
            fused_features = torch.cat([audio_features, text_features, visual_features], dim=1)
            output = self.sigmoid(self.fc(fused_features))

        elif self.config.FUSION_STRATEGY == "attention":
            # 注意力融合
            # 将特征转换为序列形式 [batch_size, 3, feature_dim]
            features = torch.stack([audio_features, text_features, visual_features], dim=1)
            attn_output, _ = self.attention(features, features, features)
            # 取平均作为最终特征
            fused_features = torch.mean(attn_output, dim=1)
            output = self.sigmoid(self.fc(fused_features))

        return output