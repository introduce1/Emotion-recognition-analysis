import librosa
import numpy as np
import os
from config import EmotionConfig


class AudioFeatureExtractor:
    def __init__(self):
        self.config = EmotionConfig()

    def extract_audio_features(self, audio_path):
        """提取音频特征并预测情感 - 改进版本"""
        try:
            print(f"分析音频文件: {audio_path}")
            y, sr = librosa.load(audio_path, sr=self.config.SAMPLE_RATE)

            # 提取多种音频特征
            features = []

            # 1. MFCC特征 (13个系数)
            mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
            features.extend(np.mean(mfcc, axis=1))  # 均值
            features.extend(np.std(mfcc, axis=1))  # 标准差

            # 2. 色度特征 (12个音级)
            chroma = librosa.feature.chroma_stft(y=y, sr=sr)
            features.extend(np.mean(chroma, axis=1))

            # 3. 频谱特征
            spectral_centroid = librosa.feature.spectral_centroid(y=y, sr=sr)
            features.append(np.mean(spectral_centroid))
            features.append(np.std(spectral_centroid))

            spectral_bandwidth = librosa.feature.spectral_bandwidth(y=y, sr=sr)
            features.append(np.mean(spectral_bandwidth))

            # 4. 频谱对比度 (7个频带)
            spectral_contrast = librosa.feature.spectral_contrast(y=y, sr=sr)
            features.extend(np.mean(spectral_contrast, axis=1))

            # 5. 节奏特征
            tempo, _ = librosa.beat.beat_track(y=y, sr=sr)
            features.append(tempo)

            # 6. 零交叉率
            zero_crossing_rate = librosa.feature.zero_crossing_rate(y)
            features.append(np.mean(zero_crossing_rate))

            # 7. 能量特征
            rms = librosa.feature.rms(y=y)
            features.append(np.mean(rms))
            features.append(np.std(rms))

            print(f"提取了 {len(features)} 个音频特征")

            # 基于特征的情感预测
            emotion_probs = self.predict_emotion_from_audio(features)
            return np.array([emotion_probs])

        except Exception as e:
            print(f"音频特征提取错误: {e}")
            return np.array([[0.33, 0.33, 0.34]])

    def predict_emotion_from_audio(self, features):
        """从音频特征预测情感 - 改进版本"""
        if len(features) < 20:
            return [0.33, 0.33, 0.34]

        positive_score = 0
        negative_score = 0

        # 提取关键特征
        spectral_centroid = features[38] if len(features) > 38 else 1000  # 频谱质心
        tempo = features[51] if len(features) > 51 else 120  # 节奏
        energy = features[53] if len(features) > 53 else 0.01  # 能量

        # 高频声音通常更积极
        if spectral_centroid > 2000:
            positive_score += 0.3
        elif spectral_centroid < 800:
            negative_score += 0.3

        # 快节奏通常更积极
        if tempo > 140:
            positive_score += 0.2
        elif tempo < 80:
            negative_score += 0.2

        # 高能量通常更积极
        if energy > 0.05:
            positive_score += 0.1
        elif energy < 0.01:
            negative_score += 0.1

        # 确保分数合理
        positive_score = min(positive_score, 0.6)
        negative_score = min(negative_score, 0.6)

        # 计算中性分数
        neutral_score = 1.0 - positive_score - negative_score
        if neutral_score < 0:
            # 重新归一化
            total = positive_score + negative_score
            positive_score /= total
            negative_score /= total
            neutral_score = 0

        return [positive_score, negative_score, neutral_score]