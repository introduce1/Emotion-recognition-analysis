import os
import argparse
import numpy as np
from features.visual_extractor import VisualFeatureExtractor
from features.audio_extractor import AudioFeatureExtractor
from config import EmotionConfig
import warnings

warnings.filterwarnings("ignore")


class VideoEmotionPredictor:
    def __init__(self):
        self.config = EmotionConfig()

        # 创建输出目录
        os.makedirs(self.config.OUTPUT_DIR, exist_ok=True)

        # 初始化特征提取器
        print("初始化视觉特征提取器...")
        self.visual_extractor = VisualFeatureExtractor()
        print("初始化音频特征提取器...")
        self.audio_extractor = AudioFeatureExtractor()

    def extract_audio_from_video(self, video_path, audio_path):
        """从视频中提取音频"""
        try:
            import subprocess
            print(f"从视频提取音频到: {audio_path}")
            command = f'ffmpeg -i "{video_path}" -ab 160k -ac 2 -ar 16000 -vn "{audio_path}" -y -loglevel quiet'
            result = subprocess.run(command, shell=True, capture_output=True, text=True, timeout=60)

            if result.returncode == 0:
                if os.path.exists(audio_path) and os.path.getsize(audio_path) > 0:
                    print("音频提取成功")
                    return True
                else:
                    print("音频文件为空或不存在")
            else:
                print(f"音频提取失败: {result.stderr}")

            return False

        except subprocess.TimeoutExpired:
            print("音频提取超时")
            return False
        except Exception as e:
            print(f"音频提取错误: {e}")
            return False

    def weighted_fusion(self, audio_features, visual_features):
        """加权融合特征"""
        audio_probs = audio_features.flatten()
        visual_probs = visual_features.flatten()

        print(f"音频情感概率: {audio_probs}")
        print(f"视觉情感概率: {visual_probs}")

        # 根据置信度调整权重
        audio_confidence = np.max(audio_probs) - np.min(audio_probs)
        visual_confidence = np.max(visual_probs) - np.min(visual_probs)

        if audio_confidence > 0.3 and visual_confidence > 0.3:
            audio_weight = 0.4
            visual_weight = 0.6
        elif audio_confidence > 0.3:
            audio_weight = 0.7
            visual_weight = 0.3
        elif visual_confidence > 0.3:
            audio_weight = 0.3
            visual_weight = 0.7
        else:
            audio_weight = 0.5
            visual_weight = 0.5

        print(f"权重分配 - 音频: {audio_weight}, 视觉: {visual_weight}")

        # 加权融合
        fused_probs = audio_weight * audio_probs + visual_weight * visual_probs
        fused_probs = fused_probs / np.sum(fused_probs)

        return fused_probs

    def generate_output_filename(self, input_path):
        """生成输出文件名"""
        base_name = os.path.basename(input_path)
        name_without_ext = os.path.splitext(base_name)[0]
        output_video = os.path.join(self.config.OUTPUT_DIR, f"{name_without_ext}_emotion.avi")
        output_report = os.path.join(self.config.OUTPUT_DIR, f"{name_without_ext}_report.txt")
        return output_video, output_report

    def save_report(self, report_path, results, video_path):
        """保存分析报告"""
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write("视频情感分析报告\n")
            f.write("=" * 50 + "\n")
            f.write(f"视频文件: {video_path}\n")
            f.write(f"分析时间: {np.datetime64('now')}\n")
            f.write("\n情感分析结果:\n")
            f.write("-" * 30 + "\n")
            for emotion, score in results.items():
                f.write(f"{emotion:8}: {score:.4f}\n")

            # 输出主要情感
            main_emotion = max(results.items(), key=lambda x: x[1])
            f.write(f"\n主要情感: {main_emotion[0]} ({main_emotion[1]:.2%})\n")

            f.write("\n颜色说明:\n")
            f.write("- 绿色: 积极情感\n")
            f.write("- 红色: 消极情感\n")
            f.write("- 黄色: 中性情感\n")

    def predict(self, video_path):
        """预测视频情感并生成标注视频"""
        if not os.path.exists(video_path):
            print(f"错误: 视频文件 {video_path} 不存在")
            return {"positive": 0.33, "negative": 0.33, "neutral": 0.34}

        # 生成输出文件名
        output_video, output_report = self.generate_output_filename(video_path)

        # 创建临时目录
        os.makedirs("temp", exist_ok=True)
        audio_path = os.path.join("temp", "extracted_audio.wav")

        print("\n=== 开始视频情感分析 ===")

        # 提取和分析视觉特征（同时生成标注视频）
        print("\n1. 分析视觉特征并生成标注视频...")
        visual_features = self.visual_extractor.extract_visual_emotion(video_path, output_video)

        # 提取和分析音频特征
        audio_features = np.array([[0.33, 0.33, 0.34]])
        print("\n2. 提取和分析音频特征...")
        audio_extracted = self.extract_audio_from_video(video_path, audio_path)

        if audio_extracted and os.path.exists(audio_path):
            audio_features = self.audio_extractor.extract_audio_features(audio_path)
            # 清理临时文件
            try:
                os.remove(audio_path)
                os.rmdir("temp")
            except:
                pass
        else:
            print("使用默认音频特征")

        # 特征融合
        print("\n3. 融合多模态特征...")
        final_probs = self.weighted_fusion(audio_features, visual_features)

        # 转换为结果字典
        results = {}
        for i, emotion in enumerate(self.config.EMOTION_CLASSES):
            results[emotion] = float(final_probs[i])

        # 保存分析报告
        self.save_report(output_report, results, video_path)
        print(f"分析报告已保存到: {output_report}")

        return results, output_video


def main():
    parser = argparse.ArgumentParser(description="视频情感预测")
    parser.add_argument("--video_path", type=str, required=True, help="输入视频路径")

    args = parser.parse_args()

    print("初始化情感分析器...")
    predictor = VideoEmotionPredictor()

    print("开始分析视频情感...")
    results, output_video = predictor.predict(args.video_path)

    print("\n=== 分析结果 ===")
    print("-" * 30)
    for emotion, score in results.items():
        print(f"{emotion:8}: {score:.4f}")

    # 输出主要情感
    main_emotion = max(results.items(), key=lambda x: x[1])
    print(f"\n主要情感: {main_emotion[0]} ({main_emotion[1]:.2%})")

    print(f"\n标注视频已生成: {output_video}")
    print("颜色说明:")
    print("- 绿色边框: 积极情感")
    print("- 红色边框: 消极情感")
    print("- 黄色边框: 中性情感")


if __name__ == "__main__":
    main()