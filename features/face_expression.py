import cv2
import numpy as np
from deepface import DeepFace
from config import EmotionConfig


class FaceExpressionRecognizer:
    def __init__(self):
        self.config = EmotionConfig()
        self.expression_mapping = {
            'happy': 'positive',
            'surprise': 'positive',
            'neutral': 'neutral',
            'sad': 'negative',
            'angry': 'negative',
            'fear': 'negative',
            'disgust': 'negative'
        }

    def analyze_expression(self, face_image):
        """使用DeepFace分析面部表情"""
        try:
            # 确保图像是BGR格式
            if len(face_image.shape) == 3:
                rgb_image = cv2.cvtColor(face_image, cv2.COLOR_BGR2RGB)
            else:
                rgb_image = cv2.cvtColor(face_image, cv2.COLOR_GRAY2RGB)

            # 分析表情
            analysis = DeepFace.analyze(rgb_image, actions=['emotion'], enforce_detection=False)

            if analysis and len(analysis) > 0:
                emotion = analysis[0]['dominant_emotion']
                confidence = analysis[0]['emotion'][emotion] / 100.0
                return emotion, confidence

        except Exception as e:
            print(f"表情分析错误: {e}")

        return 'neutral', 0.5

    def map_to_basic_emotion(self, expression, confidence):
        """将详细表情映射到基本情感"""
        return self.expression_mapping.get(expression, 'neutral'), confidence
