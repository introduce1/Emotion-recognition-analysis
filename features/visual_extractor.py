import cv2
import numpy as np
import os
from config import EmotionConfig
from features.face_expression import FaceExpressionRecognizer


class VisualFeatureExtractor:
    def __init__(self):
        self.config = EmotionConfig()
        self.face_net = None
        self.face_cascade = None
        self.expression_recognizer = FaceExpressionRecognizer()

        self.load_models()

    def load_models(self):
        """加载预训练模型"""
        # 加载人脸检测模型
        if os.path.exists(self.config.FACE_DETECTION_PROTO) and os.path.exists(self.config.FACE_DETECTION_MODEL):
            try:
                self.face_net = cv2.dnn.readNetFromCaffe(
                    self.config.FACE_DETECTION_PROTO,
                    self.config.FACE_DETECTION_MODEL
                )
                print("人脸检测模型加载成功")
            except Exception as e:
                print(f"人脸检测模型加载失败: {e}")
                self.face_net = None

        # 加载Haar级联分类器作为备用
        try:
            cascade_path = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
            if os.path.exists(cascade_path):
                self.face_cascade = cv2.CascadeClassifier(cascade_path)
                print("Haar级联分类器加载成功")
            else:
                print("Haar级联分类器未找到")
        except Exception as e:
            print(f"Haar级联分类器加载失败: {e}")

    def detect_faces(self, frame):
        """检测人脸并返回人脸位置和置信度"""
        faces = []
        confidences = []

        # 首先尝试使用DNN人脸检测
        if self.face_net is not None:
            try:
                h, w = frame.shape[:2]
                blob = cv2.dnn.blobFromImage(frame, 1.0, (300, 300), [104, 117, 123], False, False)

                self.face_net.setInput(blob)
                detections = self.face_net.forward()

                for i in range(detections.shape[2]):
                    confidence = detections[0, 0, i, 2]
                    if confidence > self.config.FACE_CONFIDENCE:
                        x1 = int(detections[0, 0, i, 3] * w)
                        y1 = int(detections[0, 0, i, 4] * h)
                        x2 = int(detections[0, 0, i, 5] * w)
                        y2 = int(detections[0, 0, i, 6] * h)

                        x1, y1 = max(0, x1), max(0, y1)
                        x2, y2 = min(w, x2), min(h, y2)

                        if x2 > x1 and y2 > y1:
                            faces.append((x1, y1, x2, y2))
                            confidences.append(confidence)
            except Exception as e:
                print(f"DNN人脸检测失败: {e}")

        # 如果DNN没有检测到人脸或失败，使用Haar级联分类器
        if not faces and self.face_cascade is not None:
            try:
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                haar_faces = self.face_cascade.detectMultiScale(gray, 1.1, 4)
                for (x, y, w, h) in haar_faces:
                    faces.append((x, y, x + w, y + h))
                    confidences.append(0.6)
            except Exception as e:
                print(f"Haar人脸检测失败: {e}")

        return faces, confidences

    def process_frame(self, frame):
        """处理单帧图像，返回标注后的帧和情感信息"""
        # 检测人脸
        faces, confidences = self.detect_faces(frame)

        frame_emotions = []

        # 处理每个检测到的人脸
        for i, (x1, y1, x2, y2) in enumerate(faces):
            face_roi = frame[y1:y2, x1:x2]
            if face_roi.size > 0 and face_roi.shape[0] > 20 and face_roi.shape[1] > 20:
                try:
                    # 使用DeepFace分析表情
                    expression, expr_confidence = self.expression_recognizer.analyze_expression(face_roi)
                    emotion, confidence = self.expression_recognizer.map_to_basic_emotion(expression, expr_confidence)
                    frame_emotions.append((emotion, confidence))

                    # 绘制人脸框
                    color = self.config.COLORS[emotion]
                    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

                    # 绘制情感标签
                    label = f"{emotion}: {confidence:.2f}"
                    cv2.putText(frame, label, (x1, y1 - 10),
                                self.config.FONT, self.config.FONT_SCALE, color, self.config.FONT_THICKNESS)

                    # 添加详细表情信息
                    detail_label = f"expr: {expression}"
                    cv2.putText(frame, detail_label, (x1, y1 - 30),
                                self.config.FONT, self.config.FONT_SCALE * 0.7, color, 1)

                except Exception as e:
                    print(f"人脸表情分析失败: {e}")
                    # 使用备用方法
                    emotion, confidence = self.fallback_emotion_detection(face_roi)
                    frame_emotions.append((emotion, confidence))

        return frame, frame_emotions

    def fallback_emotion_detection(self, face_roi):
        """备用情感检测方法"""
        try:
            # 简单的基于嘴部特征的情感检测
            gray = cv2.cvtColor(face_roi, cv2.COLOR_BGR2GRAY)

            # 检测嘴部区域（简单假设嘴部在下半部分）
            height, width = gray.shape
            mouth_region = gray[int(height * 0.6):, :]

            # 计算嘴部区域的对比度（大笑时嘴部张开，对比度增加）
            mouth_contrast = np.std(mouth_region)

            if mouth_contrast > 40:  # 高对比度可能表示大笑
                return "positive", 0.7
            else:
                return "neutral", 0.5

        except:
            return "neutral", 0.5

    def extract_visual_emotion(self, video_path, output_path=None):
        """提取视觉情感特征并生成标注视频"""
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            print(f"无法打开视频文件: {video_path}")
            return np.array([[0.33, 0.33, 0.34]])

        # 获取视频信息
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        # 设置输出视频
        if output_path:
            fourcc = cv2.VideoWriter_fourcc(*'XVID')
            out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

        frame_count = 0
        emotion_stats = {'positive': 0, 'negative': 0, 'neutral': 0}
        total_faces = 0

        print(f"视频信息: {total_frames} 帧, {fps} FPS, 尺寸: {width}x{height}")

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            # 处理当前帧
            processed_frame, frame_emotions = self.process_frame(frame)

            # 统计情感
            for emotion, confidence in frame_emotions:
                emotion_stats[emotion] += confidence
                total_faces += 1

            # 在帧上添加全局信息
            cv2.putText(processed_frame, f"Frame: {frame_count}/{total_frames}",
                        (10, 30), self.config.FONT, self.config.FONT_SCALE,
                        self.config.COLORS['text'], self.config.FONT_THICKNESS)

            # 添加统计信息
            if total_faces > 0:
                stats_text = f"Pos: {emotion_stats['positive'] / total_faces:.2f} | Neg: {emotion_stats['negative'] / total_faces:.2f} | Neu: {emotion_stats['neutral'] / total_faces:.2f}"
                cv2.putText(processed_frame, stats_text,
                            (10, 60), self.config.FONT, self.config.FONT_SCALE * 0.7,
                            self.config.COLORS['text'], 1)

            # 写入输出视频
            if output_path:
                out.write(processed_frame)

            frame_count += 1

            # 显示进度
            if frame_count % 50 == 0:
                print(f"已处理 {frame_count}/{total_frames} 帧")
                if total_faces > 0:
                    print(
                        f"当前统计: 积极 {emotion_stats['positive'] / total_faces:.2f}, 消极 {emotion_stats['negative'] / total_faces:.2f}, 中性 {emotion_stats['neutral'] / total_faces:.2f}")

        cap.release()
        if output_path:
            out.release()
            print(f"标注视频已保存到: {output_path}")

        # 计算整体情感概率
        if total_faces > 0:
            positive_prob = emotion_stats['positive'] / total_faces
            negative_prob = emotion_stats['negative'] / total_faces
            neutral_prob = emotion_stats['neutral'] / total_faces

            # 归一化
            total = positive_prob + negative_prob + neutral_prob
            if total > 0:
                positive_prob /= total
                negative_prob /= total
                neutral_prob /= total
            else:
                positive_prob, negative_prob, neutral_prob = 0.33, 0.33, 0.34

            return np.array([[positive_prob, negative_prob, neutral_prob]])

        return np.array([[0.33, 0.33, 0.34]])