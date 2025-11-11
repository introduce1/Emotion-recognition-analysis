import os


class EmotionConfig:
    # 路径配置
    DATA_DIR = "./data"
    MODEL_DIR = "./models"
    OUTPUT_DIR = "./output"
    RAW_VIDEO_DIR = os.path.join(DATA_DIR, "raw")

    # 情感类别
    EMOTION_CLASSES = ['positive', 'negative', 'neutral']

    # 模型配置
    USE_AUDIO = True
    USE_VISUAL = True

    # 视觉模型配置
    FACE_DETECTION_PROTO = os.path.join(MODEL_DIR, "deploy.prototxt")
    FACE_DETECTION_MODEL = os.path.join(MODEL_DIR, "res10_300x300_ssd_iter_140000.caffemodel")

    # 特征提取配置
    FRAME_RATE = 1
    IMG_SIZE = (48, 48)
    FACE_CONFIDENCE = 0.5
    SAMPLE_RATE = 16000

    # 输出视频配置
    OUTPUT_VIDEO_SIZE = (640, 480)  # 输出视频尺寸
    FONT = 0  # OpenCV字体
    FONT_SCALE = 0.7
    FONT_THICKNESS = 2

    # 颜色配置
    COLORS = {
        'positive': (0, 255, 0),  # 绿色 - 积极
        'negative': (0, 0, 255),  # 红色 - 消极
        'neutral': (0, 255, 255),  # 黄色 - 中性
        'text': (255, 255, 255),  # 白色 - 文字
        'box': (255, 255, 255)  # 白色 - 边框
    }