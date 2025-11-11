import os
import urllib.request
import zipfile
import tarfile


def download_models():
    model_dir = "models"
    os.makedirs(model_dir, exist_ok=True)

    print("开始下载预训练模型...")

    # 1. 人脸检测模型 (OpenCV)
    face_proto = os.path.join(model_dir, "deploy.prototxt")
    face_model = os.path.join(model_dir, "res10_300x300_ssd_iter_140000.caffemodel")

    if not os.path.exists(face_proto):
        print("下载人脸检测配置文件...")
        urllib.request.urlretrieve(
            "https://raw.githubusercontent.com/opencv/opencv/master/samples/dnn/face_detector/deploy.prototxt",
            face_proto
        )

    if not os.path.exists(face_model):
        print("下载人脸检测模型...")
        urllib.request.urlretrieve(
            "https://raw.githubusercontent.com/opencv/opencv_3rdparty/dnn_samples_face_detector_20170830/res10_300x300_ssd_iter_140000.caffemodel",
            face_model
        )

    # 2. 使用公开的TensorFlow Hub模型（不需要下载）
    print("将使用TensorFlow Hub的预训练模型")

    # 3. 下载简单的音频特征提取模型（如果需要）
    print("模型下载完成！")


if __name__ == "__main__":
    download_models()