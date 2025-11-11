# 视频情感识别分析系统

一个基于多模态融合的视频情感分析系统，能够从视频中提取视觉和音频特征，进行实时情感识别和标注。

## 功能特性

- **多模态情感分析**: 结合视觉（面部表情）和音频（语音特征）进行综合分析
- **实时视频标注**: 自动生成带有情感标注的视频文件
- **详细分析报告**: 生成包含情感统计和主要情感的分析报告
- **支持多种情感**: 识别积极、消极、中性三种基本情感
- **可视化界面**: 彩色边框标注不同情感（绿色-积极，红色-消极，黄色-中性）

## 系统架构

```
视频输入 → 特征提取 → 情感分析 → 结果输出
    │           │           │          │
   视频文件     视觉特征     多模态融合    标注视频
               音频特征     情感预测     分析报告
```

## 安装依赖

```bash
pip install -r requirements.txt
```

### 主要依赖库

- `opencv-python>=4.5.0` - 计算机视觉处理
- `numpy>=1.21.0` - 数值计算
- `librosa>=0.8.0` - 音频特征提取
- `tensorflow>=2.8.0` - 深度学习框架
- `deepface>=0.0.95` - 面部表情分析
- `ffmpeg-python>=0.2.0` - 视频/音频处理

## 快速开始

### 1. 下载预训练模型

```bash
python download.py
```

### 2. 运行情感分析

```bash
python emotion_predict.py --video_path "your_video.mp4"
```

### 3. 查看输出结果

- 标注视频: `output/your_video_emotion.avi`
- 分析报告: `output/your_video_report.txt`

## 使用方法

### 命令行参数

```bash
python emotion_predict.py --video_path <视频文件路径>
```

### 输出说明

1. **标注视频**: 包含彩色情感边框和实时统计信息的视频文件
2. **分析报告**: 包含详细情感统计和主要情感分析的文本报告

## 项目结构

```
emotion-recognition-analysis/
├── config.py              # 配置文件
├── emotion_predict.py     # 主程序
├── download.py            # 模型下载脚本
├── requirements.txt       # 依赖文件
├── models/                # 模型文件目录
├── output/                # 输出文件目录
├── features/              # 特征提取模块
│   ├── audio_extractor.py    # 音频特征提取
│   ├── visual_extractor.py   # 视觉特征提取
│   └── face_expression.py    # 面部表情识别
└── data/                  # 数据目录
```

## 技术细节

### 视觉特征提取

- 使用OpenCV DNN模块进行人脸检测
- 采用DeepFace进行面部表情分析
- 支持多种人脸检测模型（Caffe模型 + Haar级联分类器）

### 音频特征提取

- 使用Librosa提取MFCC、色度、频谱等特征
- 基于音频特征进行情感倾向分析

### 多模态融合

- 视觉和音频特征的加权融合
- 自适应权重调整机制

## 配置选项

在 `config.py` 中可以调整以下参数：

- 情感类别定义
- 模型文件路径
- 特征提取参数
- 输出视频配置
- 颜色标注方案

## 性能优化

- 支持帧采样率调整
- 可选禁用音频/视觉分析
- 内存优化处理

## 常见问题

### Q: 模型下载失败怎么办？
A: 可以手动下载模型文件到 `models/` 目录：
- deploy.prototxt
- res10_300x300_ssd_iter_140000.caffemodel

### Q: 音频提取失败怎么办？
A: 确保系统已安装FFmpeg，或使用预提取的音频文件

### Q: 人脸检测效果不佳？
A: 尝试调整 `FACE_CONFIDENCE` 参数或使用不同的人脸检测模型

## 开发说明

### 扩展新的情感类别

1. 在 `config.py` 中修改 `EMOTION_CLASSES`
2. 更新 `face_expression.py` 中的表情映射
3. 调整颜色配置和输出格式

### 添加新的特征提取器

1. 在 `features/` 目录下创建新的提取器类
2. 实现特征提取接口
3. 在主程序中集成新的特征源

## 许可证

本项目基于MIT许可证开源。

