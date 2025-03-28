import os
import yaml
import random
import shutil
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split
from torchvision.utils import save_image
from ultralytics import YOLO

# 确保环境变量不会影响性能
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"

# 配置参数
CONFIG = {
    "data_dir": "Datasets",
    "image_dir": "images",
    "label_dir": "labels",
    "split_ratio": [0.7, 0.2, 0.1],
    "model_type": "yolov8n-pose.pt",
    "epochs": 200,
    "imgsz": 640,
    "batch": 32,
    "device": "0",
    "workers": 0,  # 避免 dataloader 线程问题
    "seed": 42
}

# 创建目录
os.makedirs("models", exist_ok=True)
os.makedirs("results", exist_ok=True)


def plot_training_curves(model):
    """ 绘制损失曲线、mAP 曲线等训练指标 """
    try:
        csv_file = next((f for f in os.listdir(model.trainer.save_dir) if f.endswith(".csv")), None)
        if not csv_file:
            print("⚠️ 未找到训练日志 CSV 文件，跳过绘图")
            return

        metrics = pd.read_csv(os.path.join(model.trainer.save_dir, csv_file))

        plt.figure(figsize=(12, 8))

        # 1. 损失曲线
        plt.subplot(2, 2, 1)
        for col in [c for c in metrics.columns if 'loss' in c]:
            plt.plot(metrics["epoch"], metrics[col], label=col.split('/')[-1])
        plt.title("Training Loss")
        plt.legend()

        # 2. mAP 变化曲线
        plt.subplot(2, 2, 2)
        for col in [c for c in metrics.columns if 'metrics/mAP' in c]:
            plt.plot(metrics["epoch"], metrics[col], label=col.split('/')[-1])
        plt.title("Validation mAP")
        plt.legend()

        # 3. 学习率变化曲线
        plt.subplot(2, 2, 3)
        if 'lr/pg0' in metrics.columns:
            plt.plot(metrics["epoch"], metrics["lr/pg0"], label='Learning Rate')
        plt.title("Learning Rate Schedule")
        plt.legend()

        plt.tight_layout()
        plt.savefig("results/training_curves.png", dpi=300)
        plt.close()
    except Exception as e:
        print(f"⚠️ 绘制训练曲线时出错: {str(e)}")


def analyze_results(model):
    """鲁棒性更强的结果分析函数"""
    try:
        val_results = model.val(split="val")
        os.makedirs("results", exist_ok=True)

        # 初始化变量
        px, py, ap = None, None, None

        if hasattr(val_results, 'box'):
            box = val_results.box

            # 安全获取指标数据
            py = getattr(box, 'p', None)
            if py is None:
                py = getattr(box, 'precisions', [0.5])  # 兼容不同版本

            # 确保py是可迭代对象
            if not isinstance(py, (list, np.ndarray)):
                py = [float(py)] if py else [0.5]

            py = np.array(py).flatten()  # 强制转为1D数组

            # 生成对应x轴
            px = np.linspace(0, 1, len(py)) if len(py) > 1 else np.array([0, 1])

            # 获取AP值（兼容不同格式）
            ap = getattr(box, 'ap', None)
            if ap is None:
                ap = getattr(box, 'ap50', 0.0)
            ap_value = float(ap[0] if isinstance(ap, (list, np.ndarray)) else ap)

            # 绘制PR曲线
            plt.figure(figsize=(9, 6))
            if len(py) == 1:  # 单点情况
                plt.scatter(0.5, py[0], s=100, c='blue',
                            label=f'mAP@0.5: {ap_value:.3f}')
            else:
                plt.plot(px, py, 'b-', linewidth=3,
                         label=f'mAP@0.5: {ap_value:.3f}')

            plt.xlabel('Recall')
            plt.ylabel('Precision')
            plt.xlim(0, 1)
            plt.ylim(0, 1)
            plt.title('Precision-Recall Curve')
            plt.legend()
            plt.show()
            plt.savefig("results/pr_curve.png", dpi=300, bbox_inches='tight')
            plt.close()
            print("✅ PR曲线已保存到 results/pr_curve.png")

    except Exception as e:
        print(f"⚠️ 结果分析出错: {str(e)}")
        # 安全打印调试信息
        debug_info = {
            'px': px.shape if hasattr(px, 'shape') else str(px),
            'py': py.shape if hasattr(py, 'shape') else str(py),
            'ap': str(ap)
        }
        print(f"🐞 调试信息: {debug_info}")


def train_model():
    """ 训练模型并保存到 models 目录 """
    model = YOLO(CONFIG["model_type"])

    results = model.train(
        data=f"{CONFIG['data_dir']}/data.yaml",
        epochs=CONFIG["epochs"],
        imgsz=CONFIG["imgsz"],
        batch=CONFIG["batch"],
        device=CONFIG["device"],
        save=True,
        save_period=10,
        exist_ok=True,
        workers=CONFIG["workers"],
        save_dir="models",
        fliplr = 0,  # 50% 概率水平翻转
        flipud = 0,  # 20% 概率垂直翻转
        degrees = 180 # 允许±10° 旋转
    )

    best_model_path = os.path.join("models", "best.pt")
    last_model_path = os.path.join("models", "last.pt")

    if os.path.exists(best_model_path):
        shutil.move(best_model_path, "models/best.pt")
    if os.path.exists(last_model_path):
        shutil.move(last_model_path, "models/last.pt")

    return model


def test_model(model):
    """ 在测试集上推理，并保存结果 """
    print("\n🔍 在测试集上运行推理...")
    results = model.predict(
        source=f"{CONFIG['data_dir']}/test/images",
        save=True,
        project="results",
        name="test_predictions"
    )
    print("✅ 推理完成，结果已保存到 results/test_predictions")


def save_augmented_samples(model, num_samples=5):
    """ 从 dataloader 直接获取增强后的训练数据，并保存 """
    print("\n📷 正在保存训练时的数据增强样本...")

    # 获取训练时的数据加载器
    dataloader = model.trainer.get_dataloader('./Datasets/train/images')  # 直接访问 dataloader
    os.makedirs("results/augmentations/train", exist_ok=True)

    # 遍历数据集，提取增强后的样本
    for i, (images, labels) in enumerate(dataloader):
        if i >= num_samples:  # 只保存指定数量的样本
            break
        save_path = f"results/augmentations/train/sample_{i}.png"

        # 保存 PyTorch 张量格式的增强图片
        save_image(images / 255.0, save_path)  # 归一化到 [0,1]，避免颜色错误

    print(f"✅ 数据增强样本已保存至 'results/augmentations/train'")


def save_augmentation_samples(model, output_dir="aug_samples"):
    """
    保存训练过程中的增强样本
    :param model: 训练好的YOLO模型
    :param output_dir: 输出目录
    """
    os.makedirs(output_dir, exist_ok=True)

    # 从训练日志中提取增强样本
    train_log_dir = model.trainer.save_dir
    mosaic_files = []

    # 查找所有增强样本图片
    for root, _, files in os.walk(train_log_dir):
        for f in files:
            if f.startswith("train_batch") and f.endswith(".jpg"):
                mosaic_files.append(os.path.join(root, f))

    # 复制最新的10张增强样本
    for i, src in enumerate(sorted(mosaic_files)[-10:]):
        dst = os.path.join(output_dir, f"aug_sample_{i + 1}.jpg")
        shutil.copy(src, dst)

    # 如果没有自动保存的样本，手动生成
    if len(mosaic_files) == 0:
        print("\n警告：未找到自动保存的增强样本，将手动生成...")
        from ultralytics.data.augment import Albumentations
        from PIL import Image

        # 示例增强管道
        augmenter = Albumentations(
            augment=True,
            hsv_h=0.015,  # 色调增强
            hsv_s=0.7,  # 饱和度增强
            hsv_v=0.4,  # 明度增强
            degrees=10,  # 旋转角度
            translate=0.1,  # 平移
            scale=0.5,  # 缩放
            shear=2  # 剪切
        )

        # 随机选择训练集图片演示
        img_files = [f for f in os.listdir("yolo_dataset/train/images")
                     if f.endswith(('.jpg', '.png'))][:5]

        for i, img_file in enumerate(img_files):
            img = Image.open(os.path.join("yolo_dataset/train/images", img_file))

            # 应用增强
            augmented = augmenter(image=np.array(img))['image']

            # 保存对比图
            fig, ax = plt.subplots(1, 2, figsize=(12, 6))
            ax[0].imshow(img)
            ax[0].set_title("Original")
            ax[0].axis('off')

            ax[1].imshow(augmented)
            ax[1].set_title("Augmented")
            ax[1].axis('off')

            plt.show()
            plt.savefig(os.path.join(output_dir, f"aug_compare_{i + 1}.jpg"),
                        dpi=150, bbox_inches='tight')
            plt.close()

    print(f"\n增强样本已保存到: {os.path.abspath(output_dir)}")

def visualize_augmentations(model):
    """
    1. 正确获取训练时的增强数据
    2. 仅用于可视化，不进行推理
    """
    save_augmented_samples(model, num_samples=5)


if __name__ == "__main__":
    random.seed(CONFIG["seed"])
    np.random.seed(CONFIG["seed"])

    print("🚀 开始训练 YOLOv8 ...")
    model = train_model()

    print("\n📊 评估训练结果...")
    analyze_results(model)

    print("\n🎯 绘制训练曲线...")
    plot_training_curves(model)

    print("\n🛠 运行测试集推理...")
    test_model(model)

    print("\n🔍 可视化训练数据增强...")
    # visualize_augmentations(model)

    save_augmentation_samples(model)


    print("\n✅ 训练完成！")
