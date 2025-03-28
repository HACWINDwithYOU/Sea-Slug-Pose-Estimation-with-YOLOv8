import cv2
import os
import random
import numpy as np

import matplotlib.pyplot as plt

def show_image_with_matplotlib(img, title="Image"):
    """ 使用 Matplotlib 显示图像（适用于 GUI 不支持的情况） """
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    plt.figure(figsize=(8, 6))
    plt.imshow(img_rgb)
    plt.title(title)
    plt.axis("off")
    plt.show()



def visualize_annotations(image_dir, label_dir, num_samples=10):
    """
    可视化验证 YOLO 格式的标注数据
    :param image_dir: 图片目录路径
    :param label_dir: 标签目录路径
    :param num_samples: 随机检查的样本数
    """
    # 颜色定义
    BOX_COLOR = (0, 255, 0)  # 绿色边界框
    KPT_COLORS = [
        (255, 0, 0),  # 红色 - head
        (0, 0, 255),  # 蓝色 - back
        (255, 255, 0),  # 青色 - tail
        (255, 0, 255),  # 紫色 - left
        (0, 255, 255)  # 黄色 - right
    ]
    KPT_RADIUS = 5
    BOX_THICKNESS = 2

    # 获取可用的图片文件
    image_files = [f for f in os.listdir(image_dir) if f.endswith(('.jpg', '.png')) and os.path.isfile(os.path.join(image_dir, f))]
    if not image_files:
        print(f"错误：{image_dir} 中没有找到图片文件")
        return

    # 随机选择样本
    samples = random.sample(image_files, min(num_samples, len(image_files)))

    for img_file in samples:
        # 读取图片
        img_path = os.path.join(image_dir, img_file)
        img = cv2.imread(img_path)
        if img is None:
            print(f"警告：无法读取图片 {img_file}，请检查文件路径或图片是否损坏。")
            continue

        h, w = img.shape[:2]
        label_path = os.path.join(label_dir, os.path.splitext(img_file)[0] + ".txt")

        # 读取标注
        if not os.path.exists(label_path):
            print(f"警告：未找到 {img_file} 对应的标注文件，跳过。")
            continue

        with open(label_path) as f:
            annotations = f.readlines()

        # 绘制每个标注
        for ann in annotations:
            parts = ann.strip().split()
            if len(parts) < 5:  # 至少包含 class_id + bbox
                print(f"警告：{label_path} 文件格式错误，跳过该行。")
                continue

            # 解析边界框 (xywh 归一化 → 像素坐标)
            try:
                parts = list(map(float, parts))  # 确保所有数据都是浮点数
            except ValueError:
                print(f"错误：{label_path} 中包含无法解析的字符，跳过该文件。")
                break  # 直接跳过当前文件

            class_id = int(parts[0])
            x_center, y_center, width, height = parts[1:5]
            x1 = int((x_center - width / 2) * w)
            y1 = int((y_center - height / 2) * h)
            x2 = int((x_center + width / 2) * w)
            y2 = int((y_center + height / 2) * h)

            # 绘制边界框
            cv2.rectangle(img, (x1, y1), (x2, y2), BOX_COLOR, BOX_THICKNESS)
            cv2.putText(img, f"Class {class_id}", (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, BOX_COLOR, 2)

            # 绘制关键点 (如果有)
            if len(parts) > 5:
                kpts = parts[5:]
                if len(kpts) % 2 != 0:
                    print(f"警告：{label_path} 关键点数据格式错误，跳过关键点绘制。")
                    continue  # 关键点数据应成对出现

                for i in range(0, len(kpts), 2):
                    x = int(kpts[i] * w)
                    y = int(kpts[i + 1] * h)
                    color = KPT_COLORS[i // 2 % len(KPT_COLORS)]  # 防止超出索引范围
                    cv2.circle(img, (x, y), KPT_RADIUS, color, -1)
                    cv2.putText(img, str(i // 2), (x + 5, y - 5),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)

        # 显示结果
        show_image_with_matplotlib(img, title=f"Visualization: {img_file}")



# 使用示例
visualize_annotations(
    image_dir="images",  # 修改为你的图片路径
    label_dir="labels",  # 修改为你的标签路径
    num_samples=10  # 随机检查3张
)
