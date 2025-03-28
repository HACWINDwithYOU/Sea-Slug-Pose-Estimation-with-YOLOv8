import os
import cv2
import numpy as np
import matplotlib.pyplot as plt

# 配置路径
IMAGE_DIR = "images"  # 你的图像文件夹
LABEL_DIR = "labels"  # 你的关键点标注文件夹
SAVE_DIR = "visualize_labels"  # 可视化结果保存路径

os.makedirs(SAVE_DIR, exist_ok=True)

# 获取所有图片文件
image_files = [f for f in os.listdir(IMAGE_DIR) if f.endswith((".jpg", ".png"))]

# 颜色列表（用于绘制不同的关键点）
COLORS = [(255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0), (255, 0, 255)]

# 关键点半径
POINT_RADIUS = 4

# 遍历每张图片，读取并绘制关键点
for img_file in image_files:
    img_path = os.path.join(IMAGE_DIR, img_file)
    label_path = os.path.join(LABEL_DIR, os.path.splitext(img_file)[0] + ".txt")

    # 读取图像
    image = cv2.imread(img_path)
    if image is None:
        print(f"⚠️ 无法读取图像: {img_path}")
        continue

    # 读取关键点标注
    if not os.path.exists(label_path):
        print(f"⚠️ 未找到对应的标注文件: {label_path}")
        continue

    with open(label_path, "r") as f:
        lines = f.readlines()

    for line in lines:
        parts = line.strip().split()
        if len(parts) < 11:
            print(f"⚠️ 标注格式错误: {label_path} → {line}")
            continue

        # 解析关键点坐标（假设前 1 个是类别，接下来的 10 个是 5 组关键点 (x, y)）
        keypoints = np.array(parts[1:11], dtype=np.float32).reshape(-1, 2)

        # 反归一化：YOLO 格式是 (x_center, y_center) 归一化到 0~1，需要转换到像素坐标
        h, w, _ = image.shape
        keypoints[:, 0] *= w  # 还原 x 坐标
        keypoints[:, 1] *= h  # 还原 y 坐标

        # 绘制关键点
        for idx, (x, y) in enumerate(keypoints):
            x, y = int(x), int(y)
            cv2.circle(image, (x, y), POINT_RADIUS, COLORS[idx % len(COLORS)], -1)
            cv2.putText(image, str(idx), (x + 5, y - 5), cv2.FONT_HERSHEY_SIMPLEX,
                        0.5, COLORS[idx % len(COLORS)], 1)

    # 显示并保存结果
    save_path = os.path.join(SAVE_DIR, img_file)
    cv2.imwrite(save_path, image)

    plt.figure(figsize=(6, 6))
    plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    plt.title(f"Keypoints: {img_file}")
    plt.axis("off")
    plt.show()

print(f"✅ 关键点可视化完成，结果保存在 '{SAVE_DIR}' 目录")
