import os
import yaml
import numpy as np
import torch
from tqdm import tqdm
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error

from ultralytics import YOLO

# 配置
CONFIG = {
    "data_dir": "Datasets",
    "model_path": "models/best.pt",
    "imgsz": 640,
    "device": "cuda:0",  # <-- 这里改了
    "save_dir": "results/test_evaluation"
}

# 创建保存目录
os.makedirs(CONFIG["save_dir"], exist_ok=True)

# 加载模型
model = YOLO(CONFIG["model_path"])
model.to(CONFIG["device"])

# 读取数据集信息
with open(f"{CONFIG['data_dir']}/data.yaml", "r") as f:
    data_info = yaml.safe_load(f)
test_img_dir = os.path.join(CONFIG["data_dir"], "test", "images")
test_label_dir = os.path.join(CONFIG["data_dir"], "test", "labels")
names = data_info["names"]

# 推理参数
CONF_THRESH = 0.25  # 置信度阈值

# 存储统计信息
all_true = []
all_preds = []

# 计算欧氏距离
def euclidean_distance(p1, p2):
    return np.sqrt(np.sum((p1 - p2) ** 2))

# 遍历测试集推理
test_images = [f for f in os.listdir(test_img_dir) if f.endswith(('.jpg', '.png'))]
for img_name in tqdm(test_images, desc="Testing"):
    img_path = os.path.join(test_img_dir, img_name)
    label_path = os.path.join(test_label_dir, os.path.splitext(img_name)[0] + ".txt")

    # 推理
    results = model.predict(img_path, save=False, imgsz=CONFIG["imgsz"], conf=CONF_THRESH, device=CONFIG["device"])
    pred = results[0]

    # 提取预测的关键点坐标
    if pred.keypoints is not None:
        pred_coords = pred.keypoints[0].cpu().numpy()  # 获取第一个检测到的关键点
    else:
        pred_coords = np.zeros((5, 2))  # 如果没有关键点预测，则默认为0

    # 加载GT标签
    if os.path.exists(label_path):
        with open(label_path, "r") as f:
            lines = f.readlines()
            gt_coords = []
            for line in lines:
                parts = line.strip().split()
                coords = list(map(float, parts[1:]))  # 获取真实坐标
                gt_coords.append(coords)
            gt_coords = np.array(gt_coords)
    else:
        gt_coords = np.zeros((0, 2))

    # 确保预测和真实坐标都具有相同的数量
    if len(pred_coords) == len(gt_coords):
        for pred_coord, gt_coord in zip(pred_coords, gt_coords):
            distance = euclidean_distance(pred_coord, gt_coord)  # 计算欧氏距离
            all_true.append(gt_coord)
            all_preds.append(pred_coord)

# 计算平均绝对误差 (MAE)
all_true = np.array(all_true)
all_preds = np.array(all_preds)
mae = mean_absolute_error(all_true, all_preds)

# 汇总评估结果
print("\n📈 测试集评估结果:")
print(f"Mean Absolute Error (MAE): {mae:.4f}")
