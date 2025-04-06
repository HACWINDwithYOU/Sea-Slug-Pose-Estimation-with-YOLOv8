import os
import xml.etree.ElementTree as ET
from tqdm import tqdm


import os
import xml.etree.ElementTree as ET
from tqdm import tqdm

def convert_cvat_skeleton_to_yolo(xml_path, image_dir, output_dir, prefix):
    """确保关键点按指定顺序存储"""
    tree = ET.parse(xml_path)
    root = tree.getroot()
    os.makedirs(output_dir, exist_ok=True)

    # 关键点顺序定义
    kpt_order = ["head", "back", "tail", "left", "right"]
    img_width = img_height = 540

    # 按帧号整理标注
    frame_annotations = {}
    for track in tqdm(root.findall(".//track"), desc="处理标注"):
        for skeleton in track.findall(".//skeleton"):
            if skeleton.attrib.get("outside") == "1":
                continue

            frame_id = int(skeleton.attrib["frame"])
            if frame_id not in frame_annotations:
                frame_annotations[frame_id] = []

            # 按kpt_order顺序收集关键点
            kpts_dict = {}
            for point in skeleton.findall(".//points"):
                label = point.attrib["label"]
                if label in kpt_order:
                    x, y = map(float, point.attrib["points"].split(","))
                    kpts_dict[label] = [x/img_width, y/img_height]

            # 确保所有关键点都存在并按顺序排列
            if len(kpts_dict) == len(kpt_order):
                ordered_kpts = []
                for label in kpt_order:
                    ordered_kpts.extend(kpts_dict[label])
                frame_annotations[frame_id].append(ordered_kpts)

    # 写入YOLO格式
    for frame_id, instances in tqdm(frame_annotations.items(), desc="写入标签"):
        image_name = f"{prefix}_frame_{frame_id:06d}.jpg"
        label_path = os.path.join(output_dir, os.path.splitext(image_name)[0] + ".txt")

        with open(label_path, "w") as f:
            for kpts in instances:
                # 计算边界框
                xs = kpts[::2]  # 所有x坐标
                ys = kpts[1::2] # 所有y坐标
                x_center = (min(xs) + max(xs)) / 2
                y_center = (min(ys) + max(ys)) / 2
                width = max(xs) - min(xs)
                height = max(ys) - min(ys)

                # 写入: class + box + 有序关键点
                line = [0, x_center, y_center, width, height] + kpts
                f.write(" ".join(f"{x:.6f}" for x in line) + "\n")

# 使用示例
convert_cvat_skeleton_to_yolo(
    xml_path="annotations/annotations2.xml",
    image_dir="images",
    output_dir="labels",
    prefix="video2"
)

import os


def keep_first_line(directory):
    """
    遍历指定目录中的所有 .txt 文件，仅保留第一行内容。

    :param directory: 包含 .txt 文件的文件夹路径
    """
    # 获取所有 .txt 文件
    txt_files = [f for f in os.listdir(directory) if f.endswith('.txt')]

    if not txt_files:
        print("未找到任何 .txt 文件！")
        return

    for file in txt_files:
        file_path = os.path.join(directory, file)

        try:
            # 读取文件的第一行
            with open(file_path, 'r', encoding='utf-8') as f:
                first_line = f.readline().strip()  # 仅读取第一行，并去掉首尾空格

            # 重新写入文件，仅包含第一行
            with open(file_path, 'w', encoding='utf-8') as f:
                f.write(first_line + "\n")

            print(f"已处理: {file_path}")

        except Exception as e:
            print(f"处理 {file} 时发生错误: {e}")


# 使用示例
keep_first_line("labels")  # 替换为你的文件夹路径
