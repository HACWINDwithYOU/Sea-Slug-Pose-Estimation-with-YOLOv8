import cv2
import os
from tqdm import tqdm


def extract_frames(video_path, output_dir, frame_interval=30, prefix="frame_"):
    """
    从视频中按固定间隔抽帧
    :param video_path: 输入视频路径
    :param output_dir: 输出目录
    :param frame_interval: 抽帧间隔（每N帧取1帧）
    :param prefix: 文件名前缀（需与CVAT标注一致）
    """
    # 创建输出目录
    os.makedirs(output_dir, exist_ok=True)

    # 打开视频文件
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"无法打开视频文件 {video_path}")

    # 获取视频信息
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration = total_frames / fps
    print(f"视频信息: {fps:.2f} FPS, 总帧数: {total_frames}, 时长: {duration:.2f}秒")

    # 开始抽帧
    saved_count = 0
    for frame_num in tqdm(range(total_frames), desc="正在抽帧"):
        ret, frame = cap.read()
        if not ret:
            break

        # 每N帧保存一次
        if frame_num % frame_interval == 0:
            # 生成与CVAT标注匹配的文件名（6位数字补零）
            output_name = f"{prefix}{frame_num:06d}.jpg"
            output_path = os.path.join(output_dir, output_name)

            # 保存图片（质量95%）
            cv2.imwrite(output_path, frame, [cv2.IMWRITE_JPEG_QUALITY, 95])
            saved_count += 1

    cap.release()
    print(f"抽帧完成！共保存 {saved_count} 张图片到 {output_dir}")


# 使用示例
if __name__ == "__main__":
    video_path = "../Videos/video2.mp4"  # 替换为你的视频路径
    output_dir = "images"

    # 每30帧抽1帧，文件名格式为frame_000000.jpg
    extract_frames(
        video_path=video_path,
        output_dir=output_dir,
        frame_interval=30,
        prefix="video2_frame_"  # 必须与CVAT标注的图片名前缀一致
    )