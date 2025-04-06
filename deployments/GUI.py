import sys
import cv2
import numpy as np
from PyQt5.QtWidgets import QApplication, QWidget, QLabel, QPushButton, QFileDialog, QHBoxLayout, QVBoxLayout
from PyQt5.QtGui import QImage, QPixmap
from PyQt5.QtCore import QTimer
from ultralytics import YOLO


class KalmanFilter2D:
    def __init__(self):
        self.kf = cv2.KalmanFilter(4, 2)  # (x, y, dx, dy)
        self.kf.measurementMatrix = np.array([[1, 0, 0, 0],
                                               [0, 1, 0, 0]], np.float32)
        self.kf.transitionMatrix = np.array([[1, 0, 1, 0],
                                              [0, 1, 0, 1],
                                              [0, 0, 1, 0],
                                              [0, 0, 0, 1]], np.float32)
        self.kf.processNoiseCov = np.eye(4, dtype=np.float32) * 0.03
        self.kf.statePost = np.zeros((4, 1), dtype=np.float32)

    def update(self, coord):
        measurement = np.array([[np.float32(coord[0])], [np.float32(coord[1])]])
        self.kf.correct(measurement)
        prediction = self.kf.predict()
        return prediction[0][0], prediction[1][0]


class VideoApp(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Pose Estimation with Kalman Filter")
        self.model = YOLO("../runs/pose/train/weights/best.pt")
        self.kalman_filters = []  # 每个关键点一个卡尔曼滤波器

        self.label_raw = QLabel("Raw")
        self.label_filtered = QLabel("Filtered")

        self.btn = QPushButton("选择视频")
        self.btn.clicked.connect(self.load_video)

        layout = QVBoxLayout()
        h_layout = QHBoxLayout()
        h_layout.addWidget(self.label_raw)
        h_layout.addWidget(self.label_filtered)

        layout.addLayout(h_layout)
        layout.addWidget(self.btn)
        self.setLayout(layout)

        self.timer = QTimer()
        self.timer.timeout.connect(self.update_frame)

    def load_video(self):
        path, _ = QFileDialog.getOpenFileName()
        if path:
            self.cap = cv2.VideoCapture(path)
            self.timer.start(30)

    def update_frame(self):
        ret, frame = self.cap.read()
        if not ret:
            self.timer.stop()
            return

        results = self.model.predict(source=frame, save=False, conf=0.3, verbose=False)[0]

        raw_frame = frame.copy()
        filtered_frame = frame.copy()

        for person in results.keypoints.xy:
            person = person.cpu().numpy()

            if len(self.kalman_filters) != len(person):
                self.kalman_filters = [KalmanFilter2D() for _ in person]

            filtered_points = []
            for i, (x, y) in enumerate(person):
                if x < 0 or y < 0:
                    filtered_points.append(None)
                    continue

                # 原始点颜色（只显示3/4特殊色，其余绿色）
                if i == 3:
                    color = (255, 0, 0)  # 蓝色：左点
                elif i == 4:
                    color = (255, 165, 0)  # 橙色：右点
                else:
                    color = (0, 255, 0)

                cv2.circle(raw_frame, (int(x), int(y)), 4, color, -1)

                fx, fy = self.kalman_filters[i].update((x, y))
                cv2.circle(filtered_frame, (int(fx), int(fy)), 4, color, -1)

                filtered_points.append((fx, fy))

            if filtered_points[3] and filtered_points[4]:
                left = np.array(filtered_points[3])
                right = np.array(filtered_points[4])
                mid = ((left + right) / 2).astype(int)

                # 🔴 中点（红色圆）
                cv2.circle(filtered_frame, tuple(mid), 6, (0, 0, 255), -1)

                # 🔁 箭头方向（3→4 向量 - 90°）
                vec = right - left
                angle_rad = np.arctan2(vec[1], vec[0])
                rotated_angle = angle_rad - np.pi / 2
                angle_deg = np.degrees(rotated_angle) % 360

                # 箭头终点（箭头长度设为40）
                arrow_length = 40
                arrow_dx = int(arrow_length * np.cos(rotated_angle))
                arrow_dy = int(arrow_length * np.sin(rotated_angle))
                arrow_end = (mid[0] + arrow_dx, mid[1] + arrow_dy)

                cv2.arrowedLine(filtered_frame, tuple(mid), arrow_end, (0, 0, 255), 2, tipLength=0.3)

                # ⬇️ 在右下角显示白色信息（角度和坐标）
                text_angle = f"Angle: {angle_deg:.1f} deg"
                text_coord = f"Midpoint: ({mid[0]}, {mid[1]})"

                org = (filtered_frame.shape[1] - 260, filtered_frame.shape[0] - 30)
                org2 = (org[0], org[1] + 25)

                cv2.putText(filtered_frame, text_angle, org, cv2.FONT_HERSHEY_SIMPLEX,
                            0.6, (255, 255, 255), 2, cv2.LINE_AA)
                cv2.putText(filtered_frame, text_coord, org2, cv2.FONT_HERSHEY_SIMPLEX,
                            0.6, (255, 255, 255), 2, cv2.LINE_AA)

        self.show_image(self.label_raw, raw_frame)
        self.show_image(self.label_filtered, filtered_frame)

    def show_image(self, label, frame):
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb.shape
        bytes_per_line = ch * w
        img = QImage(rgb.data, w, h, bytes_per_line, QImage.Format_RGB888)
        label.setPixmap(QPixmap.fromImage(img).scaled(label.width(), label.height()))


if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = VideoApp()
    window.resize(1200, 600)
    window.show()
    sys.exit(app.exec_())
