import numpy as np
import pygame
import sys

# 初始化 Pygame
pygame.init()
WIDTH, HEIGHT = 800, 1000
screen = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption("Sea Slug Navigation Simulation")
clock = pygame.time.Clock()

# 场地和仿真参数
RADIUS = 300
CENTER = np.array([WIDTH // 2, 400])
dt = 0.02
v = 100

# 初始控制器参数
k_p = 6.0
k_d = 2.5

# 初始状态
pos = CENTER + np.array([0, -150])
theta = np.pi / 2
omega = 0
trajectory = []

# 控制器参数历史记录（用于估计比例系数）
alpha_history = []
theta_history = []
error_history = []
history_len = 200  # 滑动窗口长度
ema_alpha = 0.1  # 指数加权平均因子

def get_dynamic_gain():
    return 1.0 + 0.5 * np.sin(pygame.time.get_ticks() * 0.001)

# 控制器函数
def feedback_controller(pos, theta, omega, target, k_p=5.0, k_d=2.0):
    vec_to_target = target - pos
    desired_theta = np.arctan2(vec_to_target[1], vec_to_target[0])
    error = desired_theta - theta
    error = (error + np.pi) % (2 * np.pi) - np.pi
    alpha = (k_p * error - k_d * omega) * get_dynamic_gain()
    return alpha, error


# 平滑历史误差和输出，用于估计控制器参数
def update_control_params(error, alpha, ema_alpha=0.1):
    global k_p, k_d

    # 指数移动平均估计比例系数
    error_history.append(error)
    alpha_history.append(alpha)

    if len(error_history) > history_len:
        error_history.pop(0)
        alpha_history.pop(0)

    # 使用历史误差和控制量进行估算（这里的估算方法可以改进）
    avg_error = np.mean(error_history)
    avg_alpha = np.mean(alpha_history)

    # 基于误差的大小更新k_p和k_d
    k_p = 6.0 + ema_alpha * avg_error  # 更新比例系数
    k_d = 2.5 + ema_alpha * avg_alpha  # 更新微分系数


# 主循环
target = CENTER.copy()
running = True

while running:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False
        if event.type == pygame.MOUSEBUTTONDOWN:
            target = np.array(pygame.mouse.get_pos())

    # 控制器计算
    alpha, error = feedback_controller(pos, theta, omega, target, k_p, k_d)
    update_control_params(error, alpha, ema_alpha=0.1)  # 更新PD控制器参数

    omega += alpha * dt
    theta += omega * dt

    # 更新位置
    pos = pos + v * dt * np.array([np.cos(theta), np.sin(theta)])
    trajectory.append(pos.copy())

    # 出界检测
    if np.linalg.norm(pos - CENTER) > RADIUS:
        print("出界，仿真结束")
        break

    # GUI绘图
    screen.fill((255, 255, 255))

    # 绘制主场景
    pygame.draw.circle(screen, (230, 230, 230), CENTER, RADIUS, 5)
    pygame.draw.circle(screen, (255, 0, 0), target.astype(int), 6)
    pygame.draw.circle(screen, (0, 0, 255), pos.astype(int), 8)

    arrow_end = pos + 20 * np.array([np.cos(theta), np.sin(theta)])
    pygame.draw.line(screen, (0, 0, 255), pos, arrow_end, 3)

    if len(trajectory) > 1:
        pygame.draw.lines(screen, (0, 200, 0), False, trajectory, 2)

    # 绘制动态比例系数
    graph_top = 820
    graph_height = 160
    graph_width = WIDTH
    pygame.draw.rect(screen, (245, 245, 245), (0, graph_top, graph_width, graph_height))
    pygame.draw.line(screen, (0, 0, 0), (0, graph_top + graph_height // 2), (WIDTH, graph_top + graph_height // 2), 1)

    # 绘制k_p和k_d曲线
    if len(alpha_history) > 1:
        for i in range(1, len(alpha_history)):
            x1 = int((i - 1) * WIDTH / history_len)
            x2 = int(i * WIDTH / history_len)

            y1_kp = int(graph_top + graph_height // 2 - k_p * 10)
            y2_kp = int(graph_top + graph_height // 2 - k_p * 10)
            pygame.draw.line(screen, (255, 0, 0), (x1, y1_kp), (x2, y2_kp), 2)

            y1_kd = int(graph_top + graph_height // 2 - k_d * 10)
            y2_kd = int(graph_top + graph_height // 2 - k_d * 10)
            pygame.draw.line(screen, (0, 0, 255), (x1, y1_kd), (x2, y2_kd), 2)

    pygame.display.flip()
    clock.tick(60)

pygame.quit()
sys.exit()
