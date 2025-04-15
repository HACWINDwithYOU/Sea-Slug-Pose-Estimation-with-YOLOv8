import numpy as np
import pygame
import sys

# 初始化 Pygame
pygame.init()
WIDTH, HEIGHT = 800, 1000  # 加高窗口用于绘图
screen = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption("Sea Slug Navigation Simulation")
clock = pygame.time.Clock()

# 场地和仿真参数
RADIUS = 300
CENTER = np.array([WIDTH // 2, 400])  # 场地位置上移，留出下方绘图空间
dt = 0.02
v = 100
k_p = 6.0
k_d = 2.5

# 初始状态
pos = CENTER + np.array([0, -150])
theta = np.pi / 2
omega = 0
trajectory = []

# 用于绘图的历史记录
alpha_history = []
theta_history = []
history_len = 200  # 滑动窗口长度

# 控制器函数
def feedback_controller(pos, theta, omega, target, k_p=5.0, k_d=2.0):
    vec_to_target = target - pos
    desired_theta = np.arctan2(vec_to_target[1], vec_to_target[0])
    error = desired_theta - theta
    error = (error + np.pi) % (2 * np.pi) - np.pi
    alpha = k_p * error - k_d * omega
    return alpha

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
    alpha = feedback_controller(pos, theta, omega, target, k_p, k_d)
    omega += alpha * dt
    theta += omega * dt

    # 更新位置
    pos = pos + v * dt * np.array([np.cos(theta), np.sin(theta)])
    trajectory.append(pos.copy())

    # 更新历史记录
    alpha_history.append(alpha)
    theta_history.append(theta)
    if len(alpha_history) > history_len:
        alpha_history.pop(0)
        theta_history.pop(0)

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

    # 绘制 alpha 和 theta 曲线
    graph_top = 820
    graph_height = 160
    graph_width = WIDTH
    pygame.draw.rect(screen, (245, 245, 245), (0, graph_top, graph_width, graph_height))
    pygame.draw.line(screen, (0, 0, 0), (0, graph_top + graph_height // 2), (WIDTH, graph_top + graph_height // 2), 1)

    if len(alpha_history) > 1:
        for i in range(1, len(alpha_history)):
            x1 = int((i - 1) * WIDTH / history_len)
            x2 = int(i * WIDTH / history_len)

            y1_alpha = int(graph_top + graph_height // 2 - alpha_history[i - 1] * 5)
            y2_alpha = int(graph_top + graph_height // 2 - alpha_history[i] * 5)
            pygame.draw.line(screen, (255, 0, 0), (x1, y1_alpha), (x2, y2_alpha), 2)

            # y1_theta = int(graph_top + graph_height // 2 - theta_history[i - 1] * 6)
            # y2_theta = int(graph_top + graph_height // 2 - theta_history[i] * 6)
            # pygame.draw.line(screen, (0, 0, 255), (x1, y1_theta), (x2, y2_theta), 2)

    pygame.display.flip()
    clock.tick(60)

pygame.quit()
sys.exit()
