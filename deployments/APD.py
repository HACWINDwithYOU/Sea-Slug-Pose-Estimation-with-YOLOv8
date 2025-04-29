import pygame
import numpy as np

# 初始化
pygame.init()
screen = pygame.display.set_mode((800, 800))
clock = pygame.time.Clock()

# 海兔参数
pos = np.array([400.0, 400.0])
theta = 0.0
omega = 0.0
dt = 0.02
v = 100

# 控制器基础增益
base_k_p = 6.0
base_k_d = 2.5
alpha_p = 0.02
alpha_d = 0.01

# 目标点
target = np.array([600.0, 600.0])

# 时变增益扰动函数
def get_dynamic_gain():
    return 1.0 + 5 * np.sin(pygame.time.get_ticks() * 0.001)

# 控制器函数
def feedback_controller(pos, theta, omega, target, k_p, k_d):
    vec_to_target = target - pos
    desired_theta = np.arctan2(vec_to_target[1], vec_to_target[0])
    error = desired_theta - theta
    error = (error + np.pi) % (2 * np.pi) - np.pi
    alpha = k_p * error - k_d * omega
    return alpha, error

# 主循环
running = True
while running:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False
        elif event.type == pygame.MOUSEBUTTONDOWN:
            target = np.array(pygame.mouse.get_pos(), dtype=float)

    screen.fill((255, 255, 255))
    pygame.draw.circle(screen, (0, 0, 255), target.astype(int), 8)

    # 控制器计算
    alpha, error = feedback_controller(pos, theta, omega, target, base_k_p, base_k_d)

    # 自适应调整增益参数
    k_p = base_k_p + alpha_p * abs(error)
    k_d = base_k_d + alpha_d * abs(omega)

    # 引入时变系统动态增益
    dynamic_gain = get_dynamic_gain()
    omega += dynamic_gain * alpha * dt
    theta += omega * dt

    # 更新位置
    pos += v * np.array([np.cos(theta), np.sin(theta)]) * dt

    # 绘制海兔
    pygame.draw.circle(screen, (255, 0, 0), pos.astype(int), 10)
    dir_line = pos + 20 * np.array([np.cos(theta), np.sin(theta)])
    pygame.draw.line(screen, (0, 0, 0), pos.astype(int), dir_line.astype(int), 2)

    pygame.display.flip()
    clock.tick(50)

pygame.quit()
