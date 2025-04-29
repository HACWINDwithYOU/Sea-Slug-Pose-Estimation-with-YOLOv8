import numpy as np
import pygame
import sys
import heapq

# 初始化 Pygame
pygame.init()
WIDTH, HEIGHT = 800, 1000
screen = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption("Sea Slug Navigation Simulation with Pathfinding")
clock = pygame.time.Clock()

# 场地与仿真参数
RADIUS = 300
CENTER = np.array([WIDTH // 2, 400])
dt = 0.02
v = 100
k_p = 6.0
k_d = 2.5

# 初始状态
pos = CENTER + np.array([0, -150])
theta = np.pi / 2
omega = 0
trajectory = []

# 历史记录
alpha_history = []
theta_history = []
history_len = 200

# 障碍线段（起点，终点）
obstacle_line = (CENTER + np.array([-100, -50]), CENTER + np.array([100, 50]))

# 栅格化参数
grid_size = 20
grid_width = WIDTH // grid_size
grid_height = HEIGHT // grid_size

def to_grid(p):
    return tuple((p // grid_size).astype(int))

def to_world(g):
    return np.array(g) * grid_size + grid_size // 2

def line_intersects(p1, p2, q1, q2):
    def ccw(a, b, c): return (c[1]-a[1]) * (b[0]-a[0]) > (b[1]-a[1]) * (c[0]-a[0])
    return ccw(p1, q1, q2) != ccw(p2, q1, q2) and ccw(p1, p2, q1) != ccw(p1, p2, q2)

def is_blocked(p1, p2):
    return line_intersects(p1, p2, *obstacle_line)

def a_star(start, goal):
    moves = [(1,0), (-1,0), (0,1), (0,-1), (1,1), (-1,-1), (1,-1), (-1,1)]
    frontier = []
    heapq.heappush(frontier, (0, 0, start, None))
    came_from = {}
    direction_from = {}

    while frontier:
        _, cost, current, prev = heapq.heappop(frontier)
        if current in came_from:
            continue
        came_from[current] = prev
        if current == goal:
            break

        for move in moves:
            neighbor = (current[0] + move[0], current[1] + move[1])
            if not (0 <= neighbor[0] < grid_width and 0 <= neighbor[1] < grid_height):
                continue

            p1 = to_world(np.array(current))
            p2 = to_world(np.array(neighbor))
            if is_blocked(p1, p2):
                continue

            move_dir = move
            prev_dir = direction_from.get(current)
            turn_penalty = 0.3 if prev_dir and move_dir != prev_dir else 0.0

            new_cost = cost + np.linalg.norm(move) + turn_penalty
            priority = new_cost + np.linalg.norm(np.array(neighbor) - np.array(goal))
            heapq.heappush(frontier, (priority, new_cost, neighbor, current))
            direction_from[neighbor] = move

    # 回溯路径并简化
    path = []
    node = goal
    while node:
        path.append(to_world(np.array(node)))
        node = came_from.get(node)
    path.reverse()

    # 简化路径：只保留转向点
    def direction(p1, p2):
        d = p2 - p1
        return tuple(np.round(d / np.linalg.norm(d))) if np.linalg.norm(d) > 0 else (0, 0)

    simplified = [path[0]]
    for i in range(1, len(path) - 1):
        d1 = direction(path[i - 1], path[i])
        d2 = direction(path[i], path[i + 1])
        if d1 != d2:
            simplified.append(path[i])
    if len(path) > 1:
        simplified.append(path[-1])
    return simplified

def feedback_controller(pos, theta, omega, target):
    vec = target - pos
    desired_theta = np.arctan2(vec[1], vec[0])
    error = (desired_theta - theta + np.pi) % (2 * np.pi) - np.pi
    alpha = k_p * error - k_d * omega
    return alpha


# 主循环
target = CENTER.copy()
waypoints = []
current_index = 0
running = True

while running:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False
        if event.type == pygame.MOUSEBUTTONDOWN:
            target = np.array(pygame.mouse.get_pos())
            start, goal = to_grid(pos), to_grid(target)
            waypoints = a_star(start, goal)
            current_index = 0

    if current_index < len(waypoints):
        next_target = waypoints[current_index]
        if np.linalg.norm(next_target - pos) < 50:
            current_index += 1
        else:
            alpha = feedback_controller(pos, theta, omega, next_target)
            # alpha = controller_tanh(pos, theta, omega, next_target)
            omega += alpha * dt
            theta += omega * dt
            pos = pos + v * dt * np.array([np.cos(theta), np.sin(theta)])

    trajectory.append(pos.copy())
    alpha_history.append(omega)
    theta_history.append(theta)
    if len(alpha_history) > history_len:
        alpha_history.pop(0)
        theta_history.pop(0)

    if np.linalg.norm(pos - CENTER) > RADIUS:
        print("出界，仿真结束")
        break

    screen.fill((255, 255, 255))
    pygame.draw.circle(screen, (230, 230, 230), CENTER, RADIUS, 5)
    pygame.draw.line(screen, (0, 0, 0), *obstacle_line, 4)
    pygame.draw.circle(screen, (255, 0, 0), target.astype(int), 6)
    pygame.draw.circle(screen, (0, 0, 255), pos.astype(int), 8)
    pygame.draw.line(screen, (0, 0, 255), pos, pos + 20 * np.array([np.cos(theta), np.sin(theta)]), 3)

    if len(trajectory) > 1:
        pygame.draw.lines(screen, (0, 200, 0), False, trajectory, 2)
    if len(waypoints) > 1:
        pygame.draw.lines(screen, (200, 100, 0), False, waypoints, 2)

    graph_top = 820
    graph_height = 160
    pygame.draw.rect(screen, (245, 245, 245), (0, graph_top, WIDTH, graph_height))
    pygame.draw.line(screen, (0, 0, 0), (0, graph_top + graph_height // 2), (WIDTH, graph_top + graph_height // 2), 1)

    for i in range(1, len(alpha_history)):
        x1 = int((i - 1) * WIDTH / history_len)
        x2 = int(i * WIDTH / history_len)
        y1 = int(graph_top + graph_height // 2 - alpha_history[i - 1] * 5)
        y2 = int(graph_top + graph_height // 2 - alpha_history[i] * 5)
        pygame.draw.line(screen, (255, 0, 0), (x1, y1), (x2, y2), 2)

    pygame.display.flip()
    clock.tick(60)

pygame.quit()
sys.exit()
