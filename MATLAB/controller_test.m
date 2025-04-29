% 时间设置
dt = 0.001;
T = 2.5;
time = 0:dt:T;
n = length(time);

% 初始变量
theta_P = zeros(1, n); omega_P = zeros(1, n);
theta_PD = zeros(1, n); omega_PD = zeros(1, n);
theta_AD = zeros(1, n); omega_AD = zeros(1, n); % AD: Adaptive PD

% 控制参数
Kp = 10; Kd = 4;
base_Kp = 10; base_Kd = 4;
alpha_p = 1.0; alpha_d = 0.5;

% 误差记录
theta_ref = ones(1, n); % 期望值（单位阶跃）

% 模拟主循环
for i = 2:n
    % 时变增益
    gain = 1.0 + 0.5 * sin(2 * time(i));

    % ----------- P 控制器 -----------
    e_P = theta_ref(i) - theta_P(i-1);
    u_P = Kp * e_P;
    alpha_P = gain * u_P;
    omega_P(i) = omega_P(i-1) + alpha_P * dt;
    theta_P(i) = theta_P(i-1) + omega_P(i) * dt;

    % ----------- PD 控制器 -----------
    e_PD = theta_ref(i) - theta_PD(i-1);
    de_PD = -omega_PD(i-1);  % 误差导数近似为 - 角速度
    u_PD = Kp * e_PD + Kd * de_PD;
    alpha_PD = gain * u_PD;
    omega_PD(i) = omega_PD(i-1) + alpha_PD * dt;
    theta_PD(i) = theta_PD(i-1) + omega_PD(i) * dt;

    % ----------- 自适应 PD 控制器 -----------
    e_AD = theta_ref(i) - theta_AD(i-1);
    de_AD = -omega_AD(i-1);
    k_p = base_Kp + alpha_p * abs(e_AD);
    k_d = base_Kd + alpha_d * abs(omega_AD(i-1));
    u_AD = k_p * e_AD + k_d * de_AD;
    alpha_AD = gain * u_AD;
    omega_AD(i) = omega_AD(i-1) + alpha_AD * dt;
    theta_AD(i) = theta_AD(i-1) + omega_AD(i) * dt;
end

% ---------- 绘图 ----------
figure;
plot(time, theta_ref, 'k--', 'LineWidth', 1.5); hold on;
plot(time, theta_P, 'r', 'LineWidth', 1.5);
plot(time, theta_PD, 'b', 'LineWidth', 1.5);
plot(time, theta_AD, 'g', 'LineWidth', 1.5);
xlabel('时间 (s)');
ylabel('角度输出');
legend('期望值', 'P控制器', 'PD控制器', '自适应PD控制器');
title('三种控制器在时变系统下的响应对比');
grid on;

%%
% 时间设置
clear; clc;
dt = 0.01;
T = 10;
time = 0:dt:T;
n = length(time);

% 初始变量
theta_P = zeros(1, n); omega_P = zeros(1, n);
theta_PD = zeros(1, n); omega_PD = zeros(1, n);
theta_AD = zeros(1, n); omega_AD = zeros(1, n); % AD: Adaptive PD

% 控制参数
Kp = 10; Kd = 4;
base_Kp = 10; base_Kd = 4;
alpha_p = 1; alpha_d = 1;

% 误差记录
theta_ref = ones(1, n); % 期望值（单位阶跃）
k_p = Kp;
k_d = Kd;
% 模拟主循环
for i = 2:n
    % 时变增益
    gain = 1;

    % ----------- P 控制器 -----------
    e_P = theta_ref(i) - theta_P(i-1);
    u_P = Kp * e_P;
    alpha_P = gain * u_P;
    omega_P(i) = omega_P(i-1) + alpha_P * dt;
    theta_P(i) = theta_P(i-1) + omega_P(i) * dt;

    % ----------- PD 控制器 -----------
    e_PD = theta_ref(i) - theta_PD(i-1);
    de_PD = -omega_PD(i-1);  % 误差导数近似为 - 角速度
    u_PD = Kp * e_PD + Kd * de_PD;
    alpha_PD = gain * u_PD;
    omega_PD(i) = omega_PD(i-1) + alpha_PD * dt;
    theta_PD(i) = theta_PD(i-1) + omega_PD(i) * dt;

    % ----------- 自适应 PD 控制器 -----------
    e_AD = theta_ref(i) - theta_AD(i-1);
    de_AD = -omega_AD(i-1);
    k_p = k_p + alpha_p * e_AD;
    k_d = k_d + alpha_d * ();
    u_AD = k_p * e_AD + k_d * de_AD;
    alpha_AD = gain * u_AD;
    omega_AD(i) = omega_AD(i-1) + alpha_AD * dt;
    theta_AD(i) = theta_AD(i-1) + omega_AD(i) * dt;
end

% ---------- 绘图 ----------
figure;
plot(time, theta_ref, 'k--', 'LineWidth', 1.5); hold on;
plot(time, theta_P, 'r', 'LineWidth', 1.5);
plot(time, theta_PD, 'b', 'LineWidth', 1.5);
plot(time, theta_AD, 'g', 'LineWidth', 1.5);
xlabel('时间 (s)');
ylabel('角度输出');
legend('期望值', 'P控制器', 'PD控制器', '自适应PD控制器');
title('三种控制器在时变系统下的响应对比');
grid on;

% ---------- 计算性能指标 ----------
controllers = {'P', 'PD', 'AD'};
thetas = {theta_P, theta_PD, theta_AD};
omegas = {omega_P, omega_PD, omega_AD};

fprintf('\n控制器性能对比（单位阶跃响应）：\n');
for k = 1:3
    y = thetas{k};
    e = theta_ref - y;
    overshoot = (max(y) - 1.0) * 100;
    settle_time = time(find(abs(y - 1.0) < 0.02, 1));
    ss_error = abs(y(end) - 1.0);
    ise = sum(e.^2) * dt;
    fprintf('%s控制器: 超调 = %.2f%%, 建立时间 = %.2fs, 稳态误差 = %.4f, ISE = %.4f\n', ...
        controllers{k}, overshoot, settle_time, ss_error, ise);
end

%%
% 时间设置
clear; clc;
dt = 0.01;
T = 10;
time = 0:dt:T;
n = length(time);

% 初始变量
theta_P = zeros(1, n); omega_P = zeros(1, n);
theta_PD = zeros(1, n); omega_PD = zeros(1, n);
theta_AD = zeros(1, n); omega_AD = zeros(1, n); % AD: Adaptive PD

% 控制参数
Kp = 10; Kd = 4;
base_Kp = 10; base_Kd = 4;
alpha_p = 1; alpha_d = 0;

% 误差记录
theta_ref = ones(1, n); % 期望值（单位阶跃）
k_p = Kp;
k_d = Kd;
prev = 1;
% 模拟主循环
for i = 2:n
    % 时变增益
    gain = 1;

    % ----------- P 控制器 -----------
    e_P = theta_ref(i) - theta_P(i-1);
    u_P = Kp * e_P;
    alpha_P = gain * u_P;
    omega_P(i) = omega_P(i-1) + alpha_P * dt;
    theta_P(i) = theta_P(i-1) + omega_P(i) * dt;

    % ----------- PD 控制器 -----------
    e_PD = theta_ref(i) - theta_PD(i-1);
    de_PD = -omega_PD(i-1);  % 误差导数近似为 - 角速度
    u_PD = Kp * e_PD + Kd * de_PD;
    alpha_PD = gain * u_PD;
    omega_PD(i) = omega_PD(i-1) + alpha_PD * dt;
    theta_PD(i) = theta_PD(i-1) + omega_PD(i) * dt;

    % ----------- 自适应 PD 控制器 -----------
    e_AD = theta_ref(i) - theta_AD(i-1);
    k_p = k_p + alpha_p * e_AD;
    k_d = k_d + alpha_d * (e_AD - prev)/dt;
    u_AD = k_p * e_AD + k_d * (e_AD - prev)/dt;
    alpha_AD = gain * u_AD;
    omega_AD(i) = omega_AD(i-1) + alpha_AD * dt;
    theta_AD(i) = theta_AD(i-1) + omega_AD(i) * dt;
    prev = e_AD;
end

% ---------- 绘图 ----------
figure;
plot(time, theta_ref, 'k--', 'LineWidth', 1.5); hold on;
plot(time, theta_P, 'r', 'LineWidth', 1.5);
plot(time, theta_PD, 'b', 'LineWidth', 1.5);
plot(time, theta_AD, 'g', 'LineWidth', 1.5);
xlabel('时间 (s)');
ylabel('角度输出');
legend('期望值', 'P控制器', 'PD控制器', '自适应PD控制器');
title('三种控制器在时变系统下的响应对比');
grid on;

% ---------- 计算性能指标 ----------
controllers = {'P', 'PD', 'AD'};
thetas = {theta_P, theta_PD, theta_AD};
omegas = {omega_P, omega_PD, omega_AD};

fprintf('\n控制器性能对比（单位阶跃响应）：\n');
for k = 1:3
    y = thetas{k};
    e = theta_ref - y;
    overshoot = (max(y) - 1.0) * 100;
    settle_time = time(find(abs(y - 1.0) < 0.02, 1));
    ss_error = abs(y(end) - 1.0);
    ise = sum(e.^2) * dt;
    fprintf('%s控制器: 超调 = %.2f%%, 建立时间 = %.2fs, 稳态误差 = %.4f, ISE = %.4f\n', ...
        controllers{k}, overshoot, settle_time, ss_error, ise);
end