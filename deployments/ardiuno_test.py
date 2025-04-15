import serial
import time

# 配置串口（端口和波特率需与 Arduino 一致）
ser = serial.Serial('COM3', 9600, timeout=1)  # Windows 端口示例（如 COM3）
# ser = serial.Serial('/dev/ttyUSB0', 9600)   # Linux 端口示例
# ser = serial.Serial('/dev/cu.usbmodem14101', 9600)  # macOS 端口示例

value = 42

try:
    while True:
        # 模拟实时输出值（替换为你的实际数据）
        # data_to_send = f"Value: {value}\n"  # 添加换行符作为结束标记
        data_to_send = input()
        ser.write(data_to_send.encode())  # 发送数据到 Arduino
        print(f"Sent: {data_to_send.strip()}")  # 打印发送内容（可选）
        time.sleep(1)  # 发送间隔（根据需求调整）
        value += 1

except KeyboardInterrupt:
    ser.close()  # 关闭串口
    print("Serial connection closed.")