#include <Wire.h>
#include <Adafruit_GFX.h>
#include <Adafruit_SSD1306.h>

#define SCREEN_WIDTH 128
#define SCREEN_HEIGHT 64
#define OLED_ADDR 0x3C

Adafruit_SSD1306 display(SCREEN_WIDTH, SCREEN_HEIGHT, &Wire, OLED_ADDR);

// PWM输出引脚
const int pwmPinPositive = 9; // 正向PWM输出口
const int pwmPinNegative = 10; // 负向PWM输出口

void setup() {
  Serial.begin(115200);  // 初始化串口（波特率需与 Python 一致）
  display.begin(SSD1306_SWITCHCAPVCC, OLED_ADDR);
  display.clearDisplay();
  display.setTextSize(2);  // 增大字体
  display.setTextColor(WHITE);

  // 设置PWM引脚为输出模式
  pinMode(pwmPinPositive, OUTPUT);
  pinMode(pwmPinNegative, OUTPUT);

  // 初始关闭PWM输出
  analogWrite(pwmPinPositive, 0);
  analogWrite(pwmPinNegative, 0);
}

void loop() {
  if (Serial.available() > 0) {
    String receivedData = Serial.readStringUntil('\n');  // 读取一行数据
    receivedData.trim();  // 去除换行符

    float a = receivedData.toFloat();  // 转成浮点数

    // 在屏幕上显示接收到的数据
    display.clearDisplay();
    display.setCursor(0, 10);
    display.print("Brightness\n----------\n");
    // display.setCursor(0, 20);
    display.print(a, 3);  // 显示两位小数
    display.display();

    // 控制PWM输出
    if (a > 0) {
      int pwmValue = int(a * 255.0);  // 正数映射到0~255
      pwmValue = constrain(pwmValue, 0, 255);  // 保证在有效范围
      analogWrite(pwmPinPositive, pwmValue);
      analogWrite(pwmPinNegative, 0);  // 另一个口关掉
    }
    else if (a < 0) {
      int pwmValue = int(-a * 255.0);  // 负数映射到0~255
      pwmValue = constrain(pwmValue, 0, 255);
      analogWrite(pwmPinPositive, 0);  // 一个口关掉
      analogWrite(pwmPinNegative, pwmValue);
    }
    else {
      // a == 0，两个PWM口都关掉
      analogWrite(pwmPinPositive, 0);
      analogWrite(pwmPinNegative, 0);
    }
  }
}
