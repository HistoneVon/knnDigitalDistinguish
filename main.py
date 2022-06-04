from digitalDistinguish import *
import time

if __name__ == '__main__':  # 声明主函数
    a = time.time()  # 设置起始时间
    distinguish()  # 调用测试函数
    b = time.time() - a  # 计算运行时间
    print("运行时间:", b)  # 输出运行时间
