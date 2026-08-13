"""chapter1 片段 10：多卡 GPU 状态监控与批量管理工具（原文代码）

注意：set_gpu_power_limit 需要 root 权限；monitor_and_adjust 是无限循环。
"""
import subprocess
import time
import datetime
def run_command(command):
    """
    运行系统命令，并返回结果
    """
    try:
        result = subprocess.run(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        if result.returncode != 0:
            print(f"命令执行错误:{result.stderr}")
            return None
        return result.stdout.strip()
    except Exception as e:
        print(f"运行命令失败:{e}")
        return None
def to_float(value):
    """
    修正：部分显卡（如笔记本 GPU）的 power.draw / power.limit 会返回 "[N/A]"，
    原文直接 float() 会抛 ValueError，这里退化为 0.0。
    """
    try:
        return float(value)
    except ValueError:
        return 0.0
def log_to_file(log_message, file_path="gpu_status.log"):
    """
    将日志消息写入文件
    """
    with open(file_path, "a") as log_file:
        log_file.write(f"{datetime.datetime.now()} - {log_message}\n")
def query_all_gpus():
    """
    查询所有GPU的实时状态信息
    """
    print("查询所有GPU状态中...")
    command = ["nvidia-smi", "--query-gpu=index,name,utilization.gpu,temperature.gpu,power.draw,power.limit", "--format=csv,noheader,nounits"]
    output = run_command(command)
    if output:
        gpu_info_list = []
        for line in output.splitlines():
            index, name, utilization, temp, power_draw, power_limit = line.split(", ")
            gpu_info = {
                "index": int(index),
                "name": name,
                "utilization": int(utilization),
                "temperature": int(temp),
                "power_draw": to_float(power_draw),
                "power_limit": to_float(power_limit)
            }
            gpu_info_list.append(gpu_info)
            print(f"GPU{gpu_info['index']}状态:")
            print(f"  名称:{gpu_info['name']}")
            print(f"  利用率:{gpu_info['utilization']}%")
            print(f"  温度:{gpu_info['temperature']}°C")
            print(f"  当前功耗:{gpu_info['power_draw']}W")
            print(f"  功耗限制:{gpu_info['power_limit']}W")
            print("-" * 30)
            log_to_file(f"GPU{gpu_info['index']} 状态: {gpu_info}")
        return gpu_info_list
    else:
        print("无法获取GPU状态")
        return []
def set_gpu_power_limit(gpu_index, power_limit):
    """
    设置指定GPU的功耗限制
    """
    print(f"设置GPU{gpu_index}的功耗限制为{power_limit}W...")
    command = ["nvidia-smi", "-i", str(gpu_index), "-pl", str(power_limit)]
    output = run_command(command)
    if output:
        print(f"GPU{gpu_index}功耗限制设置成功:{output}")
        log_to_file(f"GPU{gpu_index}功耗限制设置为{power_limit}W")
    else:
        print(f"GPU{gpu_index}功耗限制设置失败")
        log_to_file(f"GPU{gpu_index}功耗限制设置失败")
def monitor_and_adjust(gpus, max_temp=75):
    """
    动态监控所有GPU的温度并调整功耗限制
    """
    print("开始动态监控所有GPU的温度和功耗...")
    while True:
        for gpu in gpus:
            gpu_index = gpu["index"]
            command = ["nvidia-smi", "--query-gpu=temperature.gpu", "--format=csv,noheader,nounits", "-i", str(gpu_index)]
            temp = run_command(command)
            if temp is None:
                print(f"无法获取GPU{gpu_index}的温度")
                continue
            temp = int(temp)
            print(f"当前GPU{gpu_index}温度:{temp}°C")
            if temp > max_temp:
                print(f"GPU{gpu_index}温度过高({temp}°C)，降低功耗限制...")
                set_gpu_power_limit(gpu_index, 100)
            else:
                print(f"GPU{gpu_index}温度正常({temp}°C)，恢复默认功耗限制...")
                set_gpu_power_limit(gpu_index, 250)
        time.sleep(10)  # 每10秒监控一次
if __name__ == "__main__":
    print("====GPU状态监控与批量管理工具====")
    # 查询所有GPU状态
    gpus = query_all_gpus()
    # 设置所有GPU功耗限制为200W
    print("批量设置所有GPU功耗限制为200W...")
    for gpu in gpus:
        set_gpu_power_limit(gpu["index"], 200)
    # 开始动态监控和调整
    monitor_and_adjust(gpus)
