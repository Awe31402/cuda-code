"""chapter1 片段 9：单卡 GPU 状态查询与功耗调节工具（原文代码）

注意：set_power_limit 需要 root 权限；monitor_and_optimize 是无限循环。
"""
import shlex
import subprocess
import time
def run_command(command):
    """
    运行系统命令，并返回结果
    """
    try:
        result = subprocess.run(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        if result.returncode != 0:
            print(f"命令执行错误: {result.stderr}")
            return None
        return result.stdout.strip()
    except Exception as e:
        print(f"运行命令失败: {e}")
        return None
def query_gpu_status():
    """
    查询 GPU 的实时状态信息
    """
    print("查询 GPU 状态中...")
    command = ["nvidia-smi", "--query-gpu=index,name,utilization.gpu,temperature.gpu,power.draw,power.limit", "--format=csv,noheader,nounits"]
    print(shlex.join(command))
    output = run_command(command)
    if output:
        print("当前 GPU 状态:")
        for line in output.splitlines():
            index, name, utilization, temp, power_draw, power_limit = line.split(", ")
            print(f"GPU {index}:")
            print(f"  名称: {name}")
            print(f"  利用率: {utilization}%")
            print(f"  温度: {temp}°C")
            print(f"  当前功耗: {power_draw} W")
            print(f"  功耗限制: {power_limit} W")
            print("-" * 30)
def set_power_limit(gpu_index, power_limit):
    """
    设置 GPU 的功耗限制
    """
    print(f"设置 GPU {gpu_index} 的功耗限制为 {power_limit} W...")
    command = ["nvidia-smi", "-i", str(gpu_index), "-pl", str(power_limit)]
    output = run_command(command)
    if output:
        print(f"功耗限制设置成功: {output}")
    else:
        print("设置功耗限制失败")
def monitor_and_optimize(gpu_index, max_temp=75):
    """
    动态监控 GPU 温度并调整功耗限制
    """
    print(f"开始动态监控 GPU {gpu_index} 的温度和功耗...")
    while True:
        # 查询温度
        command = ["nvidia-smi", "--query-gpu=temperature.gpu", "--format=csv,noheader,nounits", "-i", str(gpu_index)]
        temp = run_command(command)
        if temp is None:
            print("无法获取 GPU 温度")
            break
        temp = int(temp)
        print(f"当前 GPU {gpu_index} 温度: {temp}°C")
        # 根据温度调整功耗限制
        if temp > max_temp:
            print(f"温度过高 ({temp}°C)，降低功耗限制...")
            set_power_limit(gpu_index, 100)  # 设置较低功耗限制
        else:
            print(f"温度正常 ({temp}°C)，恢复默认功耗限制...")
            set_power_limit(gpu_index, 140)  # 恢复默认功耗限制
        time.sleep(5)  # 每 5 秒监控一次
if __name__ == "__main__":
    print("==== GPU 状态查询与优化工具 ====")
    query_gpu_status()
    # 设置功耗限制示例
    gpu_index = 0  # 假设目标 GPU 为索引 0
    set_power_limit(gpu_index, 100)
    # 开始动态监控和优化
    monitor_and_optimize(gpu_index)
