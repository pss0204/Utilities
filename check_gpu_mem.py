import pynvml

def check_gpu_memory():
    """NVIDIA GPU의 메모리 사용량을 체크하고 출력하는 함수"""
    pynvml.nvmlInit()
    handle = pynvml.nvmlDeviceGetHandleByIndex(0)  # 첫 번째 GPU
    info = pynvml.nvmlDeviceGetMemoryInfo(handle)
    
    used_memory_mb = info.used / (1024 * 1024)  # 바이트를 MB로 변환
    total_memory_mb = info.total / (1024 * 1024)
    
    print(f"GPU 메모리 사용량: {used_memory_mb:.2f} MB / {total_memory_mb:.2f} MB ({used_memory_mb/total_memory_mb*100:.2f}%)")
    return used_memory_mb