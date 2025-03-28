import time
import csv
import GPUtil

log_file = 'gpu_memory_log.csv'

# CSV 파일에 헤더 작성
with open(log_file, mode='w', newline='') as f:
    writer = csv.writer(f)
    writer.writerow(['timestamp', 'gpu_id', 'name', 'memory_used_MB', 'memory_total_MB'])

while True:
    timestamp = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
    gpus = GPUtil.getGPUs()
    with open(log_file, mode='a', newline='') as f:
        writer = csv.writer(f)
        for gpu in gpus:
            writer.writerow([timestamp, gpu.id, gpu.name, gpu.memoryUsed, gpu.memoryTotal])
    time.sleep(5)
