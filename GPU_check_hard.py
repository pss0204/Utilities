import torch
import platform
import sys
import subprocess
import os
import re
from collections import defaultdict

def check_cuda_cudnn_compatibility():
    print("=" * 50)
    print("시스템 환경 및 CUDA, cuDNN 호환성 검사")
    print("=" * 50)
    
    # 시스템 정보
    print(f"OS: {platform.system()} {platform.release()}")
    print(f"Python 버전: {platform.python_version()}")
    print(f"PyTorch 버전: {torch.__version__}")
    
    # CUDA 지원 여부
    cuda_available = torch.cuda.is_available()
    print(f"CUDA 사용 가능 여부: {cuda_available}")
    
    if not cuda_available:
        print("CUDA를 사용할 수 없습니다. GPU 드라이버를 확인하세요.")
        return False
    
    # CUDA 버전 정보
    cuda_version = torch.version.cuda
    print(f"CUDA 버전(PyTorch): {cuda_version}")
    
    try:
        # nvcc를 통한 CUDA 버전 확인 시도
        nvcc_output = subprocess.check_output(["nvcc", "--version"]).decode("utf-8")
        nvcc_cuda_version = re.search(r"release (\d+\.\d+)", nvcc_output).group(1)
        print(f"CUDA 버전(nvcc): {nvcc_cuda_version}")
        
        # CUDA 버전 일치 여부
        if cuda_version != nvcc_cuda_version:
            print(f"⚠️ 경고: PyTorch CUDA 버전({cuda_version})과 시스템 CUDA 버전({nvcc_cuda_version})이 일치하지 않습니다.")
    except (subprocess.SubprocessError, FileNotFoundError):
        print("nvcc를 찾을 수 없습니다. CUDA 툴킷이 올바르게 설치되어 있는지 확인하세요.")
    
    # cuDNN 버전
    if hasattr(torch.backends.cudnn, "version"):
        cudnn_version = torch.backends.cudnn.version()
        print(f"cuDNN 버전: {cudnn_version}")
    else:
        print("cuDNN 버전을 확인할 수 없습니다.")
    
    # GPU 정보
    gpu_count = torch.cuda.device_count()
    print(f"사용 가능한 GPU 개수: {gpu_count}")
    
    if gpu_count == 0:
        print("GPU를 찾을 수 없습니다.")
        return False
    
    # 각 GPU 정보 출력
    for i in range(gpu_count):
        print(f"\n--- GPU {i} 정보 ---")
        print(f"이름: {torch.cuda.get_device_name(i)}")
        print(f"총 메모리: {torch.cuda.get_device_properties(i).total_memory / 1024**3:.2f} GB")
        print(f"현재 할당된 메모리: {torch.cuda.memory_allocated(i) / 1024**3:.2f} GB")
        print(f"캐시된 메모리: {torch.cuda.memory_reserved(i) / 1024**3:.2f} GB")
        print(f"계산 능력(Compute Capability): {torch.cuda.get_device_capability(i)}")
    
    # CUDA 기능 검사
    print("\n--- CUDA 기능 검사 ---")
    print(f"cuDNN 사용 가능: {torch.backends.cudnn.is_available()}")
    print(f"cuDNN 활성화 상태: {torch.backends.cudnn.enabled}")
    print(f"cuDNN 벤치마크 모드: {torch.backends.cudnn.benchmark}")
    print(f"cuDNN 결정론적 모드: {torch.backends.cudnn.deterministic}")
    
    # 간단한 텐서 연산 테스트
    print("\n--- GPU 동작 테스트 ---")
    try:
        x = torch.randn(1000, 1000).cuda()
        y = torch.randn(1000, 1000).cuda()
        start_time = torch.cuda.Event(enable_timing=True)
        end_time = torch.cuda.Event(enable_timing=True)
        
        start_time.record()
        z = torch.matmul(x, y)
        end_time.record()
        
        # 결과가 계산될 때까지 대기
        torch.cuda.synchronize()
        
        print(f"행렬 곱셈(1000x1000) 테스트 결과: {start_time.elapsed_time(end_time):.2f} ms")
        print("GPU 연산 테스트 통과!")
    except Exception as e:
        print(f"GPU 테스트 실패: {str(e)}")
        return False
    
    # PyTorch 빌드 정보
    print("\n--- PyTorch 빌드 정보 ---")
    build_info = defaultdict(str)
    
    for name in dir(torch.version):
        if not name.startswith('_'):
            value = getattr(torch.version, name)
            if value:
                build_info[name] = value
    
    for key, value in build_info.items():
        print(f"{key}: {value}")
    
    print("\n호환성 검사 완료!")
    return True

# 호환성 검사 실행
compatibility_result = check_cuda_cudnn_compatibility()
print(f"\n총평: {'호환성 검사 통과' if compatibility_result else '호환성 문제 발견'}")
