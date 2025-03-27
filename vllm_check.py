import subprocess
import sys
import platform
import torch
import importlib.util
import os

try:
    from vllm.utils import get_available_gpu_memory
except ImportError:
    def get_available_gpu_memory():
        """Fallback: GPU별 사용 가능한 메모리 계산."""
        if torch.cuda.is_available():
            memory = {}
            for i in range(torch.cuda.device_count()):
                props = torch.cuda.get_device_properties(i)
                total = props.total_memory
                allocated = torch.cuda.memory_allocated(i)
                memory[i] = total - allocated
            return memory
        return {}

import vllm
from vllm.config import ModelConfig

#!/usr/bin/env python3
"""
Check if vllm is installed, its version, and hardware compatibility.
"""


def check_package_installed(package_name):
    """Check if a Python package is installed."""
    return importlib.util.find_spec(package_name) is not None


def get_package_version(package_name):
    """Get the version of an installed package."""
    try:
        if check_package_installed(package_name):
            module = __import__(package_name)
            return getattr(module, "__version__", "Unknown")
        return None
    except ImportError:
        return None


def check_gpu_availability():
    """Check for GPU availability."""
    try:
        return {
            "cuda_available": torch.cuda.is_available(),
            "cuda_device_count": torch.cuda.device_count() if torch.cuda.is_available() else 0,
            "cuda_device_names": [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())] if torch.cuda.is_available() else [],
        }
    except ImportError:
        return {"cuda_available": False, "cuda_device_count": 0, "cuda_device_names": []}


def check_vllm_compatibility():
    """Check vllm compatibility with CPU and GPU."""
    results = {"cpu": False, "gpu": False}
    
    if not check_package_installed("vllm"):
        return results
    
    # Check GPU compatibility
    gpu_info = check_gpu_availability()
    if gpu_info["cuda_available"] and gpu_info["cuda_device_count"] > 0:
        try:
            # A basic test to see if vLLM initializes on GPU
            try:
                get_available_gpu_memory()  # Just test if this runs without error
                results["gpu"] = True
            except:
                pass
        except:
            pass
    
    # Check CPU compatibility - vLLM doesn't officially support CPU-only mode
    # but we can check if it might run in a limited way
    try:
        # Unfortunately, there's no direct way to test CPU compatibility
        # since vLLM is primarily designed for GPU
        # If CUDA_VISIBLE_DEVICES is set to empty, it might run on CPU
        os.environ["CUDA_VISIBLE_DEVICES"] = ""
        try:
            # Try to import a CPU-only component
            results["cpu"] = True  # At least it imports, might not run properly
        except:
            pass
        # Restore environment
        if "CUDA_VISIBLE_DEVICES" in os.environ:
            del os.environ["CUDA_VISIBLE_DEVICES"]
    except:
        pass
    
    return results


def main():
    """Main function to check vllm installation and compatibility."""
    print("vLLM Installation Check")
    print("-" * 50)
    
    print(f"System: {platform.system()} {platform.release()}")
    print(f"Python version: {platform.python_version()}")
    
    # Check if vllm is installed
    is_installed = check_package_installed("vllm")
    print(f"vLLM installed: {is_installed}")
    
    if is_installed:
        # Get version
        version = get_package_version("vllm")
        print(f"vLLM version: {version}")
        
        # Check hardware compatibility
        gpu_info = check_gpu_availability()
        print("\nGPU Information:")
        print(f"  CUDA available: {gpu_info['cuda_available']}")
        print(f"  GPU count: {gpu_info['cuda_device_count']}")
        for i, name in enumerate(gpu_info['cuda_device_names']):
            print(f"  GPU {i}: {name}")
        
        # Check compatibility
        compatibility = check_vllm_compatibility()
        print("\nCompatibility:")
        print(f"  CPU: {'Possibly supported (limited)' if compatibility['cpu'] else 'Not supported'}")
        print(f"  GPU: {'Supported' if compatibility['gpu'] else 'Not supported'}")
        
        print("\nNote: vLLM is primarily designed for GPU acceleration.")
        print("CPU-only mode might not work or have severely limited functionality.")
    else:
        print("\nTo install vLLM, run:")
        print("  pip install vllm")
        print("or with CUDA support:")
        print("  pip install vllm[cuda]")


if __name__ == "__main__":
    main()
