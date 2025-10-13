#!/usr/bin/env python3
"""
ChronoTick Configuration Selector

Helps choose the optimal configuration based on your hardware and requirements.
"""

import argparse
import os
import sys
from pathlib import Path
import yaml
import subprocess
import shutil

def detect_hardware():
    """Detect available hardware capabilities."""
    has_gpu = False
    gpu_memory = 0
    cpu_cores = os.cpu_count()
    
    # Try to detect GPU
    try:
        import torch
        if torch.cuda.is_available():
            has_gpu = True
            gpu_memory = torch.cuda.get_device_properties(0).total_memory // (1024**3)  # GB
    except ImportError:
        pass
    
    # Try nvidia-smi as backup
    if not has_gpu:
        try:
            result = subprocess.run(['nvidia-smi', '--query-gpu=memory.total', '--format=csv,noheader,nounits'], 
                                  capture_output=True, text=True)
            if result.returncode == 0:
                has_gpu = True
                gpu_memory = int(result.stdout.strip()) // 1024  # Convert MB to GB
        except (subprocess.SubprocessError, FileNotFoundError):
            pass
    
    return {
        'has_gpu': has_gpu,
        'gpu_memory_gb': gpu_memory,
        'cpu_cores': cpu_cores
    }

def get_available_configs():
    """Get list of available configuration files."""
    config_dir = Path(__file__).parent / "configs"
    configs = {}
    
    if config_dir.exists():
        for config_file in config_dir.glob("*.yaml"):
            name = config_file.stem
            configs[name] = {
                'file': config_file,
                'description': get_config_description(config_file)
            }
    
    return configs

def get_config_description(config_file):
    """Extract description from config file."""
    try:
        with open(config_file, 'r') as f:
            first_line = f.readline().strip()
            if first_line.startswith('#'):
                return first_line[1:].strip()
    except Exception:
        pass
    return "No description available"

def recommend_config(hardware):
    """Recommend configuration based on hardware using proper GPU+CPU combinations."""
    configs = get_available_configs()
    
    if not configs:
        return None, "No configurations found"
    
    # Correct logic: GPU + CPU model combinations for optimal hardware utilization
    if hardware['has_gpu']:
        # GPU available: Use hybrid GPU+CPU setup for best performance
        if hardware['gpu_memory_gb'] >= 8:
            # High-end GPU: TimesFM (GPU) + Chronos (CPU) - your preferred default
            if 'hybrid_timesfm_chronos' in configs:
                return 'hybrid_timesfm_chronos', "Optimal GPU+CPU: TimesFM (GPU) + Chronos (CPU)"
            elif 'gpu_only_timesfm' in configs:
                return 'gpu_only_timesfm', "High-end GPU with TimesFM"
        
        elif hardware['gpu_memory_gb'] >= 4:
            # Mid-range GPU: Toto (GPU) + Chronos (CPU) - efficient combination
            if 'hybrid_toto_chronos' in configs:
                return 'hybrid_toto_chronos', "Efficient GPU+CPU: Toto (GPU) + Chronos (CPU)"
            elif 'gpu_only_toto' in configs:
                return 'gpu_only_toto', "Mid-range GPU with efficient Toto"
        
        else:
            # Low VRAM GPU: fallback to CPU only
            if 'cpu_only_chronos' in configs:
                return 'cpu_only_chronos', "Low GPU VRAM, using CPU with Chronos"
    
    # CPU only: Single CPU model (your preferred default)
    if hardware['cpu_cores'] >= 8:
        if 'cpu_only_chronos' in configs:
            return 'cpu_only_chronos', "Multi-core CPU with Chronos (your preferred default)"
        elif 'cpu_only_ttm' in configs:
            return 'cpu_only_ttm', "Multi-core CPU with TTM"
    
    else:
        # Limited CPU: stick with efficient Chronos
        if 'cpu_only_chronos' in configs:
            return 'cpu_only_chronos', "Basic CPU with Chronos"
    
    # Fallback to first available
    first_config = next(iter(configs.keys()))
    return first_config, f"Default fallback: {configs[first_config]['description']}"

def copy_config(config_name, target_path=None):
    """Copy selected configuration to the main config location."""
    configs = get_available_configs()
    
    if config_name not in configs:
        raise ValueError(f"Configuration '{config_name}' not found")
    
    if target_path is None:
        target_path = Path(__file__).parent / "config.yaml"
    
    source_path = configs[config_name]['file']
    shutil.copy2(source_path, target_path)
    
    return target_path

def validate_config(config_file):
    """Validate configuration file."""
    try:
        with open(config_file, 'r') as f:
            config = yaml.safe_load(f)
        
        # Basic validation
        required_sections = ['short_term', 'long_term', 'fusion', 'performance', 'clock']
        for section in required_sections:
            if section not in config:
                return False, f"Missing required section: {section}"
        
        return True, "Configuration is valid"
    
    except Exception as e:
        return False, f"Configuration error: {e}"

def main():
    parser = argparse.ArgumentParser(description="ChronoTick Configuration Selector")
    parser.add_argument('--list', action='store_true', help='List available configurations')
    parser.add_argument('--recommend', action='store_true', help='Recommend configuration based on hardware')
    parser.add_argument('--select', type=str, help='Select a specific configuration')
    parser.add_argument('--validate', type=str, help='Validate a configuration file')
    parser.add_argument('--hardware', action='store_true', help='Show detected hardware information')
    parser.add_argument('--output', type=str, help='Output path for selected configuration')
    
    args = parser.parse_args()
    
    if args.hardware:
        hardware = detect_hardware()
        print("Detected Hardware:")
        print(f"  CPU Cores: {hardware['cpu_cores']}")
        print(f"  GPU Available: {hardware['has_gpu']}")
        if hardware['has_gpu']:
            print(f"  GPU Memory: {hardware['gpu_memory_gb']} GB")
        return
    
    if args.list:
        configs = get_available_configs()
        print("Available Configurations:")
        print("-" * 50)
        for name, info in configs.items():
            print(f"  {name}")
            print(f"    {info['description']}")
            print()
        return
    
    if args.recommend:
        hardware = detect_hardware()
        config_name, reason = recommend_config(hardware)
        
        print("Hardware Detection:")
        print(f"  CPU Cores: {hardware['cpu_cores']}")
        print(f"  GPU Available: {hardware['has_gpu']}")
        if hardware['has_gpu']:
            print(f"  GPU Memory: {hardware['gpu_memory_gb']} GB")
        print()
        
        if config_name:
            print(f"Recommended Configuration: {config_name}")
            print(f"Reason: {reason}")
            print()
            print("To use this configuration, run:")
            print(f"  python config_selector.py --select {config_name}")
        else:
            print("No suitable configuration found")
        return
    
    if args.select:
        try:
            target_path = args.output or (Path(__file__).parent / "config.yaml")
            copied_path = copy_config(args.select, target_path)
            
            print(f"Configuration '{args.select}' copied to {copied_path}")
            
            # Validate the copied configuration
            is_valid, message = validate_config(copied_path)
            if is_valid:
                print(f"✓ {message}")
            else:
                print(f"⚠ {message}")
            
        except Exception as e:
            print(f"Error: {e}")
            sys.exit(1)
        return
    
    if args.validate:
        is_valid, message = validate_config(args.validate)
        if is_valid:
            print(f"✓ {message}")
        else:
            print(f"✗ {message}")
            sys.exit(1)
        return
    
    # Default: show help
    parser.print_help()

if __name__ == "__main__":
    main()