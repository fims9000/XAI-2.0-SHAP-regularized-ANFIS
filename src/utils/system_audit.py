"""
Модуль для аудита системы и железа
"""
import platform
import psutil
import torch
import numpy as np
import sys
from datetime import datetime
import json

class SystemAuditor:
    """Класс для аудита системы и железа"""
    
    def __init__(self):
        self.audit_data = {}
    
    def run_full_audit(self):
        """Полный аудит системы"""
        print("[INFO] Запуск аудита системы...")
        
        self.audit_data = {
            'timestamp': datetime.now().isoformat(),
            'system': self._get_system_info(),
            'hardware': self._get_hardware_info(),
            'python': self._get_python_info(),
            'libraries': self._get_library_versions(),
            'gpu': self._get_gpu_info(),
            'memory': self._get_memory_info(),
            'cpu': self._get_cpu_info()
        }
        
        return self.audit_data
    
    def _get_system_info(self):
        """Информация о системе"""
        return {
            'platform': platform.platform(),
            'system': platform.system(),
            'release': platform.release(),
            'version': platform.version(),
            'machine': platform.machine(),
            'processor': platform.processor()
        }
    
    def _get_hardware_info(self):
        """Информация о железе"""
        return {
            'cpu_count_physical': psutil.cpu_count(logical=False),
            'cpu_count_logical': psutil.cpu_count(logical=True),
            'cpu_freq': {
                'current': psutil.cpu_freq().current if psutil.cpu_freq() else None,
                'min': psutil.cpu_freq().min if psutil.cpu_freq() else None,
                'max': psutil.cpu_freq().max if psutil.cpu_freq() else None
            },
            'memory_total_gb': psutil.virtual_memory().total / (1024**3),
            'memory_available_gb': psutil.virtual_memory().available / (1024**3),
            'disk_usage': {
                'total_gb': psutil.disk_usage('/').total / (1024**3),
                'used_gb': psutil.disk_usage('/').used / (1024**3),
                'free_gb': psutil.disk_usage('/').free / (1024**3)
            }
        }
    
    def _get_python_info(self):
        """Информация о Python"""
        return {
            'version': sys.version,
            'version_info': {
                'major': sys.version_info.major,
                'minor': sys.version_info.minor,
                'micro': sys.version_info.micro
            },
            'executable': sys.executable
        }
    
    def _get_library_versions(self):
        """Версии библиотек"""
        versions = {}
        libraries = [
            'numpy', 'pandas', 'scikit-learn', 'matplotlib', 
            'seaborn', 'torch', 'shap', 'PyYAML'
        ]
        
        for lib in libraries:
            try:
                module = __import__(lib)
                if hasattr(module, '__version__'):
                    versions[lib] = module.__version__
                elif lib == 'PyYAML':
                    import yaml
                    versions[lib] = yaml.__version__
            except ImportError:
                versions[lib] = 'not installed'
            except Exception as e:
                versions[lib] = f'error: {str(e)}'
        
        return versions
    
    def _get_gpu_info(self):
        """Информация о GPU"""
        gpu_info = {
            'available': torch.cuda.is_available(),
            'count': torch.cuda.device_count() if torch.cuda.is_available() else 0,
            'devices': []
        }
        
        if torch.cuda.is_available():
            for i in range(torch.cuda.device_count()):
                device_info = {
                    'device_id': i,
                    'name': torch.cuda.get_device_name(i),
                    'memory_total_gb': torch.cuda.get_device_properties(i).total_memory / (1024**3),
                    'compute_capability': f"{torch.cuda.get_device_properties(i).major}.{torch.cuda.get_device_properties(i).minor}"
                }
                gpu_info['devices'].append(device_info)
        else:
            gpu_info['devices'].append({
                'device_id': 0,
                'name': 'CPU only',
                'memory_total_gb': None,
                'compute_capability': None
            })
        
        return gpu_info
    
    def _get_memory_info(self):
        """Текущее использование памяти"""
        memory = psutil.virtual_memory()
        return {
            'total_gb': memory.total / (1024**3),
            'available_gb': memory.available / (1024**3),
            'used_gb': memory.used / (1024**3),
            'percent': memory.percent,
            'free_gb': memory.free / (1024**3)
        }
    
    def _get_cpu_info(self):
        """Информация о CPU"""
        return {
            'usage_percent': psutil.cpu_percent(interval=1),
            'per_cpu_usage': psutil.cpu_percent(interval=1, percpu=True),
            'load_average': psutil.getloadavg() if hasattr(psutil, 'getloadavg') else None
        }
    
    def print_audit_report(self):
        """Вывод отчета об аудите"""
        print("\n" + "="*60)
        print("ОТЧЕТ ОБ АУДИТЕ СИСТЕМЫ")
        print("="*60)
        
        print(f"\n📅 Время аудита: {self.audit_data['timestamp']}")
        
        print("\n[INFO] СИСТЕМА:")
        sys_info = self.audit_data['system']
        print(f"   Платформа: {sys_info['platform']}")
        print(f"   Система: {sys_info['system']} {sys_info['release']}")
        print(f"   Процессор: {sys_info['processor']}")
        
        print("\n[ЖЕЛЕЗО]")
        hw_info = self.audit_data['hardware']
        print(f"   CPU (физических/логических): {hw_info['cpu_count_physical']}/{hw_info['cpu_count_logical']}")
        if hw_info['cpu_freq']['current']:
            print(f"   Частота CPU: {hw_info['cpu_freq']['current']:.2f} MHz")
        print(f"   Память: {hw_info['memory_total_gb']:.2f} GB (доступно: {hw_info['memory_available_gb']:.2f} GB)")
        print(f"   Диск: {hw_info['disk_usage']['total_gb']:.2f} GB (свободно: {hw_info['disk_usage']['free_gb']:.2f} GB)")
        
        print("\n[PYTHON]")
        py_info = self.audit_data['python']
        print(f"   Версия: {py_info['version_info']['major']}.{py_info['version_info']['minor']}.{py_info['version_info']['micro']}")
        print(f"   Исполняемый файл: {py_info['executable']}")
        
        print("\n[БИБЛИОТЕКИ]")
        for lib, version in self.audit_data['libraries'].items():
            print(f"   {lib}: {version}")
        
        print("\n[GPU]")
        gpu_info = self.audit_data['gpu']
        print(f"   Доступен: {'Да' if gpu_info['available'] else 'Нет'}")
        print(f"   Количество устройств: {gpu_info['count']}")
        for device in gpu_info['devices']:
            print(f"   Устройство {device['device_id']}: {device['name']}")
            if device['memory_total_gb']:
                print(f"      Память: {device['memory_total_gb']:.2f} GB")
                print(f"      Compute Capability: {device['compute_capability']}")
        
        print("\n[ПАМЯТЬ] Текущее состояние:")
        mem_info = self.audit_data['memory']
        print(f"   Использовано: {mem_info['used_gb']:.2f} GB ({mem_info['percent']:.1f}%)")
        print(f"   Свободно: {mem_info['free_gb']:.2f} GB")
        
        print("\n[INFO] CPU (текущее состояние):")
        cpu_info = self.audit_data['cpu']
        print(f"   Использование: {cpu_info['usage_percent']:.1f}%")
        
        print("="*60 + "\n")
    
    def save_audit_report(self, filepath):
        """Сохранение отчета в JSON"""
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(self.audit_data, f, indent=2, ensure_ascii=False)
        print(f"[OK] Отчет сохранен в: {filepath}")

