import os
import sys
import yaml
import random
import warnings
from pathlib import Path
import platform
import multiprocessing

os.environ['PYTHONHASHSEED'] = '0'

if platform.system() == 'Windows':
    multiprocessing.set_start_method('spawn', force=True)
    os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
    os.environ['OMP_NUM_THREADS'] = '1'
    os.environ['MKL_NUM_THREADS'] = '1'
    os.environ['NUMEXPR_NUM_THREADS'] = '1'
    os.environ['OPENBLAS_NUM_THREADS'] = '1'
    os.environ['NUMPY_MADVISE_HUGEPAGE'] = '0'
    if hasattr(os, 'add_dll_directory'):
        try:
            conda_env = os.environ.get('CONDA_PREFIX', '')
            if conda_env:
                os.add_dll_directory(os.path.join(conda_env, 'Library', 'bin'))
                os.add_dll_directory(os.path.join(conda_env, 'DLLs'))
        except Exception:
            pass

try:
    import numpy as np
except ImportError as e:
    print(f"[ERROR] NumPy导入失败: {e}")
    print("[INFO] 请尝试重新安装NumPy: pip uninstall numpy && pip install numpy==1.21.6")
    sys.exit(1)

try:
    import torch
except ImportError as e:
    print(f"[ERROR] PyTorch导入失败: {e}")
    print("[INFO] 请检查PyTorch安装是否正确")
    sys.exit(1)

try:
    from ultralytics import YOLO
except ImportError as e:
    print(f"[ERROR] Ultralytics导入失败: {e}")
    print("[INFO] 请安装ultralytics: pip install ultralytics")
    sys.exit(1)

warnings.filterwarnings("ignore")
warnings.filterwarnings("ignore", category=UserWarning, module="ultralytics")
warnings.filterwarnings("ignore", category=FutureWarning)

def initialize_environment():
    print("[INFO] 正在初始化环境...")
    
    try:
        test_array = np.array([1, 2, 3])
        print(f"[INFO] NumPy版本: {np.__version__}")
        print("[INFO] NumPy工作正常")
    except Exception as e:
        print(f"[ERROR] NumPy测试失败: {e}")
        return False
    
    try:
        test_tensor = torch.tensor([1, 2, 3])
        print(f"[INFO] PyTorch版本: {torch.__version__}")
        print(f"[INFO] CUDA可用: {torch.cuda.is_available()}")
        if torch.cuda.is_available():
            print(f"[INFO] CUDA版本: {torch.version.cuda}")
        print("[INFO] PyTorch工作正常")
    except Exception as e:
        print(f"[ERROR] PyTorch测试失败: {e}")
        return False
    
    print("[INFO] 环境初始化完成")
    return True
class CFG:
    DEBUG = True 
    FRACTION = 0.3  
    SEED = 88
    CLASSES = [
        'Hardhat', 'Mask', 'NO-Hardhat', 'NO-Mask',
        'NO-Safety Vest', 'Person', 'Safety Cone',
        'Safety Vest', 'machinery', 'vehicle'
    ]
    NUM_CLASSES_TO_TRAIN = len(CLASSES)
    EPOCHS = 3
    BATCH_SIZE = 220
    BASE_MODEL = r'/root/gpufree-data/1005_report/YOLO_Retraining/yolov8n.pt'
    MODEL_WEIGHTS_NAME = 'SafetyHelmetWearing.pt' 
    EXP_NAME = f'css_ppe_fast'
    DATA_YAML_PATH = r'/root/gpufree-data/1005_report/YOLO_Retraining/data.yaml'
    CSS_DATA_PATH = r'/root/gpufree-data/1005_report/YOLO_Retraining/css-data'
    OUTPUT_DIR = r'/root/gpufree-data/1005_report/YOLO_Retraining/runs'
    DEVICE = 0 if torch.cuda.is_available() else 'cpu'
    CLEAN_PREVIOUS_RUNS = True
def set_seed(seed=CFG.SEED):
    random.seed(seed)
    np.random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.benchmark = True
        torch.backends.cudnn.deterministic = False
        torch.cuda.empty_cache()

def apply_mixup(images, labels, alpha=1.0):
    if alpha <= 0:
        return None
    lam = np.random.beta(alpha,alpha)
    batch_size = images.size(0)
    index = torch.randperm(batch_size)
    mixed_images = lam*images+(1-lam)*images[index, :]
    mixed_labels = lam*labels+(1-lam)*labels[index, :]
    return mixed_images,mixed_labels

def apply_cutmix(images, labels, alpha=1.0):
    if alpha <= 0:
        return None
    lam = np.random.beta(alpha, alpha)
    batch_size = images.size(0)
    index = torch.randperm(batch_size)
    h, w = images.size()[2:]
    cx = np.random.uniform(0, w)
    cy = np.random.uniform(0, h)
    w = int(w * np.sqrt(1. - lam))
    h = int(h * np.sqrt(1. - lam))
    x0 = int(np.round(max(cx - w / 2, 0)))
    x1 = int(np.round(min(cx + w / 2, w)))
    y0 = int(np.round(max(cy - h / 2, 0)))
    y1 = int(np.round(min(cy + h / 2, h)))
    mixed_images = images.clone()
    mixed_images[:, :, y0:y1, x0:x1] = images[index, :, y0:y1, x0:x1]
    mixed_labels = lam * labels + (1 - lam) * labels[index, :]
    return mixed_labels

def apply_mosaic(images, labels):
    if len(images) < 4:
        return None
    indices = torch.randperm(len(images))[:4]
    h, w = images.size()[2:]
    half_h, half_w = h // 2, w // 2
    return indices,half_h,half_w

def apply_random_erasing(images, p=0.5, sl=0.02, sh=0.4, r1=0.3):
    if np.random.rand() > p:
        return None 
    h, w = images.size()[2:]
    area = h * w
    target_area = np.random.uniform(sl, sh) * area
    aspect_ratio = np.random.uniform(r1, 1/r1)
    return target_area,aspect_ratio
def cleanup_previous_training():
    output_path = Path(r'/root/gpufree-data/1005_report/YOLO_Retraining/runs') / CFG.EXP_NAME
    if output_path.exists():
        print(f"[INFO] 清理之前的训练结果: {output_path}")
        import shutil
        shutil.rmtree(output_path)
        print("[INFO] 清理完成")

def train_model():
    print(f"[INFO] 开始训练 {CFG.BASE_MODEL} 模型...")
    print(f"[INFO] 使用设备: {CFG.DEVICE}")
    print(f"[INFO] PyTorch版本: {torch.__version__}")
    if torch.cuda.is_available():
        print(f"[INFO] CUDA版本: {torch.version.cuda}")
        print(f"[INFO] GPU数量: {torch.cuda.device_count()}")
        print(f"[INFO] 当前GPU: {torch.cuda.get_device_name()}")
    
    if CFG.CLEAN_PREVIOUS_RUNS:
        cleanup_previous_training()
    
    output_dir = r'/root/gpufree-data/1005_report/YOLO_Retraining/runs'
    os.makedirs(output_dir, exist_ok=True)
    
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    
    print("[INFO] 正在加载预训练模型...")
    model = YOLO(CFG.BASE_MODEL)
    print("[INFO] 预训练模型加载完成")
    
    print("[INFO] 检查数据文件...")
    data_yaml_path = Path(CFG.DATA_YAML_PATH)
    if not data_yaml_path.exists():
        raise FileNotFoundError(f"数据配置文件不存在: {CFG.DATA_YAML_PATH}")
    print("[INFO] 数据文件检查完成")
    train_args = {
        'data': CFG.DATA_YAML_PATH,
        'epochs': CFG.EPOCHS,
        'batch': CFG.BATCH_SIZE,
        'imgsz': 320,
        'device': CFG.DEVICE,
        'workers': min(2, multiprocessing.cpu_count()) if platform.system() == 'Windows' else 4,
        'project': r'/root/gpufree-data/1005_report/YOLO_Retraining',
        'name': f'runs/{CFG.EXP_NAME}',
        'cache': 'disk',
        'single_cls': False,
        'rect': False,
        'cos_lr': False,
        'close_mosaic': 0, 
        'amp': False, 
        'plots': False, 
        'val': True, 
        'save': True, 
        'save_period': -1,
        'exist_ok': True,
        'verbose': True,
        'patience': 10,
        'lr0': 0.01,
        'momentum': 0.937,
        'optimizer': 'SGD',
    }
    
    print("[INFO] 正在启动训练...")
    print(f"[INFO] 训练参数: epochs={CFG.EPOCHS}, batch_size={CFG.BATCH_SIZE}, imgsz=320")
    print("[INFO] 数据加载和缓存可能需要一些时间，请耐心等待...")
    
    try:
        results = model.train(**train_args)
        
        default_weights_path = Path(r'/root/gpufree-data/1005_report/YOLO_Retraining/runs') / CFG.EXP_NAME / 'weights' / 'best.pt'
        custom_weights_path = Path(r'/root/gpufree-data/1005_report/YOLO_Retraining/runs') / CFG.EXP_NAME / 'weights' / CFG.MODEL_WEIGHTS_NAME
        
        if default_weights_path.exists():
            if custom_weights_path.exists():
                print(f"[INFO] 删除已存在的权重文件: {custom_weights_path}")
                custom_weights_path.unlink()
            
            import shutil
            shutil.copy2(default_weights_path, custom_weights_path)
            print(f"[INFO] 模型权重已保存为: {custom_weights_path}")
        else:
            print(f"[WARNING] 未找到训练生成的权重文件: {default_weights_path}")
        
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        print("[INFO] 训练完成!")
        return results
    except RuntimeError as e:
        if "out of memory" in str(e).lower():
            print(f"[ERROR] GPU内存不足: {str(e)}")
            print("[INFO] 建议: 减少batch_size或使用更小的模型")
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        else:
            print(f"[ERROR] 运行时错误: {str(e)}")
        raise
    except Exception as e:
        print(f"[ERROR] 训练过程中出错: {str(e)}")
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        raise

def main():
    print("="*60)
    print("Construction Site Safety PPE Detection - Fast YOLO Training")
    print("="*60)
    
    if not initialize_environment():
        print("[ERROR] 环境初始化失败，程序退出")
        return
    
    set_seed()
    
    results = train_model()
    
    print("="*60)
    print("训练流程完成!")
    print("="*60)

if __name__ == "__main__":
    main()