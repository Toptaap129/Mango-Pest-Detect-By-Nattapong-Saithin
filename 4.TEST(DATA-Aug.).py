# ============================================================================
# 🥭 MANGO PEST DETECTION - CUDA ACCELERATED TRAINING PIPELINE (PyTorch)
# ============================================================================
# ✅ FIXED: Corrected Transform Order for RandomErasing
# ============================================================================
import os
import sys
import warnings
import time
import platform
from datetime import datetime
from pathlib import Path
from typing import Tuple, Dict, Any
import numpy as np
import argparse
import logging
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
import json
import random

# ============================================================================
# 🔥 CUDA OPTIMIZATIONS (SET BEFORE PYTORCH IMPORT)
# ============================================================================
os.environ['OMP_NUM_THREADS'] = '2'
os.environ['OPENBLAS_NUM_THREADS'] = '2'
os.environ['MKL_NUM_THREADS'] = '2'
os.environ['VECLIB_MAXIMUM_THREADS'] = '2'
os.environ['NUMEXPR_NUM_THREADS'] = '2'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

# ============================================================================
# IMPORT PYTORCH AFTER ENVIRONMENT CONFIGURATION
# ============================================================================
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms
from torchvision.datasets import ImageFolder
from torch.optim.lr_scheduler import ReduceLROnPlateau, CosineAnnealingLR
import torch.nn.functional as F
from torch.cuda.amp import GradScaler, autocast
from sklearn.metrics import classification_report, confusion_matrix, precision_recall_fscore_support

# Import tqdm for progress bars (only if available)
try:
    from tqdm import tqdm
    TQDM_AVAILABLE = True
except ImportError:
    TQDM_AVAILABLE = False
    class tqdm:
        """Fallback tqdm implementation if package not installed"""
        def __init__(self, iterable, **kwargs):
            self.iterable = iterable
            self.total = kwargs.get('total', len(iterable) if hasattr(iterable, '__len__') else None)
            self.desc = kwargs.get('desc', '')
            self.current = 0
            if self.desc:
                print(f"\n{self.desc}", end='', flush=True)
        
        def __iter__(self):
            for item in self.iterable:
                self.current += 1
                if self.total and self.current % max(1, self.total // 10) == 0:
                    percent = (self.current / self.total) * 100
                    print(f" {percent:.0f}%", end='', flush=True)
                yield item
            if self.desc:
                print(" 100%")

warnings.filterwarnings('ignore', category=UserWarning, module='torch')
warnings.filterwarnings('ignore', category=FutureWarning)

# ============================================================================
# LOGGING SETUP
# ============================================================================
Path("logs").mkdir(exist_ok=True)
log_file = f"logs/cuda_training_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    handlers=[
        logging.FileHandler(log_file),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

# ============================================================================
# ⚡ CUDA VERIFICATION & OPTIMIZATION (CORRECTED - NO DEFAULT TENSOR TYPE)
# ============================================================================
def verify_cuda_support() -> Tuple[bool, str]:
    cuda_available = torch.cuda.is_available()
    cuda_info = ""
    
    if cuda_available:
        cuda_info = f"CUDA {torch.version.cuda} | "
        cuda_info += f"Device: {torch.cuda.get_device_name(0)} | "
        cuda_info += f"Compute Capability: {torch.cuda.get_device_capability(0)} | "
        cuda_info += f"Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f} GB"
    else:
        cuda_info = "CUDA NOT AVAILABLE - Falling back to CPU"
    
    return cuda_available, cuda_info

def setup_cuda_optimization() -> torch.device:
    logger.info("="*80)
    logger.info("🚀 INITIALIZING CUDA ACCELERATION LAYER")
    logger.info("="*80)
    
    cuda_available, cuda_info = verify_cuda_support()
    logger.info(f"CUDA Status: {'✅ AVAILABLE' if cuda_available else '❌ UNAVAILABLE'}")
    logger.info(f"GPU Details: {cuda_info}")
    
    if cuda_available:
        device = torch.device('cuda:0')
        torch.backends.cudnn.benchmark = True
        torch.backends.cudnn.deterministic = False
        torch.cuda.empty_cache()
        
        logger.info("\n✓✓✓ CUDA OPTIMIZATION CONFIGURATION APPLIED ✓✓✓")
        logger.info(f"  → cuDNN benchmark: ENABLED (auto-tunes convolution algorithms)")
        logger.info(f"  → Device: {device}")
        logger.info(f"  → GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f} GB")
    else:
        device = torch.device('cpu')
        logger.warning("\n⚠️  CUDA NOT AVAILABLE - Falling back to CPU training (significantly slower)")
        logger.info("   Tip: Install NVIDIA drivers and CUDA toolkit for GPU acceleration")
    
    logger.info("="*80)
    return device

# ============================================================================
# ✅ CUDA-OPTIMIZED DATA PIPELINE (PYTORCH) - WITH CORRECTED TRANSFORM ORDER
# ============================================================================
class CUDAOptimizedDataPipeline:
    def __init__(self, img_size: int = 224, num_workers: int = 4):
        self.img_size = img_size
        self.num_workers = num_workers
        
        # ✅ FIXED: Correct order for Tensor-based transforms
        self.train_transform = transforms.Compose([
            # 1️⃣ PIL-based Geometric & Photometric Transforms
            transforms.Resize((img_size, img_size), interpolation=transforms.InterpolationMode.BILINEAR),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomRotation(15, interpolation=transforms.InterpolationMode.BILINEAR, expand=False),
            transforms.RandomResizedCrop(img_size, scale=(0.8, 1.0), ratio=(0.9, 1.1), interpolation=transforms.InterpolationMode.BILINEAR),
            transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.15),
            transforms.GaussianBlur(kernel_size=(3, 7), sigma=(0.1, 2.0)),
            
            # 2️⃣ Convert to PyTorch Tensor
            transforms.ToTensor(),
            
            # 3️⃣ Tensor-based Augmentations (MUST BE AFTER ToTensor())
            transforms.RandomErasing(p=0.2, scale=(0.02, 0.15), ratio=(0.3, 3.3), value='random'),
            
            # 4️⃣ Normalization
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        
        self.val_test_transform = transforms.Compose([
            transforms.Resize((img_size, img_size), interpolation=transforms.InterpolationMode.BILINEAR),
            transforms.CenterCrop(img_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
    
    def load_datasets(self, data_path: str, batch_size: int = 64) -> Tuple:
        data_path = Path(data_path)
        if not data_path.exists():
            raise FileNotFoundError(f"Dataset path not found: {data_path}")
        
        has_subdirs = any((data_path / subdir).exists() for subdir in ['train', 'val', 'test'])
        
        if has_subdirs:
            train_dir = data_path / 'train'
            val_dir = data_path / 'val'
            test_dir = data_path / 'test'
            
            if not train_dir.exists():
                raise FileNotFoundError(f"Training directory not found: {train_dir}")
            
            train_dataset = ImageFolder(str(train_dir), transform=self.train_transform)
            class_names = train_dataset.classes
            num_classes = len(class_names)
            logger.info(f"✓ Detected {num_classes} classes: {class_names}")
            
            if not val_dir.exists():
                logger.warning("Validation directory not found. Using 20% of training data.")
                train_size = int(0.8 * len(train_dataset))
                val_size = len(train_dataset) - train_size
                
                generator = torch.Generator(device='cpu').manual_seed(42)
                train_dataset, val_dataset = random_split(
                    train_dataset, 
                    [train_size, val_size],
                    generator=generator
                )
                val_dataset.dataset.transform = self.val_test_transform
            else:
                val_dataset = ImageFolder(str(val_dir), transform=self.val_test_transform)
                if val_dataset.classes != class_names:
                    logger.warning("Validation set class names don't match training set. Reindexing...")
                    val_dataset = self._reindex_dataset(val_dataset, class_names)
            
            test_dir = test_dir if test_dir.exists() else (val_dir if val_dir.exists() else train_dir)
            test_dataset = ImageFolder(str(test_dir), transform=self.val_test_transform)
            if test_dataset.classes != class_names:
                test_dataset = self._reindex_dataset(test_dataset, class_names)
            
        else:
            logger.info("Using 70/15/15 split from single directory")
            full_dataset = ImageFolder(str(data_path), transform=self.train_transform)
            class_names = full_dataset.classes
            num_classes = len(class_names)
            logger.info(f"✓ Detected {num_classes} classes: {class_names}")
            
            total_size = len(full_dataset)
            train_size = int(0.7 * total_size)
            val_size = int(0.15 * total_size)
            test_size = total_size - train_size - val_size
            
            generator = torch.Generator(device='cpu').manual_seed(42)
            train_dataset, val_dataset, test_dataset = random_split(
                full_dataset,
                [train_size, val_size, test_size],
                generator=generator
            )
            
            val_dataset.dataset.transform = self.val_test_transform
            test_dataset.dataset.transform = self.val_test_transform
        
        train_loader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=True,
            prefetch_factor=4,
            persistent_workers=True,
            drop_last=True
        )
        
        val_loader = DataLoader(
            val_dataset,
            batch_size=batch_size * 2,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True,
            prefetch_factor=2,
            persistent_workers=True
        )
        
        test_loader = DataLoader(
            test_dataset,
            batch_size=batch_size * 2,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True,
            prefetch_factor=2,
            persistent_workers=True
        )
        
        logger.info(f"\n✓ DATASET OPTIMIZED FOR CUDA:")
        logger.info(f"  → Training:   {len(train_dataset):,} images ({len(class_names)} classes)")
        logger.info(f"  → Validation: {len(val_dataset):,} images")
        logger.info(f"  → Test:       {len(test_dataset):,} images")
        logger.info(f"  → Train Batch size: {batch_size}")
        logger.info(f"  → Val/Test Batch size: {batch_size * 2}")
        logger.info(f"  → DataLoader workers: {self.num_workers}")
        logger.info(f"  → Pin memory: ENABLED")
        logger.info(f"  → Augmentation: ENHANCED (Geometric + Photometric + Environmental)")
        
        return train_loader, val_loader, test_loader, class_names
    
    def _reindex_dataset(self, dataset: ImageFolder, target_classes: list) -> ImageFolder:
        class_to_idx = {cls: idx for idx, cls in enumerate(target_classes)}
        dataset.class_to_idx = class_to_idx
        dataset.classes = target_classes
        
        new_samples = []
        for path, old_idx in dataset.samples:
            class_name = Path(path).parent.name
            if class_name in class_to_idx:
                new_samples.append((path, class_to_idx[class_name]))
            else:
                logger.warning(f"Class '{class_name}' not in target classes, skipping")
        dataset.samples = new_samples
        return dataset

# ============================================================================
# ✅ CUDA-OPTIMIZED MODEL ARCHITECTURE (PYTORCH) - UNCHANGED
# ============================================================================
class CUDAOptimizedCNN(nn.Module):
    def __init__(self, num_classes: int, img_size: int = 224):
        super(CUDAOptimizedCNN, self).__init__()
        
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 32, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Dropout2d(0.1),
            
            nn.Conv2d(32, 64, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Dropout2d(0.1),
            
            nn.Conv2d(64, 128, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Dropout2d(0.2),
            
            nn.Conv2d(128, 256, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d(1)
        )
        
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(256, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.4),
            nn.Linear(512, num_classes)
        )
        
        self._initialize_weights()
    
    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                nn.init.constant_(m.bias, 0)
    
    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x

# ============================================================================
# ✅ TRAINING WITH CUDA OPTIMIZATIONS + REAL-TIME PROGRESS MONITORING + FULLY INFERENCE-READY CHECKPOINTS - UNCHANGED
# ============================================================================
class EarlyStopping:
    def __init__(self, patience=20, min_delta=1e-4, verbose=True):
        self.patience = patience
        self.min_delta = min_delta
        self.verbose = verbose
        self.counter = 0
        self.best_loss = None
        self.early_stop = False
        self.best_model_state = None
    
    def __call__(self, val_loss, model):
        """Returns True if a new best model was found"""
        if self.best_loss is None:
            self.best_loss = val_loss
            self._save_checkpoint(model)
            return True
        elif val_loss > self.best_loss - self.min_delta:
            self.counter += 1
            return False
        else:
            self.best_loss = val_loss
            self.counter = 0
            self._save_checkpoint(model)
            return True
    
    def _save_checkpoint(self, model):
        if self.verbose:
            logger.info(f"  → Validation loss decreased. Saving model state to memory...")
        self.best_model_state = {k: v.cpu() for k, v in model.state_dict().items()}

def format_time(seconds):
    """Convert seconds to HH:MM:SS format"""
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = int(seconds % 60)
    if hours > 0:
        return f"{hours:02d}:{minutes:02d}:{secs:02d}"
    else:
        return f"{minutes:02d}:{secs:02d}"

def get_gpu_memory(device):
    """Get formatted GPU memory usage"""
    if device.type == 'cuda':
        allocated = torch.cuda.memory_allocated(0) / 1024**3
        reserved = torch.cuda.memory_reserved(0) / 1024**3
        total = torch.cuda.get_device_properties(0).total_memory / 1024**3
        return f"{allocated:.2f}/{total:.1f}GB"
    return "CPU"

def save_inference_ready_checkpoint(
    model: nn.Module,
    class_names: list,
    img_size: int,
    epoch: int,
    val_loss: float,
    val_acc: float,
    train_loss: float,
    train_acc: float,
    timestamp: str,
    checkpoint_dir: str = "checkpoints"
) -> str:
    """
    Save a COMPLETELY INFERENCE-READY checkpoint with ALL necessary components
    This model can be loaded and used for prediction immediately without any additional files
    """
    Path(checkpoint_dir).mkdir(exist_ok=True)
    
    checkpoint_data = {
        'model_state_dict': {k: v.cpu() for k, v in model.state_dict().items()},
        'class_names': class_names,
        'class_indices': {name: idx for idx, name in enumerate(class_names)},
        'img_size': img_size,
        'num_classes': len(class_names),
        'epoch': epoch,
        'val_loss': val_loss,
        'val_acc': val_acc,
        'train_loss': train_loss,
        'train_acc': train_acc,
        'timestamp': timestamp,
        'model_architecture': 'CUDAOptimizedCNN',
        'pytorch_version': torch.__version__,
        'cuda_version': torch.version.cuda if torch.cuda.is_available() else None,
        '_inference_note': (
            "TO USE THIS MODEL FOR INFERENCE:\n"
            "1. Load checkpoint: checkpoint = torch.load('model.pt')\n"
            "2. Create model: model = CUDAOptimizedCNN(num_classes=checkpoint['num_classes'], img_size=checkpoint['img_size'])\n"
            "3. Load weights: model.load_state_dict(checkpoint['model_state_dict'])\n"
            "4. Set to eval: model.eval()\n"
            "5. Preprocess image to checkpoint['img_size'] and normalize with ImageNet stats\n"
            "6. Predict: output = model(image_tensor); pred_idx = torch.argmax(output, dim=1).item()\n"
            "7. Get class: prediction = checkpoint['class_names'][pred_idx]"
        )
    }
    
    unique_filename = f"{checkpoint_dir}/best_model_epoch_{epoch:03d}_val_loss_{val_loss:.4f}_val_acc_{val_acc:.4f}.pt"
    torch.save(checkpoint_data, unique_filename)
    
    current_best_path = f"{checkpoint_dir}/best_model.pt"
    torch.save(checkpoint_data, current_best_path)
    
    return unique_filename, current_best_path

def train_cuda_model(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    criterion: nn.Module,
    optimizer: optim.Optimizer,
    scheduler: Any,
    device: torch.device,
    epochs: int = 150,
    timestamp: str = None,
    num_classes: int = None,
    class_names: list = None,
    img_size: int = 224
) -> Tuple[Dict, float, int, float, float]:
    logger.info(f"\n[🚀 STEP 3] STARTING CUDA ACCELERATED TRAINING")
    logger.info(f"  → Mixed precision training: {'ENABLED' if device.type == 'cuda' else 'DISABLED'}")
    logger.info(f"  → Checkpointing: Saving FULLY INFERENCE-READY .pt models on every validation loss improvement")
    logger.info(f"  → Early stopping: Enabled (patience=20 epochs)")
    logger.info("="*80)
    
    Path("checkpoints").mkdir(exist_ok=True)
    early_stopping = EarlyStopping(patience=20, verbose=False)
    history = {
        'train_loss': [], 'train_acc': [], 'train_prec': [], 'train_rec': [],
        'val_loss': [], 'val_acc': [], 'val_prec': [], 'val_rec': []
    }
    
    scaler = GradScaler() if device.type == 'cuda' else None
    
    start_time = time.time()
    best_epoch = 0
    best_val_loss = float('inf')
    best_val_acc = 0.0
    
    use_tqdm = TQDM_AVAILABLE and sys.stdout.isatty()
    
    for epoch in range(epochs):
        epoch_start = time.time()
        
        lr = optimizer.param_groups[0]['lr']
        gpu_mem = get_gpu_memory(device)
        elapsed_total = time.time() - start_time
        avg_epoch_time = elapsed_total / (epoch + 1) if epoch > 0 else 0
        eta_total = avg_epoch_time * (epochs - epoch - 1) if avg_epoch_time > 0 else 0
        
        logger.info(f"\n┌─ Epoch {epoch+1:3d}/{epochs} "
                    f"| LR: {lr:.2e} "
                    f"| GPU: {gpu_mem} "
                    f"| Elapsed: {format_time(elapsed_total)} "
                    f"| ETA: {format_time(eta_total)}")
        logger.info(f"├─ Training Progress")
        
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        all_preds = []
        all_labels = []
        
        if use_tqdm:
            progress_bar = tqdm(
                enumerate(train_loader),
                total=len(train_loader),
                desc=f"  Batch",
                leave=False,
                ncols=100,
                bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]'
            )
        else:
            progress_bar = enumerate(train_loader)
            logger.info(f"  0% [{' ' * 50}] 0/{len(train_loader)} batches")
        
        for batch_idx, (inputs, labels) in progress_bar:
            inputs = inputs.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            
            optimizer.zero_grad()
            with autocast(enabled=(device.type == 'cuda')):
                outputs = model(inputs)
                loss = criterion(outputs, labels)
            
            if scaler:
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                optimizer.step()
            
            batch_loss = loss.item()
            running_loss += batch_loss * inputs.size(0)
            _, predicted = torch.max(outputs, 1)
            batch_correct = (predicted == labels).sum().item()
            batch_total = labels.size(0)
            total += batch_total
            correct += batch_correct
            
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            
            batch_acc = batch_correct / batch_total
            if use_tqdm:
                progress_bar.set_postfix({
                    'loss': f"{batch_loss:.4f}",
                    'acc': f"{batch_acc:.1%}",
                    'gpu': gpu_mem.split('/')[0]
                })
            else:
                if (batch_idx + 1) % max(1, len(train_loader) // 10) == 0:
                    percent = (batch_idx + 1) / len(train_loader) * 100
                    bar_len = 50
                    filled = int(bar_len * percent / 100)
                    bar = '█' * filled + ' ' * (bar_len - filled)
                    logger.info(f"  {percent:5.1f}% [{bar}] {batch_idx+1}/{len(train_loader)} batches "
                                f"(loss={batch_loss:.4f}, acc={batch_acc:.1%})")
        
        train_loss = running_loss / total
        train_acc = correct / total
        train_prec, train_rec, _, _ = precision_recall_fscore_support(
            all_labels, all_preds, average='weighted', zero_division=0
        )
        
        logger.info(f"├─ Validation Progress")
        
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0
        val_preds = []
        val_labels = []
        
        if use_tqdm:
            val_progress = tqdm(
                enumerate(val_loader),
                total=len(val_loader),
                desc=f"  Val Batch",
                leave=False,
                ncols=100,
                bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt}'
            )
        else:
            val_progress = enumerate(val_loader)
        
        with torch.no_grad():
            for batch_idx, (inputs, labels) in val_progress:
                inputs = inputs.to(device, non_blocking=True)
                labels = labels.to(device, non_blocking=True)
                
                with autocast(enabled=(device.type == 'cuda')):
                    outputs = model(inputs)
                    loss = criterion(outputs, labels)
                
                val_loss += loss.item() * inputs.size(0)
                _, predicted = torch.max(outputs, 1)
                val_total += labels.size(0)
                val_correct += (predicted == labels).sum().item()
                val_preds.extend(predicted.cpu().numpy())
                val_labels.extend(labels.cpu().numpy())
                
                if use_tqdm and batch_idx % 5 == 0:
                    val_progress.set_postfix({'loss': f"{loss.item():.4f}"})
        
        val_loss = val_loss / val_total
        val_acc = val_correct / val_total
        val_prec, val_rec, _, _ = precision_recall_fscore_support(
            val_labels, val_preds, average='weighted', zero_division=0
        )
        
        if isinstance(scheduler, ReduceLROnPlateau):
            scheduler.step(val_loss)
        else:
            scheduler.step()
        
        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        history['train_prec'].append(train_prec)
        history['train_rec'].append(train_rec)
        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_acc)
        history['val_prec'].append(val_prec)
        history['val_rec'].append(val_rec)
        
        epoch_time = time.time() - epoch_start
        epoch_gpu_mem = get_gpu_memory(device)
        
        is_new_best = early_stopping(val_loss, model)
        if is_new_best:
            best_val_loss = val_loss
            best_val_acc = val_acc
            best_epoch = epoch + 1
            star_marker = " ⭐ NEW BEST"
            
            unique_path, current_best_path = save_inference_ready_checkpoint(
                model=model,
                class_names=class_names,
                img_size=img_size,
                epoch=epoch + 1,
                val_loss=val_loss,
                val_acc=val_acc,
                train_loss=train_loss,
                train_acc=train_acc,
                timestamp=timestamp,
                checkpoint_dir="checkpoints"
            )
            logger.info(f"   💾 FULLY INFERENCE-READY CHECKPOINT SAVED:")
            logger.info(f"      → Unique: '{Path(unique_path).name}'")
            logger.info(f"      → Current best (overwritten): '{Path(current_best_path).name}'")
        else:
            star_marker = ""
        
        def color_metric(value, best_value=None, is_loss=False):
            if best_value is None:
                return f"{value:.4f}"
            if (is_loss and value <= best_value) or (not is_loss and value >= best_value):
                return f"\033[92m{value:.4f}\033[0m"
            else:
                return f"\033[93m{value:.4f}\033[0m"
        
        logger.info(f"└─ Epoch Summary (took {epoch_time:.1f}s)")
        logger.info(f"   ├─ Training:  Loss={train_loss:.4f}  Acc={train_acc:.2%}  Prec={train_prec:.4f}  Rec={train_rec:.4f}")
        logger.info(f"   ├─ Validation: Loss={color_metric(val_loss, best_val_loss, is_loss=True)}  "
                    f"Acc={color_metric(val_acc, best_val_acc)}  "
                    f"Prec={val_prec:.4f}  Rec={val_rec:.4f}{star_marker}")
        logger.info(f"   └─ GPU Memory: {epoch_gpu_mem} | LR: {lr:.2e}")
        
        if early_stopping.early_stop:
            logger.info(f"\n🛑 EARLY STOPPING TRIGGERED at epoch {epoch+1} (no improvement for {early_stopping.patience} epochs)")
            model.load_state_dict(early_stopping.best_model_state)
            break
    
    total_time = time.time() - start_time
    
    if not early_stopping.early_stop:
        best_epoch = int(np.argmin(history['val_loss'])) + 1
        best_val_loss = float(np.min(history['val_loss']))
        best_val_acc = history['val_acc'][best_epoch - 1]
    
    logger.info(f"\n" + "="*80)
    logger.info(f"✓ TRAINING COMPLETED IN {format_time(total_time)} ({total_time:.1f} seconds)")
    logger.info(f"  → Best validation loss:  {best_val_loss:.6f} at epoch {best_epoch}")
    logger.info(f"  → Best validation accuracy: {best_val_acc:.2%}")
    logger.info(f"  → Final validation loss: {history['val_loss'][-1]:.6f}")
    logger.info("="*80)
    
    return history, total_time, best_epoch, best_val_loss, best_val_acc

# ============================================================================
# ✅ EVALUATION & VISUALIZATION - UNCHANGED
# ============================================================================
def evaluate_model(
    model: nn.Module,
    test_loader: DataLoader,
    class_names: list,
    device: torch.device
) -> Dict:
    logger.info("\n[🔍 STEP 4] EVALUATING MODEL ON TEST SET...")
    
    model.eval()
    all_preds = []
    all_labels = []
    all_probs = []
    
    inference_start = time.time()
    
    use_tqdm = TQDM_AVAILABLE and sys.stdout.isatty()
    if use_tqdm:
        progress_bar = tqdm(
            enumerate(test_loader),
            total=len(test_loader),
            desc="  Test Evaluation",
            ncols=100
        )
    else:
        progress_bar = enumerate(test_loader)
        logger.info(f"  0% [{' ' * 50}] 0/{len(test_loader)} batches")
    
    with torch.no_grad():
        for batch_idx, (inputs, labels) in progress_bar:
            inputs = inputs.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            
            with autocast(enabled=(device.type == 'cuda')):
                outputs = model(inputs)
                probs = F.softmax(outputs, dim=1)
                _, preds = torch.max(outputs, 1)
            
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())
            
            if not use_tqdm and (batch_idx + 1) % max(1, len(test_loader) // 10) == 0:
                percent = (batch_idx + 1) / len(test_loader) * 100
                bar_len = 50
                filled = int(bar_len * percent / 100)
                bar = '█' * filled + ' ' * (bar_len - filled)
                logger.info(f"  {percent:5.1f}% [{bar}] {batch_idx+1}/{len(test_loader)} batches")
    
    inference_time = time.time() - inference_start
    images_processed = len(all_preds)
    fps = images_processed / inference_time
    
    test_acc = np.mean(np.array(all_preds) == np.array(all_labels))
    test_prec, test_rec, test_f1, _ = precision_recall_fscore_support(
        all_labels, all_preds, average='weighted', zero_division=0
    )
    
    logger.info(f"\n{'='*70}")
    logger.info("TEST SET PERFORMANCE (MODEL SAVED AT LOWEST VAL_LOSS)")
    logger.info(f"{'='*70}")
    logger.info(f"Images Processed: {images_processed:,}")
    logger.info(f"Inference Time:   {inference_time:.2f} seconds")
    logger.info(f"Inference Speed:  {fps:.1f} FPS (images/second)")
    logger.info(f"Accuracy:         {test_acc:.4f} ({test_acc*100:.2f}%)")
    logger.info(f"Precision:        {test_prec:.4f}")
    logger.info(f"Recall:           {test_rec:.4f}")
    logger.info(f"F1-Score:         {test_f1:.4f}")
    logger.info(f"{'='*70}")
    
    cm = confusion_matrix(all_labels, all_preds)
    plt.figure(figsize=(12, 10))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=class_names, yticklabels=class_names)
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('Confusion Matrix')
    plt.tight_layout()
    plt.savefig('confusion_matrix.png', dpi=150, bbox_inches='tight')
    plt.close()
    logger.info("✓ Confusion matrix saved to 'confusion_matrix.png'")
    
    report = classification_report(all_labels, all_preds, target_names=class_names, output_dict=True)
    logger.info("\nPer-Class Performance:")
    for class_name, metrics in report.items():
        if class_name not in ['accuracy', 'macro avg', 'weighted avg']:
            logger.info(f"{class_name:20s} | Prec: {metrics['precision']:.4f} Rec: {metrics['recall']:.4f} F1: {metrics['f1-score']:.4f}")
    
    return {
        'accuracy': test_acc,
        'precision': test_prec,
        'recall': test_rec,
        'f1_score': test_f1,
        'inference_fps': fps,
        'inference_time': inference_time,
        'confusion_matrix': cm,
        'predictions': all_preds,
        'labels': all_labels,
        'probabilities': all_probs
    }

def plot_training_history(history: dict, timestamp: str, best_epoch: int):
    Path("plots").mkdir(exist_ok=True)
    
    plt.figure(figsize=(14, 5))
    
    plt.subplot(1, 2, 1)
    plt.plot(history['train_acc'], label='Train Accuracy', linewidth=2)
    plt.plot(history['val_acc'], label='Val Accuracy', linewidth=2)
    plt.axvline(x=best_epoch-1, color='r', linestyle='--', alpha=0.7, label=f'Best loss (epoch {best_epoch})')
    plt.title('Model Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.grid(alpha=0.3)
    
    plt.subplot(1, 2, 2)
    plt.plot(history['train_loss'], label='Train Loss', linewidth=2)
    plt.plot(history['val_loss'], label='Val Loss', linewidth=2)
    plt.axvline(x=best_epoch-1, color='r', linestyle='--', alpha=0.7, label=f'Best loss (epoch {best_epoch})')
    plt.scatter([best_epoch-1], [min(history['val_loss'])], 
                color='red', s=100, zorder=5, label=f'Min val_loss: {min(history["val_loss"]):.4f}')
    plt.title('Model Loss (Lower is Better)')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(f'plots/training_history_{timestamp}.png', dpi=150, bbox_inches='tight')
    plt.close()
    logger.info(f"✓ Training history saved to 'plots/training_history_{timestamp}.png'")

# ============================================================================
# ✅ SAVE CLASS NAMES TO ROOT DIRECTORY ONLY (NOT IN CHECKPOINTS/) - UNCHANGED
# ============================================================================
def save_class_names_to_root(class_names: list):
    with open("class_names.txt", 'w', encoding='utf-8') as f:
        for name in class_names:
            f.write(f"{name}\n")
    logger.info(f"✅ Class names saved to 'class_names.txt' (root directory - 1 class per line)")
    
    logger.info("\n📋 Class Names Detected:")
    for i, name in enumerate(class_names):
        logger.info(f"   [{i:2d}] {name}")

# ============================================================================
# ✅ MAIN TRAINING PIPELINE - PURE .PT CHECKPOINTS ONLY + FULLY INFERENCE-READY SAVING - UNCHANGED
# ============================================================================
def main():
    parser = argparse.ArgumentParser(description='CUDA Accelerated Mango Pest Detection (PyTorch)')
    parser.add_argument('--data_path', type=str, default='Dataset')
    parser.add_argument('--img_size', type=int, default=96)
    parser.add_argument('--epochs', type=int, default=50000)
    parser.add_argument('--learning_rate', type=float, default=0.01)
    parser.add_argument('--batch_size', type=int, default=320)
    parser.add_argument('--num_workers', type=int, default=16)
    args = parser.parse_args()
    
    torch.manual_seed(42)
    np.random.seed(42)
    random.seed(42)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(42)
    
    logger.info("="*80)
    logger.info("🥭 MANGO PEST DETECTION - CUDA ACCELERATED TRAINING PIPELINE")
    logger.info("="*80)
    logger.info(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    logger.info(f"PyTorch Version: {torch.__version__}")
    logger.info(f"CUDA Version: {torch.version.cuda if torch.cuda.is_available() else 'N/A'}")
    logger.info(f"tqdm Progress Bars: {'✅ ENABLED' if TQDM_AVAILABLE and sys.stdout.isatty() else '⚠️ DISABLED (non-interactive terminal)'}")
    logger.info(f"Data Augmentation: ENHANCED (Agricultural Domain Optimized)")
    logger.info("="*80)
    
    device = setup_cuda_optimization()
    
    logger.info(f"\n[📦 STEP 1] LOADING DATASET")
    data_pipeline = CUDAOptimizedDataPipeline(
        img_size=args.img_size,
        num_workers=args.num_workers
    )
    train_loader, val_loader, test_loader, class_names = data_pipeline.load_datasets(
        args.data_path,
        batch_size=args.batch_size
    )
    
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    
    save_class_names_to_root(class_names)
    
    num_classes = len(class_names)
    logger.info(f"\n[🧠 STEP 2] BUILDING CUDA-OPTIMIZED MODEL ({num_classes} CLASSES)")
    model = CUDAOptimizedCNN(num_classes=num_classes, img_size=args.img_size)
    model = model.to(device)
    
    logger.info(f"Model architecture: {model.__class__.__name__}")
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"Total parameters: {total_params:,}")
    logger.info(f"Trainable parameters: {trainable_params:,}")
    
    logger.info(f"\n[⚙️ STEP 3] COMPILING MODEL FOR CUDA")
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(
        model.parameters(),
        lr=args.learning_rate,
        momentum=0.9,
        nesterov=True,
        weight_decay=5e-4
    )
    
    scheduler = CosineAnnealingLR(
        optimizer,
        T_max=args.epochs,
        eta_min=1e-6
    )
    
    history, training_time, best_epoch, best_val_loss, best_val_acc = train_cuda_model(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        criterion=criterion,
        optimizer=optimizer,
        scheduler=scheduler,
        device=device,
        epochs=args.epochs,
        timestamp=timestamp,
        num_classes=num_classes,
        class_names=class_names,
        img_size=args.img_size
    )
    
    results = evaluate_model(model, test_loader, class_names, device)
    
    Path("checkpoints").mkdir(exist_ok=True)
    final_model_path = f'checkpoints/final_model_{timestamp}.pt'
    torch.save({
        'model_state_dict': {k: v.cpu() for k, v in model.state_dict().items()},
        'class_names': class_names,
        'class_indices': {name: idx for idx, name in enumerate(class_names)},
        'img_size': args.img_size,
        'num_classes': num_classes,
        'best_epoch': best_epoch,
        'best_val_loss': best_val_loss,
        'best_val_acc': best_val_acc,
        'test_accuracy': results['accuracy'],
        'test_precision': results['precision'],
        'test_recall': results['recall'],
        'test_f1': results['f1_score'],
        'inference_fps': results['inference_fps'],
        'timestamp': timestamp,
        'pytorch_version': torch.__version__,
        'cuda_version': torch.version.cuda if torch.cuda.is_available() else None,
        'model_architecture': 'CUDAOptimizedCNN'
    }, final_model_path)
    logger.info(f"\n✅ FINAL MODEL WITH TEST METRICS SAVED TO: '{final_model_path}'")
    
    info_path = f'logs/training_info_{timestamp}.txt'
    with open(info_path, 'w', encoding='utf-8') as f:
        f.write(f"MANGO PEST DETECTION - TRAINING REPORT\n")
        f.write(f"{'='*60}\n")
        f.write(f"Timestamp:          {timestamp}\n")
        f.write(f"PyTorch Version:    {torch.__version__}\n")
        f.write(f"CUDA Version:       {torch.version.cuda if torch.cuda.is_available() else 'N/A'}\n")
        f.write(f"GPU Device:         {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'}\n")
        f.write(f"Image Size:         {args.img_size}x{args.img_size}\n")
        f.write(f"Batch Size:         {args.batch_size}\n")
        f.write(f"Classes:            {num_classes} ({', '.join(class_names)})\n")
        f.write(f"Best Epoch:         {best_epoch}\n")
        f.write(f"Best Val Loss:      {best_val_loss:.6f}\n")
        f.write(f"Best Val Accuracy:  {best_val_acc:.4f}\n")
        f.write(f"Test Accuracy:      {results['accuracy']:.4f}\n")
        f.write(f"Inference Speed:    {results['inference_fps']:.1f} FPS\n")
        f.write(f"Training Time:      {format_time(training_time)}\n")
        f.write(f"\nBest Model Checkpoints Saved During Training:\n")
        f.write(f"  → All checkpoints with improving validation loss saved in 'checkpoints/' directory\n")
        f.write(f"  → Filename format: best_model_epoch_XXX_val_loss_XXXX_val_acc_XXXX.pt\n")
        f.write(f"  → Current best model always available as: checkpoints/best_model.pt\n")
        f.write(f"\nAugmentation Applied:\n")
        f.write(f"  → Geometric: Flip, Rotation(±15°), ResizedCrop\n")
        f.write(f"  → Photometric: ColorJitter(brightness/contrast/saturation=0.3, hue=0.15)\n")
        f.write(f"  → Environmental: GaussianBlur(kernel=3-7, sigma=0.1-2.0), RandomErasing(p=0.2)\n")
    logger.info(f"✅ Training metadata saved to '{info_path}' (logs directory)")
    
    plot_training_history(history, timestamp, best_epoch)
    
    logger.info("\n" + "="*80)
    logger.info("✅ CUDA TRAINING COMPLETED SUCCESSFULLY")
    logger.info("="*80)
    logger.info(f"Device:               {'GPU (CUDA)' if device.type == 'cuda' else 'CPU'}")
    if device.type == 'cuda':
        logger.info(f"GPU Model:            {torch.cuda.get_device_name(0)}")
        logger.info(f"GPU Memory:           {torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f} GB")
    logger.info(f"Best Validation Loss: {best_val_loss:.6f} (epoch {best_epoch})")
    logger.info(f"Test Accuracy:        {results['accuracy']:.4f}")
    logger.info(f"Inference Speed:      {results['inference_fps']:.1f} FPS")
    logger.info(f"Total Training Time:  {format_time(training_time)}")
    logger.info(f"\n📁 CHECKPOINTS DIRECTORY CONTENTS (PURE .PT FILES ONLY - ALL INFERENCE-READY):")
    
    checkpoint_files = sorted(Path("checkpoints").glob("*.pt"))
    for i, file in enumerate(checkpoint_files, 1):
        filename = file.name
        if 'best_model_epoch' in filename:
            logger.info(f"   [{i:2d}] {filename}  ← Improving validation loss")
        elif 'best_model.pt' == filename:
            logger.info(f"   [{i:2d}] {filename}  ← CURRENT BEST MODEL (always up-to-date)")
        elif 'final_model' in filename:
            logger.info(f"   [{i:2d}] {filename}  ← Final model with test metrics")
    
    logger.info(f"\n💡 KEY FEATURE: EVERY CHECKPOINT IS FULLY INFERENCE-READY")
    logger.info(f"   → Each .pt file contains EVERYTHING needed for deployment:")
    logger.info(f"      • Model weights (state_dict)")
    logger.info(f"      • Class names and mappings")
    logger.info(f"      • Image size requirements")
    logger.info(f"      • Architecture information")
    logger.info(f"      • Usage instructions embedded in checkpoint")
    logger.info(f"\n💡 INFERENCE USAGE (SIMPLE 7-STEP PROCESS):")
    logger.info(f"   # 1. Load checkpoint")
    logger.info(f"   checkpoint = torch.load('checkpoints/best_model.pt')")
    logger.info(f"   ")
    logger.info(f"   # 2. Create model with EXACT same architecture")
    logger.info(f"   model = CUDAOptimizedCNN(")
    logger.info(f"       num_classes=checkpoint['num_classes'],")
    logger.info(f"       img_size=checkpoint['img_size']")
    logger.info(f"   )")
    logger.info(f"   ")
    logger.info(f"   # 3. Load weights")
    logger.info(f"   model.load_state_dict(checkpoint['model_state_dict'])")
    logger.info(f"   ")
    logger.info(f"   # 4. Set to evaluation mode")
    logger.info(f"   model.eval()")
    logger.info(f"   ")
    logger.info(f"   # 5. Preprocess image to checkpoint['img_size'] with ImageNet normalization")
    logger.info(f"   # 6. Run inference")
    logger.info(f"   with torch.no_grad():")
    logger.info(f"       output = model(transformed_image.unsqueeze(0))")
    logger.info(f"       pred_idx = torch.argmax(output, dim=1).item()")
    logger.info(f"   ")
    logger.info(f"   # 7. Get prediction")
    logger.info(f"   prediction = checkpoint['class_names'][pred_idx]")
    logger.info(f"   print(f'Predicted class: {{prediction}}')")
    logger.info("="*80)

if __name__ == '__main__':
    main()

    