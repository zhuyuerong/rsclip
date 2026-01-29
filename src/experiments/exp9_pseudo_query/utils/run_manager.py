"""
可审计训练协议 - Run Manager

确保每次训练都是可追溯的:
1. 数据来源可控
2. 模型与权重可控
3. 所有超参可见
4. 随机性可控
5. 运行环境可控
6. 路径可控
7. 训练过程透明记录
8. 调试过程透明记录
"""

import os
import sys
import json
import yaml
import hashlib
import subprocess
import datetime
from pathlib import Path
from dataclasses import dataclass, asdict, field
from typing import Dict, Any, Optional, List, Tuple
import torch
import numpy as np
import random


@dataclass
class DataInfo:
    """数据来源信息"""
    dataset: str = ""
    root_path: str = ""  # 绝对路径
    train_split: str = ""
    val_split: str = ""
    test_split: str = ""
    train_samples: int = 0
    val_samples: int = 0
    test_samples: int = 0
    train_ids_hash: str = ""  # 样本ID列表的hash
    val_ids_hash: str = ""
    shuffle: bool = True
    min_box_size: Optional[int] = None
    filters: List[str] = field(default_factory=list)


@dataclass
class ModelInfo:
    """模型与权重信息"""
    model_class: str = ""
    backbone_pretrain: str = ""  # 路径
    backbone_pretrain_hash: str = ""
    detr_checkpoint: Optional[str] = None
    detr_checkpoint_hash: Optional[str] = None
    clip_checkpoint: Optional[str] = None
    clip_checkpoint_hash: Optional[str] = None
    
    # 加载结果
    missing_keys: List[str] = field(default_factory=list)
    unexpected_keys: List[str] = field(default_factory=list)


@dataclass
class EnvInfo:
    """运行环境信息"""
    # Git
    git_commit: str = ""
    git_branch: str = ""
    git_dirty: bool = True
    git_diff_file: Optional[str] = None  # 保存diff的文件
    
    # Python/PyTorch
    python_version: str = ""
    pytorch_version: str = ""
    cuda_version: str = ""
    cudnn_version: str = ""
    
    # GPU
    gpu_name: str = ""
    gpu_memory: str = ""
    gpu_driver: str = ""
    
    # 依赖
    requirements_file: Optional[str] = None


@dataclass
class RandomInfo:
    """随机性控制信息"""
    seed: int = 0
    deterministic: bool = False
    cudnn_benchmark: bool = False
    cudnn_deterministic: bool = False


@dataclass
class RunManifest:
    """运行清单 - 每次训练必须生成"""
    # 元信息
    run_id: str = ""
    exp_name: str = ""
    start_time: str = ""
    run_dir: str = ""  # 绝对路径
    
    # 8类可控信息
    data: DataInfo = field(default_factory=DataInfo)
    model: ModelInfo = field(default_factory=ModelInfo)
    env: EnvInfo = field(default_factory=EnvInfo)
    random: RandomInfo = field(default_factory=RandomInfo)
    
    # 完整config
    config: Dict[str, Any] = field(default_factory=dict)
    
    # 关键实验变量摘要
    key_vars: Dict[str, Any] = field(default_factory=dict)


class RunManager:
    """
    训练运行管理器
    
    使用方法:
        manager = RunManager(exp_name, output_dir)
        manager.setup_run(config)
        manager.verify_paths()
        manager.set_random_seed(seed)
        manager.log_checkpoint_load(...)
        manager.save_manifest()
        
        # 训练过程中
        manager.log_metrics(epoch, metrics)
        manager.log_debug("changed lr to 1e-5", reason="loss not decreasing")
        
        # 训练结束
        manager.finalize()
    """
    
    def __init__(self, exp_name: str, output_root: str):
        self.exp_name = exp_name
        self.output_root = Path(output_root).resolve()
        
        # 生成唯一run_id
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        self.run_id = f"{exp_name}_{timestamp}"
        self.run_dir = self.output_root / self.run_id
        
        # 创建目录
        self.run_dir.mkdir(parents=True, exist_ok=True)
        (self.run_dir / "checkpoints").mkdir(exist_ok=True)
        (self.run_dir / "logs").mkdir(exist_ok=True)
        
        # 初始化日志
        self.debug_log_file = self.run_dir / "debug_log.md"
        self.metrics_file = self.run_dir / "metrics.jsonl"
        
        self.manifest: Optional[RunManifest] = None
        self._debug_entries: List[Dict] = []
    
    def setup_run(self, config: Dict[str, Any]):
        """设置运行环境并收集信息"""
        
        # 收集环境信息
        env_info = self._collect_env_info()
        
        # 保存git diff
        if env_info.git_dirty:
            diff_file = self.run_dir / "patch.diff"
            self._save_git_diff(diff_file)
            env_info.git_diff_file = str(diff_file)
        
        # 保存requirements
        req_file = self.run_dir / "requirements.txt"
        self._save_requirements(req_file)
        env_info.requirements_file = str(req_file)
        
        # 创建manifest (data/model部分稍后填充)
        self.manifest = RunManifest(
            run_id=self.run_id,
            exp_name=self.exp_name,
            start_time=datetime.datetime.now().isoformat(),
            run_dir=str(self.run_dir),
            data=DataInfo(dataset="", root_path="", train_split="", val_split="", test_split=""),
            model=ModelInfo(model_class="", backbone_pretrain="", backbone_pretrain_hash=""),
            env=env_info,
            random=RandomInfo(seed=0, deterministic=False, cudnn_benchmark=False, cudnn_deterministic=False),
            config=config,
        )
        
        # 提取关键实验变量
        self._extract_key_vars(config)
        
        # 打印运行信息
        self._print_run_header()
    
    def setup_data(
        self,
        dataset: str,
        root_path: str,
        train_split: str,
        val_split: str,
        test_split: str,
        train_ids: Optional[List] = None,
        val_ids: Optional[List] = None,
        **kwargs
    ):
        """设置数据信息"""
        root_path = str(Path(root_path).resolve())
        
        self.manifest.data = DataInfo(
            dataset=dataset,
            root_path=root_path,
            train_split=train_split,
            val_split=val_split,
            test_split=test_split,
            train_samples=len(train_ids) if train_ids else 0,
            val_samples=len(val_ids) if val_ids else 0,
            train_ids_hash=self._hash_list(train_ids) if train_ids else "",
            val_ids_hash=self._hash_list(val_ids) if val_ids else "",
            **kwargs
        )
        
        print(f"📂 Data: {dataset}")
        print(f"   Root: {root_path}")
        print(f"   Train: {self.manifest.data.train_samples} samples")
        print(f"   Val: {self.manifest.data.val_samples} samples")
    
    def setup_model(
        self,
        model_class: str,
        backbone_pretrain: Optional[str] = None,
        detr_checkpoint: Optional[str] = None,
        clip_checkpoint: Optional[str] = None,
    ):
        """设置模型信息"""
        self.manifest.model = ModelInfo(
            model_class=model_class,
            backbone_pretrain=str(Path(backbone_pretrain).resolve()) if backbone_pretrain else "",
            backbone_pretrain_hash=self._hash_file(backbone_pretrain) if backbone_pretrain else "",
            detr_checkpoint=str(Path(detr_checkpoint).resolve()) if detr_checkpoint else None,
            detr_checkpoint_hash=self._hash_file(detr_checkpoint) if detr_checkpoint else None,
            clip_checkpoint=str(Path(clip_checkpoint).resolve()) if clip_checkpoint else None,
            clip_checkpoint_hash=self._hash_file(clip_checkpoint) if clip_checkpoint else None,
        )
        
        print(f"🏗️  Model: {model_class}")
        if backbone_pretrain:
            print(f"   Backbone: {backbone_pretrain}")
        if detr_checkpoint:
            print(f"   DETR: {detr_checkpoint}")
    
    def log_checkpoint_load(self, missing_keys: List[str], unexpected_keys: List[str]):
        """记录checkpoint加载结果"""
        self.manifest.model.missing_keys = missing_keys
        self.manifest.model.unexpected_keys = unexpected_keys
        
        if missing_keys:
            print(f"   ⚠️ Missing keys: {len(missing_keys)}")
            for k in missing_keys[:5]:
                print(f"      - {k}")
            if len(missing_keys) > 5:
                print(f"      ... and {len(missing_keys)-5} more")
        
        if unexpected_keys:
            print(f"   ⚠️ Unexpected keys: {len(unexpected_keys)}")
    
    def set_random_seed(self, seed: int, deterministic: bool = True):
        """设置随机种子"""
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        
        if deterministic:
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
        else:
            torch.backends.cudnn.benchmark = True
        
        self.manifest.random = RandomInfo(
            seed=seed,
            deterministic=deterministic,
            cudnn_benchmark=torch.backends.cudnn.benchmark,
            cudnn_deterministic=torch.backends.cudnn.deterministic,
        )
        
        print(f"🎲 Random seed: {seed} (deterministic={deterministic})")
    
    def verify_paths(self, paths: Dict[str, str]):
        """验证所有路径存在"""
        print("\n📁 Verifying paths...")
        all_exist = True
        
        paths_info = {}
        for name, path in paths.items():
            abs_path = str(Path(path).resolve())
            exists = os.path.exists(abs_path)
            paths_info[name] = {"path": abs_path, "exists": exists}
            
            status = "✓" if exists else "✗"
            print(f"   {status} {name}: {abs_path}")
            
            if not exists:
                all_exist = False
        
        # 保存paths.json
        with open(self.run_dir / "paths.json", "w") as f:
            json.dump(paths_info, f, indent=2)
        
        if not all_exist:
            raise FileNotFoundError("Some required paths do not exist!")
        
        return True
    
    def save_manifest(self):
        """保存运行清单"""
        manifest_file = self.run_dir / "manifest.json"
        
        # 转换为dict
        manifest_dict = asdict(self.manifest)
        
        with open(manifest_file, "w") as f:
            json.dump(manifest_dict, f, indent=2, default=str)
        
        # 同时保存yaml版本 (更易读)
        with open(self.run_dir / "manifest.yaml", "w") as f:
            yaml.dump(manifest_dict, f, default_flow_style=False)
        
        print(f"\n📋 Manifest saved to: {manifest_file}")
    
    def log_metrics(self, epoch: int, metrics: Dict[str, float], phase: str = "train"):
        """记录训练指标"""
        entry = {
            "timestamp": datetime.datetime.now().isoformat(),
            "epoch": epoch,
            "phase": phase,
            **metrics
        }
        
        with open(self.metrics_file, "a") as f:
            f.write(json.dumps(entry) + "\n")
    
    def log_debug(self, action: str, reason: str = "", expected_impact: str = "", result: str = ""):
        """记录调试操作"""
        entry = {
            "timestamp": datetime.datetime.now().isoformat(),
            "action": action,
            "reason": reason,
            "expected_impact": expected_impact,
            "result": result,
        }
        self._debug_entries.append(entry)
        
        # 追加到debug_log.md
        with open(self.debug_log_file, "a") as f:
            f.write(f"\n## {entry['timestamp']}\n")
            f.write(f"**Action:** {action}\n")
            if reason:
                f.write(f"**Reason:** {reason}\n")
            if expected_impact:
                f.write(f"**Expected Impact:** {expected_impact}\n")
            if result:
                f.write(f"**Result:** {result}\n")
    
    def check_sanity(self, epoch: int, metrics: Dict[str, float], checks: Dict[str, Tuple[str, float]]) -> List[str]:
        """检查指标是否在预期范围内"""
        warnings = []
        
        for metric_name, (op, threshold) in checks.items():
            if metric_name not in metrics:
                continue
            
            value = metrics[metric_name]
            
            if op == "<" and not (value < threshold):
                warnings.append(f"{metric_name}={value:.4f} should be < {threshold}")
            elif op == ">" and not (value > threshold):
                warnings.append(f"{metric_name}={value:.4f} should be > {threshold}")
            elif op == "<=" and not (value <= threshold):
                warnings.append(f"{metric_name}={value:.4f} should be <= {threshold}")
            elif op == ">=" and not (value >= threshold):
                warnings.append(f"{metric_name}={value:.4f} should be >= {threshold}")
        
        if warnings:
            print(f"\n⚠️ Sanity check warnings at epoch {epoch}:")
            for w in warnings:
                print(f"   - {w}")
            
            self.log_debug(
                action="Sanity check warnings",
                reason="; ".join(warnings),
                expected_impact="May indicate training issue"
            )
        
        return warnings
    
    def finalize(self):
        """训练结束时调用"""
        # 保存debug log为json
        with open(self.run_dir / "debug_log.json", "w") as f:
            json.dump(self._debug_entries, f, indent=2)
        
        # 更新manifest
        self.manifest.config["end_time"] = datetime.datetime.now().isoformat()
        self.save_manifest()
        
        print(f"\n✅ Run completed: {self.run_dir}")
    
    # ==================== 私有方法 ====================
    
    def _collect_env_info(self) -> EnvInfo:
        """收集环境信息"""
        # Git
        try:
            git_commit = subprocess.check_output(
                ["git", "rev-parse", "HEAD"], stderr=subprocess.DEVNULL
            ).decode().strip()
            git_branch = subprocess.check_output(
                ["git", "rev-parse", "--abbrev-ref", "HEAD"], stderr=subprocess.DEVNULL
            ).decode().strip()
            git_status = subprocess.check_output(
                ["git", "status", "--porcelain"], stderr=subprocess.DEVNULL
            ).decode().strip()
            git_dirty = len(git_status) > 0
        except:
            git_commit = "unknown"
            git_branch = "unknown"
            git_dirty = True
        
        # GPU
        try:
            gpu_name = torch.cuda.get_device_name(0)
            gpu_memory = f"{torch.cuda.get_device_properties(0).total_memory / 1e9:.1f}GB"
        except:
            gpu_name = "unknown"
            gpu_memory = "unknown"
        
        try:
            nvidia_smi = subprocess.check_output(
                ["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"]
            ).decode().strip().split("\n")[0]
            gpu_driver = nvidia_smi
        except:
            gpu_driver = "unknown"
        
        return EnvInfo(
            git_commit=git_commit,
            git_branch=git_branch,
            git_dirty=git_dirty,
            python_version=sys.version.split()[0],
            pytorch_version=torch.__version__,
            cuda_version=torch.version.cuda or "N/A",
            cudnn_version=str(torch.backends.cudnn.version()),
            gpu_name=gpu_name,
            gpu_memory=gpu_memory,
            gpu_driver=gpu_driver,
        )
    
    def _save_git_diff(self, output_file: Path):
        """保存git diff"""
        try:
            diff = subprocess.check_output(["git", "diff"], stderr=subprocess.DEVNULL)
            with open(output_file, "wb") as f:
                f.write(diff)
        except:
            pass
    
    def _save_requirements(self, output_file: Path):
        """保存pip freeze"""
        try:
            reqs = subprocess.check_output(
                [sys.executable, "-m", "pip", "freeze"], stderr=subprocess.DEVNULL
            )
            with open(output_file, "wb") as f:
                f.write(reqs)
        except:
            pass
    
    def _hash_file(self, filepath: Optional[str]) -> str:
        """计算文件SHA256"""
        if not filepath or not os.path.exists(filepath):
            return ""
        
        sha256 = hashlib.sha256()
        with open(filepath, "rb") as f:
            for chunk in iter(lambda: f.read(8192), b""):
                sha256.update(chunk)
        return sha256.hexdigest()[:16]  # 只取前16位
    
    def _hash_list(self, items: List) -> str:
        """计算列表内容的hash"""
        if not items:
            return ""
        content = json.dumps(sorted([str(i) for i in items]))
        return hashlib.md5(content.encode()).hexdigest()[:16]
    
    def _extract_key_vars(self, config: Dict):
        """提取关键实验变量"""
        key_vars = {}
        
        # 从config提取关键字段
        if "pseudo_config" in config and config.get("use_pseudo_query"):
            pc = config["pseudo_config"]
            key_vars["Q-Gen"] = pc.get("gen_type", "unknown")
            key_vars["K"] = pc.get("num_pseudo_queries", "unknown")
            key_vars["Pool"] = pc.get("pool_mode", "unknown")
            key_vars["Init"] = pc.get("init_mode", "unknown")
            key_vars["Align"] = pc.get("align_loss_type", "none")
            key_vars["Prior"] = pc.get("prior_loss_type", "none")
        else:
            key_vars["pseudo"] = False
        
        key_vars["seed"] = config.get("seed", "unknown")
        
        self.manifest.key_vars = key_vars
    
    def _print_run_header(self):
        """打印运行头信息"""
        print("\n" + "="*70)
        print(f"🚀 Starting Run: {self.run_id}")
        print("="*70)
        print(f"📁 Run directory: {self.run_dir}")
        print(f"🕐 Start time: {self.manifest.start_time}")
        print(f"\n🔧 Environment:")
        print(f"   Git: {self.manifest.env.git_commit[:8]}{'*' if self.manifest.env.git_dirty else ''} ({self.manifest.env.git_branch})")
        print(f"   PyTorch: {self.manifest.env.pytorch_version}")
        print(f"   CUDA: {self.manifest.env.cuda_version}")
        print(f"   GPU: {self.manifest.env.gpu_name}")
        
        if self.manifest.key_vars:
            print(f"\n🔑 Key Variables:")
            for k, v in self.manifest.key_vars.items():
                print(f"   {k}: {v}")


# ==================== 便捷函数 ====================

def create_run_manager(exp_name: str, output_root: str = "./outputs") -> RunManager:
    """创建RunManager的便捷函数"""
    return RunManager(exp_name, output_root)


if __name__ == "__main__":
    # 测试
    print("Testing RunManager...")
    
    manager = RunManager("test_exp", "./outputs/test")
    
    config = {
        "use_pseudo_query": True,
        "pseudo_config": {
            "gen_type": "heatmap",
            "num_pseudo_queries": 100,
            "pool_mode": "heatmap_weighted",
            "init_mode": "concat",
            "align_loss_type": "none",
            "prior_loss_type": "none",
        },
        "seed": 42,
    }
    
    manager.setup_run(config)
    manager.set_random_seed(42)
    manager.save_manifest()
    
    # 模拟训练
    manager.log_metrics(1, {"loss": 5.0, "recall_300": 0.1})
    manager.log_metrics(5, {"loss": 2.0, "recall_300": 0.3})
    
    manager.log_debug(
        action="Reduced learning rate",
        reason="Loss plateau at epoch 3",
        expected_impact="Slower but more stable convergence"
    )
    
    manager.finalize()
    
    print("\n✓ RunManager test passed!")
