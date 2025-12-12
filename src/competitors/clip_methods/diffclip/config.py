"""DiffCLIP配置加载器"""
import yaml
from pathlib import Path


def load_config(cfg_path=None):
    if cfg_path is None:
        current_dir = Path(__file__).parent
        project_root = current_dir.parent.parent.parent.parent
        cfg_path = project_root / 'configs' / 'methods' / 'diffclip_rs.yaml'
    with open(cfg_path, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f)

