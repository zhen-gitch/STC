# src/paths.py
from pathlib import Path

# src/paths.py 的上一级就是 STC/ 根目录
ROOT_DIR = Path(__file__).resolve().parent.parent

# 集中定义所有核心目录
CONFIG_DIR = ROOT_DIR / 'configs'
LOGS_DIR = ROOT_DIR / 'logs'
MODELS_DIR = ROOT_DIR / 'src' / 'models'

# 如果文件夹不存在，自动创建它们 (防止第一次跑项目时报错)
LOGS_DIR.mkdir(parents=True, exist_ok=True)

if __name__ == "__main__":
    print(ROOT_DIR)
    print(CONFIG_DIR)
    print(LOGS_DIR)