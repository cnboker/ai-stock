import os

# 定义目录结构
structure = [
    "app/api/v1",
    "app/core",
    "app/models",
    "app/schemas",
    "app/services",
    "app/db",
    "scripts",
    "data/vector_db",
    "logs"
]

# 定义核心文件
files = {
    "app/main.py": "# FastAPI Entry Point",
    "app/api/v1/analytics.py": "# Week 1: Hermes Review APIs",
    "app/api/v1/tasks.py": "# Week 2: Optuna Task APIs",
    "app/api/v1/trading.py": "# Week 4: Order Confirmation APIs",
    "app/core/config.py": "# Database and Env Configs",
    "app/models/prediction.py": "# Chronos Prediction Models",
    "app/models/order.py": "# Trading Order Models",
    "app/services/hermes_agent.py": "# Hermes Interaction Logic",
    "app/services/optuna_tuner.py": "# Optuna Optimization Logic",
    "scripts/daily_audit.py": "# Daily Review Script (Cron Task)",
    ".env": "DATABASE_URL=sqlite:///./ai_stock.db\nHERMES_API_KEY=your_key_here",
    "pyproject.toml": "# Dependency management"
}

def create_structure():
    # 1. 创建文件夹并添加 __init__.py
    for folder in structure:
        os.makedirs(folder, exist_ok=True)
        # 在所有子目录下创建 __init__.py 使其成为 python 包
        parts = folder.split('/')
        for i in range(1, len(parts) + 1):
            curr_path = os.path.join(*parts[:i])
            init_file = os.path.join(curr_path, "__init__.py")
            if not os.path.exists(init_file):
                with open(init_file, 'w') as f:
                    pass
        print(f"Created folder: {folder}")

    # 2. 创建核心文件
    for file_path, content in files.items():
        if not os.path.exists(file_path):
            with open(file_path, 'w', encoding='utf-8') as f:
                f.write(content)
            print(f"Created file: {file_path}")
        else:
            print(f"File already exists (skipped): {file_path}")

    print("\n✅ AI-Stock 自动化架构初始化完成！")

if __name__ == "__main__":
    create_structure()