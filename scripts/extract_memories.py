
import re

# 你的 ai-stock 系统就具备了**“从错误中进化”的闭环能力。
# 每次复盘生成的这一句话，就是你量化系统不断升值的数字资产
def sync_memories():
    with open("logs/audit_records.md", "r") as f:
        content = f.read()
    
    # 提取所有带有 [MEMORY_ENTRY] 标签的行
    memories = re.findall(r'\[MEMORY_ENTRY\](.*)', content)
    
    # 将这些“精华”存入一个专门的知识库，供 Chronos 模型参考
    with open("data/model_memories.txt", "w") as f:
        for m in set(memories): # 去重
            f.write(f"- {m.strip()}\n")

