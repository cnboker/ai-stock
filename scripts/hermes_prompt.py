import os
import json
from datetime import datetime
import re

# 配置路径
BASE_PATH = 'agents/hermes'
TEMPLATE_FILE = 'daily_prompt_template.md'
AUDIT_TEMPLATE_FILE = 'daily_audit_template.md'
LESSONS_FILE = 'recent_lessons.md'
AUDIT_OUTPUT_DIR = f'{BASE_PATH}/audits'

def prepare_hermes_prompt(json_data):
    """
    步骤 1: 读取模板并填充内容
    """
    today_str = datetime.now().strftime('%Y-%m-%d')
    
    # 读取文件内容
    with open(f"{BASE_PATH}/{TEMPLATE_FILE}", 'r', encoding='utf-8') as f:
        prompt_content = f.read()
    
    with open(f"{BASE_PATH}/{AUDIT_TEMPLATE_FILE}", 'r', encoding='utf-8') as f:
        audit_template_content = f.read()

    # 替换占位符
    prompt_content = prompt_content.replace('{YYYY-MM-DD}', today_str)
    audit_template_content = audit_template_content.replace('{YYYY-MM-DD}', today_str)  
    prompt_content = prompt_content.replace('{daily_audit_template_content}', audit_template_content)
    
    # 填充 JSON 数据
    # 确保 JSON 格式化美观以便 Hermes 阅读
    json_str = json.dumps(json_data, indent=2, ensure_ascii=False)
    prompt_content = prompt_content.replace('[在这里粘贴今日完整的原始信号JSON数据]', f'{json_str}')

    return prompt_content, today_str


def process_hermes_smart_save(hermes_output, date_str):
    """
    智能保存逻辑：
    1. 如果包含 START 标记，则拆分：上方内容存报告，中间内容存 lessons。
    2. 如果不包含 START 标记，则将全文保存为报告。
    """
    
    # 确保审计目录存在
    if not os.path.exists('audits'):
        os.makedirs('audits')

    start_marker = "--- RECENT_LESSONS_UPDATE_START ---"
    end_marker = "--- RECENT_LESSONS_UPDATE_END ---"
    report_path = f"{AUDIT_OUTPUT_DIR}/{date_str}_audit.md"

    if start_marker in hermes_output:
        # --- 情况 A：包含标记，执行拆分逻辑 ---
        
        # 1. 提取标记以上的内容
        header_part = hermes_output.split(start_marker)[0].strip()
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write(header_part)
        print(f"✅ 已找到标记，审计报告已保存至: {report_path}")

        # 2. 提取标记中间的内容更新 recent_lessons.md
        pattern = f"{start_marker}(.*?){end_marker}"
        match = re.search(pattern, hermes_output, re.DOTALL)
        
        if match:
            new_lessons = match.group(1).strip()
            with open(f'{BASE_PATH}/recent_lessons.md', 'w', encoding='utf-8') as f:
                f.write(new_lessons)
            print(f"✅ recent_lessons.md 已根据提取内容更新")
    else:
        # --- 情况 B：不包含标记，全文保存 ---
        
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write(hermes_output.strip())
        print(f"ℹ️ 未找到标记，全文已保存为报告: {report_path}")
        
