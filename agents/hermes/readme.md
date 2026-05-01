# 开始生成今日审计报告

### 操作说明：


以后每天你实际调用Hermes时，只需要：
- 打开 `daily_prompt_template.md`
- 把 `{YYYY-MM-DD}` 替换成今天日期
- 把 `{daily_audit_template_content}` 替换成你刚才创建的 `daily_audit_template.md` 的完整内容
- 把今日的JSON数据粘贴到对应位置
- 然后把整个prompt发给Hermes

---

每天生成审计报告的标准流程：

准备阶段
拿到当天的原始信号JSON数据。

调用Hermes
打开 daily_prompt_template.md，按以下方式填充后发给Hermes：
把 {YYYY-MM-DD} 替换为今天日期（如 2026-04-30）
把 {daily_audit_template_content} 替换为 daily_audit_template.md 文件的完整内容
在 “今日原始信号数据” 处粘贴当天的JSON数据

Hermes生成报告后
把生成的完整报告保存为 audits/2026-04-30_audit.md
把Hermes输出的更新后的 recent_lessons 内容，替换 recent_lessons.md 文件

每周一次（建议周末做）
让Hermes阅读最近7天的审计报告，做一次“元复盘”，决定是否更新 core_lessons.md