# README

本项目基于大语言模型（如 Qwen2.5-Instruct）进行 PQL 查询语句的生成，使用中文自然语言问题作为输入，输出结构化的 PQL 查询。训练过程中使用 LoRA技术对预训练模型进行高效微调，应用场景覆盖隐私计算、联邦学习等多个子任务。

---

## 数据格式说明

数据文件为 JSON 格式，结构如下，按任务场景组织：

```json
{
  "software_PSI": [
    {
      "question": "...",
      "Chinese_question": "如何通过id字段获取...",
      "PQL_query": "SELECT ..."
    }
  ],
  "software_MPC": [...],
  ...
}
```

共支持 8 类任务场景：

```python
scene_name_map = {
    "software_PSI": "软件PSI",
    "software_MPC": "软件MPC",
    "software_PIR": "软件PIR",
    "hardware_PSI": "硬件PSI",
    "hardware_MPC": "硬件MPC",
    "hardware_PIR": "硬件PIR",
    "hardware_PIRMPC": "硬件PIRMPC",
    "Federated_learning": "联邦学习"
}
```

## 模型训练（Fine-tuning）

使用以下命令进行模型微调：

```bash
python finetune.py configs/train.json
```

修改`configs/train.json`文件中的`model_path`参数为基础模型的路径，也可以修改文件中的训练超参数等。

## 模型推理（Inference）

运行以下命令启动交互式推理：

```bash
python inference.py \
  --base_model_path /data1/public/hf/Qwen/
   \
  --lora_model_path ./checkpoints/qwen-lora-pql/checkpoint-441
```

- `--base_model_path`：填写原始基础模型的路径（如 Qwen2.5-7B-Instruct）
- `--lora_model_path`：填写你训练后保存的 LoRA 权重路径

### 🔁 交互式使用方式

运行后将进入命令行交互界面：

- 首先输入任务场景名（如 `software_MPC`）
- 然后输入一条中文自然语言问题
- 模型将输出对应的 PQL 查询语句

### 示例：

```text
请输入场景名（如 software_PSI），或输入 exit 退出：software_PIR
请输入中文自然语言问题：如何统计在ida_en_one平台的m_enterprise_1w表中余额超过1000的记录数?

生成的PQL查询：
 SELECT COUNT(*) FROM ida_en_one.m_enterprise_1w WHERE ida_en_one.m_enterprise_1w.balance > 1000;
 --------------------------------------------------
请输入场景名（如 software_PSI），或输入 exit 退出：software_MPC
请输入中文自然语言问题：当ida_en_one平台m_enterprise_1w表和ida_en_two平台m_security_1w表的id相同时,通过安全计算求balance字段的和。
生成的PQL查询：
 SELECT ida_en_one.m_enterprise_1w.balance + ida_en_two.m_security_1w.balance FROM ida_en_one.m_enterprise_1w, ida_en_two.m_security_1w WHERE ida_en_one.m_enterprise_1w.id = ida_en_two.m_security_1w.id;
--------------------------------------------------
请输入场景名（如 software_PSI），或输入 exit 退出：exit
```

你可以连续多轮交互，输入 `exit` 退出。

## 模型评估（Evaluate）
你可以使用评估脚本对模型在测试集上的表现进行评估，采用严格字符串匹配的方式计算准确率。

### 数据要求

测试集为 JSON 格式，文件为`./PQL_generate/test_data.json`，每条样本包含：

```json
{
  "scene": "Federated_learning",
  "Chinese_question": "如何利用plat31的employee_records来训练HELR模型...",
  "PQL_query": "SELECT TRAIN(...) FROM ...;"
}
```

### 评估命令
```bash
python evaluate.py \
  --base_model_path /data1/public/hf/Qwen/Qwen2.5-7B-Instruct \
  --lora_model_path ./checkpoints/qwen-lora-pql/checkpoint-441 \
  --test_file ./PQL_generate/test_data.json
```

### 输出结果
最终在终端中输出整体准确率：

```text
总样本数: 100
正确预测数: 83
准确率: 83.00%
```
此外，会将详细预测结果保存为 `evaluation_results.json`。

## 其他细节

- 模型架构：Qwen2.5（目前） + LoRA
- 支持 fp16 加速
- 数据加载与切分自动按场景 4:1 分为训练/验证集
- 使用 HuggingFace Transformers 与 Datasets 框架
- 推理输出自动裁切，防止多轮补全