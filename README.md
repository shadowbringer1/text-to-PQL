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
  --base_model_path /data0/csu/models/Qwen-Qwen2.5-1.5B-Instruct \
  --lora_model_path ./checkpoints/qwen-lora-pql/final
```

- `--base_model_path`：填写原始基础模型的路径（如 Qwen2.5-1.5B-Instruct）
- `--lora_model_path`：填写你训练后保存的 LoRA 权重路径（默认为 `save_path/final`）

### 🔁 交互式使用方式

运行后将进入命令行交互界面：

- 首先输入任务场景名（如 `software_MPC`）
- 然后输入一条中文自然语言问题
- 模型将输出对应的 PQL 查询语句

### 示例：

```text
请输入场景名（如 software_PSI），或输入 exit 退出：software_MPC
请输入中文自然语言问题：在secure_exam平台的exam_results表中按年级分组的学生最高成绩是多少?

生成的PQL查询：
SELECT secure_exam.exam_results.grade, MAX(secure_exam.exam_results.score) FROM secure_exam.exam_results GROUP BY secure_exam.exam_results.grade;
```

你可以连续多轮交互，输入 `exit` 退出。

## 其他细节

- 模型架构：Qwen2.5（目前） + LoRA
- 支持 fp16 加速
- 数据加载与切分自动按场景 4:1 分为训练/验证集
- 使用 HuggingFace Transformers 与 Datasets 框架
- 推理输出自动裁切，防止多轮补全