import torch
import argparse
import json
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel

# 示例路径写死在代码中
EXAMPLE_PATH = "scene_examples.json"

# 场景描述映射
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


# 构造Prompt（优化版）
def build_prompt(scene_name, chinese_question, scene_examples=None):
    scene_desc = scene_name_map.get(scene_name, "未知场景")
    prompt = f"""你是一个专注于“{scene_desc}”任务的 PQL 查询生成助手。请严格按照以下要求完成任务：

- 仅输出一条 **完整且可执行的 PQL 查询语句**
- **不要添加任何解释、说明、注释或格式提示**
- 输出必须以 `PQL：` 开头

"""
    if scene_examples:
        n = len(scene_examples)
        prompt += f"以下是该场景的 {n} 个示例：\n"
        for ex in scene_examples:
            prompt += f"问题：{ex['Chinese_question']}\nPQL：{ex['PQL_query']}\n"
    else:
        prompt += "该场景暂无示例。\n"

    prompt += f"\n请根据下方问题直接生成对应的 PQL 查询语句：\n问题：{chinese_question}\nPQL："
    return prompt

# 调用模型生成
def generate_pql(model, tokenizer, prompt, max_new_tokens=1024):
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            eos_token_id=tokenizer.eos_token_id
        )
    output_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    response = output_text[len(prompt):].strip()

    # 截断无关内容
    for stop_phrase in ["Human:", "User:", "Assistant:", "\n问题：", "\nPQL查询：", "\nPQL：", "\n提示：", "注意："]:
        if stop_phrase in response:
            response = response.split(stop_phrase)[0].strip()

    # 如果还有换行后内容也去掉
    response = response.split("\n")[0].strip()

    return response

# 主程序
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base_model_path", type=str, required=True)
    parser.add_argument("--lora_model_path", type=str, required=True)
    args = parser.parse_args()

    # 加载 few-shot 示例
    with open(EXAMPLE_PATH, "r", encoding="utf-8") as f:
        scene_examples = json.load(f)

    # 加载模型
    tokenizer = AutoTokenizer.from_pretrained(args.base_model_path, trust_remote_code=True)
    base_model = AutoModelForCausalLM.from_pretrained(
        args.base_model_path,
        torch_dtype=torch.float16,
        device_map="auto",
        trust_remote_code=True
    )
    model = PeftModel.from_pretrained(base_model, args.lora_model_path)
    model.eval()

    # 交互式推理
    while True:
        scene = input("请输入场景名（如 software_PSI），或输入 exit 退出：").strip()
        if scene.lower() == "exit":
            break
        if scene not in scene_name_map:
            print("无效的场景名，请重新输入。")
            continue

        question = input("请输入中文自然语言问题：").strip()
        examples = scene_examples.get(scene, [])
        prompt = build_prompt(scene, question, examples)
        output = generate_pql(model, tokenizer, prompt)

        print("\n生成的PQL查询：\n", output)
        print("-" * 50)

if __name__ == "__main__":
    main()