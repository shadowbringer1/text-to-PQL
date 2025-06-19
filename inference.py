import torch
import argparse
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel

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

def build_prompt(scene_key, chinese_question):
    scene_desc = scene_name_map.get(scene_key, "未知场景")
    return f"""你是一个专注于{scene_desc}任务的PQL生成助手，只需输出一条合法的PQL语句，不要添加任何解释或注释。请根据下方中文问题直接生成对应的PQL语句。

问题：{chinese_question}
PQL："""

def generate_pql(model, tokenizer, prompt, max_new_tokens=64):
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
    for stop_phrase in ["Human:", "User:", "Assistant:", "\n问题：", "\nPQL查询：", "\nPQL：", "\n提示："]:
        if stop_phrase in response:
            response = response.split(stop_phrase)[0].strip()
    return response

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base_model_path", type=str, required=True)
    parser.add_argument("--lora_model_path", type=str, required=True)
    args = parser.parse_args()

    tokenizer = AutoTokenizer.from_pretrained(args.base_model_path, trust_remote_code=True)
    base_model = AutoModelForCausalLM.from_pretrained(
        args.base_model_path,
        torch_dtype=torch.float16,
        device_map="auto",
        trust_remote_code=True
    )
    model = PeftModel.from_pretrained(base_model, args.lora_model_path)
    model.eval()

    while True:
        scene = input("请输入场景名（如 software_PSI），或输入 exit 退出：").strip()
        if scene.lower() == "exit":
            break
        if scene not in scene_name_map:
            print("无效的场景名，请重新输入。")
            continue
        question = input("请输入中文自然语言问题：").strip()
        prompt = build_prompt(scene, question)
        output = generate_pql(model, tokenizer, prompt)
        print("\n生成的PQL查询：\n", output)
        print("-" * 50)

if __name__ == "__main__":
    main()