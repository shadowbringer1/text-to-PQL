import torch
import argparse
import json
import os
from flask import Flask, request, jsonify, Response
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig, TextIteratorStreamer
from peft import PeftModel
import logging
import datetime
from threading import Thread

# 配置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# 设置CPU线程数（根据实际核心数调整）
os.environ["OMP_NUM_THREADS"] = "64"
os.environ["MKL_NUM_THREADS"] = "64"
torch.set_num_threads(64)

# 全局变量（模型和Tokenizer加载后缓存）
app = Flask(__name__)
model = None
tokenizer = None
scene_examples = None
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
EXAMPLE_PATH = "scene_examples.json"


# 构造Prompt
def build_prompt(scene_name, chinese_question, examples=None):
    scene_desc = scene_name_map.get(scene_name, "未知场景")
    prompt = f"""你是一个专注于"{scene_desc}"任务的 PQL 查询生成助手。请严格按照以下要求完成任务：

- 仅输出一条 **完整且可执行的 PQL 查询语句**
- **不要添加任何解释、说明、注释或格式提示**
- 输出必须以 `PQL：` 开头

"""
    if examples:
        n = len(examples)
        prompt += f"以下是该场景的 {n} 个示例：\n"
        for ex in examples:
            prompt += f"问题：{ex['Chinese_question']}\nPQL：{ex['PQL_query']}\n"
    else:
        prompt += "该场景暂无示例。\n"

    prompt += f"\n请根据下方问题直接生成对应的 PQL 查询语句：\n问题：{chinese_question}\nPQL："
    return prompt


# 生成PQL查询（流式版本）
def generate_pql_stream(prompt, max_new_tokens=1024):
    global model, tokenizer
    if not model or not tokenizer:
        raise RuntimeError("模型未加载，请检查服务启动日志")

    inputs = tokenizer(prompt, return_tensors="pt")

    # 创建流式生成器
    streamer = TextIteratorStreamer(
        tokenizer,
        skip_prompt=True,
        skip_special_tokens=True,
        timeout=300.0  # 设置超时时间
    )

    # 在单独线程中运行生成过程
    generation_kwargs = dict(
        **inputs,
        max_new_tokens=max_new_tokens,
        do_sample=False,
        eos_token_id=tokenizer.eos_token_id,
        pad_token_id=tokenizer.pad_token_id if tokenizer.pad_token_id else tokenizer.eos_token_id,
        repetition_penalty=1.0,
        no_repeat_ngram_size=0,
        streamer=streamer
    )

    thread = Thread(target=model.generate, kwargs=generation_kwargs)
    thread.start()

    for new_text in streamer:
        # 检查是否有停止短语，如果有则截断
        for stop_phrase in ["Human:", "User:", "Assistant:", "\n问题：", "\nPQL查询：", "\nPQL：", "\n提示：", "注意："]:
            if stop_phrase in new_text:
                yield new_text.split(stop_phrase)[0].strip()
                return

        for char in new_text:
            yield char


def load_resources(base_model_path, lora_model_path, quantization):
    global model, tokenizer, scene_examples

    # 加载示例数据
    try:
        with open(EXAMPLE_PATH, "r", encoding="utf-8") as f:
            scene_examples = json.load(f)
        logger.info(f"成功加载示例数据，共包含 {len(scene_examples)} 个场景示例")
    except FileNotFoundError:
        logger.warning(f"未找到示例文件 {EXAMPLE_PATH}，将不使用示例进行推理")
        scene_examples = {}

    # 配置量化参数
    quantization_config = None
    if quantization == "int8":
        quantization_config = BitsAndBytesConfig(
            load_in_8bit=True,
            llm_int8_threshold=6.0
        )
    elif quantization == "int4":
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.float32
        )

    # 加载Tokenizer
    logger.info(f"开始加载Tokenizer: {base_model_path}")
    tokenizer = AutoTokenizer.from_pretrained(
        base_model_path,
        trust_remote_code=True,
        padding_side="left"
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    logger.info("Tokenizer加载完成")

    # 加载基础模型
    logger.info(f"开始加载基础模型: {base_model_path}（量化方式：{quantization}）")
    base_model = AutoModelForCausalLM.from_pretrained(
        base_model_path,
        quantization_config=quantization_config,
        torch_dtype=torch.float16 if quantization == "float16" else torch.float32,
        device_map="cpu",
        low_cpu_mem_usage=True,
        trust_remote_code=True,
        use_safetensors=True,
    )
    logger.info("基础模型加载完成")

    # 加载LoRA模型
    logger.info(f"开始加载LoRA模型: {lora_model_path}")
    model = PeftModel.from_pretrained(base_model, lora_model_path)
    model.eval()  # 切换到评估模式
    logger.info("LoRA模型加载完成，服务准备就绪")


# API接口：接收scene和question，返回PQL结果（流式版本）
@app.route('/generate_pql_stream', methods=['POST'])
def api_generate_pql_stream():
    global scene_examples, scene_name_map
    data = request.json

    # 参数校验
    if not data or "scene" not in data or "question" not in data:
        return jsonify({
            "code": 400,
            "message": "缺少参数：请提供scene和question",
            "result": None
        }), 400

    scene = data["scene"].strip()
    question = data["question"].strip()

    if scene not in scene_name_map:
        return jsonify({
            "code": 400,
            "message": f"无效的场景名，可选场景：{list(scene_name_map.keys())}",
            "result": None
        }), 400

    # 生成PQL
    try:
        logger.info(f"接收流式请求：场景={scene}，问题={question[:50]}...")
        examples = scene_examples.get(scene, [])
        prompt = build_prompt(scene, question, examples)

        def event_stream():
            # 发送初始信息
            yield json.dumps({
                "code": 200,
                "message": "开始生成",
                "result": {
                    "scene": scene,
                    "scene_desc": scene_name_map[scene],
                    "question": question,
                    "pql_query": "",
                    "status": "start"
                }
            }) + "\n"

            # 流式生成PQL
            for chunk in generate_pql_stream(prompt):
                yield json.dumps({
                    "code": 200,
                    "message": "生成中",
                    "result": {
                        "scene": scene,
                        "scene_desc": scene_name_map[scene],
                        "question": question,
                        "pql_query": chunk,
                        "status": "generating"
                    }
                }) + "\n"

        return Response(event_stream(), mimetype='text/event-stream')

    except Exception as e:
        logger.error(f"生成失败：{str(e)}", exc_info=True)
        return jsonify({
            "code": 500,
            "message": f"生成失败：{str(e)}",
            "result": None
        }), 500


# 保留原有的非流式接口
@app.route('/generate_pql', methods=['POST'])
def api_generate_pql():
    global scene_examples, scene_name_map
    data = request.json

    # 参数校验
    if not data or "scene" not in data or "question" not in data:
        return jsonify({
            "code": 400,
            "message": "缺少参数：请提供scene和question",
            "result": None
        }), 400

    scene = data["scene"].strip()
    question = data["question"].strip()

    if scene not in scene_name_map:
        return jsonify({
            "code": 400,
            "message": f"无效的场景名，可选场景：{list(scene_name_map.keys())}",
            "result": None
        }), 400

    # 生成PQL
    try:
        logger.info(f"接收请求：场景={scene}，问题={question[:50]}...")
        examples = scene_examples.get(scene, [])
        prompt = build_prompt(scene, question, examples)

        # 使用流式生成但收集所有结果
        full_response = ""
        for chunk in generate_pql_stream(prompt):
            full_response += chunk

        logger.info("PQL生成成功")

        return jsonify({
            "code": 200,
            "message": "生成成功",
            "result": {
                "scene": scene,
                "scene_desc": scene_name_map[scene],
                "question": question,
                "pql_query": full_response
            }
        })
    except Exception as e:
        logger.error(f"生成失败：{str(e)}", exc_info=True)
        return jsonify({
            "code": 500,
            "message": f"生成失败：{str(e)}",
            "result": None
        }), 500


# 健康检查接口
@app.route('/health', methods=['GET'])
def health_check():
    global model, tokenizer
    if model and tokenizer:
        return jsonify({
            "code": 200,
            "message": "服务运行正常",
            "timestamp": datetime.datetime.now().isoformat()
        })
    else:
        return jsonify({
            "code": 503,
            "message": "服务未就绪",
            "timestamp": datetime.datetime.now().isoformat()
        }), 503


if __name__ == '__main__':
    # 解析启动参数
    parser = argparse.ArgumentParser()
    parser.add_argument("--base_model_path", type=str, required=True, help="基础模型路径")
    parser.add_argument("--lora_model_path", type=str, required=True, help="LoRA模型路径")
    parser.add_argument("--quantization", type=str, default="int8", choices=["int4", "int8", "float16"],
                        help="量化方式（CPU推荐int8或int4）")
    parser.add_argument("--host", type=str, default="0.0.0.0", help="服务监听地址")
    parser.add_argument("--port", type=int, default=5010, help="服务监听端口")
    args = parser.parse_args()

    # 加载资源（模型、数据）
    load_resources(args.base_model_path, args.lora_model_path, args.quantization)

    # 启动服务
    method = "http"
    logger.info(f"服务启动：{method}://{args.host}:{args.port}")
    app.run(host=args.host, port=args.port, threaded=True)