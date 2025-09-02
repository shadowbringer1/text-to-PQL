import torch
import argparse
import json
import os
from flask import Flask, request, jsonify, Response
from transformers import AutoTokenizer, AutoModelForCausalLM, TextIteratorStreamer
from peft import PeftModel
import logging
import datetime
from threading import Thread

# -------------------------- 基础配置 --------------------------
# 日志配置：记录服务运行状态和错误信息
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# CPU 线程配置（根据实际硬件核心数调整）
os.environ["OMP_NUM_THREADS"] = "128"
os.environ["MKL_NUM_THREADS"] = "128"
torch.set_num_threads(128)

# 全局变量：缓存模型、Tokenizer和示例数据（避免重复加载）
app = Flask(__name__)
model = None  # 加载后的 LoRA 模型
tokenizer = None  # 加载后的 Tokenizer
scene_examples = None  # 场景示例数据
EXAMPLE_PATH = "scene_examples.json"  # 示例数据路径

# 场景名称映射（与原始代码完全一致）
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


# -------------------------- 核心工具函数 --------------------------
def build_prompt(scene_name, chinese_question, examples=None):
    """
    构造 PQL 生成提示词（与原始代码逻辑完全一致）
    :param scene_name: 场景名（如 software_PSI）
    :param chinese_question: 用户中文问题
    :param examples: 该场景的 few-shot 示例
    :return: 完整提示词
    """
    scene_desc = scene_name_map.get(scene_name, "未知场景")
    prompt = f"""你是一个专注于“{scene_desc}”任务的 PQL 查询生成助手。请严格按照以下要求完成任务：

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


def generate_pql_stream(prompt, max_new_tokens=1024):
    """
    流式生成 PQL 查询（基于原始生成逻辑改造，支持逐字符返回）
    :param prompt: 构造好的提示词
    :param max_new_tokens: 最大生成 tokens 数
    :yield: 生成的 PQL 片段（逐字符）
    """
    global model, tokenizer
    if not model or not tokenizer:
        raise RuntimeError("模型未加载，请检查服务启动日志")

    # 处理输入：与原始代码一致，确保设备匹配
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

    # 流式生成器配置：跳过提示词和特殊 tokens
    streamer = TextIteratorStreamer(
        tokenizer,
        skip_prompt=True,  # 跳过提示词部分，只返回生成内容
        skip_special_tokens=True,  # 跳过 EOS/PAD 等特殊 tokens
        timeout=300.0  # 超时时间（防止线程挂起）
    )

    # 生成参数：完全遵循原始代码逻辑（不采样、指定停止 token）
    generation_kwargs = dict(
        **inputs,
        max_new_tokens=max_new_tokens,
        do_sample=False,  # 确定性生成（与原始一致）
        eos_token_id=tokenizer.eos_token_id,  # 停止 token（与原始一致）
        pad_token_id=tokenizer.eos_token_id,  # 防止 Padding Warning（与原始一致）
        streamer=streamer  # 绑定流式生成器
    )

    # 启动异步生成线程（避免阻塞主线程）
    thread = Thread(target=model.generate, kwargs=generation_kwargs)
    thread.start()

    # 逐段处理生成结果，按原始逻辑截断无关内容
    for new_text in streamer:
        # 截断停止短语（与原始代码的停止短语列表完全一致）
        for stop_phrase in ["Human:", "User:", "Assistant:", "\n问题：", "\nPQL查询：", "\nPQL：", "\n提示：", "注意："]:
            if stop_phrase in new_text:
                yield new_text.split(stop_phrase)[0].strip()
                return  # 遇到停止短语直接返回，终止生成

        # 逐字符返回（确保流式体验）
        for char in new_text:
            yield char


def load_resources(base_model_path, lora_model_path):
    """
    加载模型、Tokenizer 和示例数据（完全遵循原始代码逻辑，不添加量化）
    :param base_model_path: 基础模型路径
    :param lora_model_path: LoRA 模型路径
    """
    global model, tokenizer, scene_examples

    # 1. 加载场景示例数据（与原始代码一致）
    try:
        with open(EXAMPLE_PATH, "r", encoding="utf-8") as f:
            scene_examples = json.load(f)
        logger.info(f"成功加载示例数据，共包含 {len(scene_examples)} 个场景示例")
    except FileNotFoundError:
        logger.warning(f"未找到示例文件 {EXAMPLE_PATH}，将不使用示例进行推理")
        scene_examples = {}

    # 2. 加载 Tokenizer（与原始代码一致）
    logger.info(f"开始加载 Tokenizer: {base_model_path}")
    tokenizer = AutoTokenizer.from_pretrained(
        base_model_path,
        trust_remote_code=True  # 支持自定义模型（与原始一致）
    )
    # 确保 Pad Token 存在（与原始代码一致）
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    logger.info("Tokenizer 加载完成")

    # 3. 加载基础模型（无量化，与原始代码完全一致）
    logger.info(f"开始加载基础模型: {base_model_path}（无量化）")
    base_model = AutoModelForCausalLM.from_pretrained(
        base_model_path,
        torch_dtype=torch.float16,  # 与原始代码一致（使用 float16）
        device_map="auto",  # 自动分配设备（CPU/GPU，与原始一致）
        trust_remote_code=True,  # 支持自定义模型（与原始一致）
        low_cpu_mem_usage=True  # 低内存占用优化
    )
    logger.info("基础模型加载完成")

    # 4. 加载 LoRA 模型（与原始代码一致）
    logger.info(f"开始加载 LoRA 模型: {lora_model_path}")
    model = PeftModel.from_pretrained(base_model, lora_model_path)
    model.eval()  # 切换到评估模式（禁止 Dropout，与原始一致）
    logger.info("LoRA 模型加载完成，服务准备就绪")


# -------------------------- API 接口 --------------------------
@app.route('/generate_pql_stream', methods=['POST'])
def api_generate_pql_stream():
    """
    流式 PQL 生成接口（POST）
    请求体：{"scene": "场景名", "question": "中文问题"}
    响应：逐段返回 JSON 格式的生成结果（Content-Type: text/event-stream）
    """
    global scene_examples, scene_name_map
    data = request.json

    # 1. 参数校验
    if not data or "scene" not in data or "question" not in data:
        return jsonify({
            "code": 400,
            "message": "缺少必填参数：scene（场景名）和 question（中文问题）",
            "result": None
        }), 400

    scene = data["scene"].strip()
    question = data["question"].strip()

    # 2. 场景合法性校验
    if scene not in scene_name_map:
        valid_scenes = list(scene_name_map.keys())
        return jsonify({
            "code": 400,
            "message": f"无效场景名！可选场景：{valid_scenes}",
            "result": None
        }), 400

    # 3. 流式生成并返回结果
    try:
        logger.info(f"接收流式请求：场景={scene}，问题={question[:50]}...")
        examples = scene_examples.get(scene, [])  # 获取该场景的示例（无示例则为空）
        prompt = build_prompt(scene, question, examples)  # 构造提示词

        def event_stream():
            """生成 SSE 格式的流式响应"""
            # 发送启动信号
            yield json.dumps({
                "code": 200,
                "message": "PQL 生成开始",
                "result": {
                    "scene": scene,
                    "scene_desc": scene_name_map[scene],
                    "question": question,
                    "pql_query": "",
                    "status": "start"
                },
                "timestamp": datetime.datetime.now().isoformat()
            }) + "\n"

            # 流式返回生成结果
            for char in generate_pql_stream(prompt):
                yield json.dumps({
                    "code": 200,
                    "message": "PQL 生成中",
                    "result": {
                        "scene": scene,
                        "scene_desc": scene_name_map[scene],
                        "question": question,
                        "pql_query": char,
                        "status": "generating"
                    },
                    "timestamp": datetime.datetime.now().isoformat()
                }) + "\n"

            # 发送结束信号
            yield json.dumps({
                "code": 200,
                "message": "PQL 生成完成",
                "result": {
                    "scene": scene,
                    "scene_desc": scene_name_map[scene],
                    "question": question,
                    "pql_query": char,
                    "status": "completed"
                },
                "timestamp": datetime.datetime.now().isoformat()
            }) + "\n"

        # 返回流式响应（SSE 格式）
        return Response(event_stream(), mimetype='text/event-stream')

    except Exception as e:
        logger.error(f"流式生成失败：{str(e)}", exc_info=True)
        return jsonify({
            "code": 500,
            "message": f"生成失败：{str(e)}",
            "result": None
        }), 500


@app.route('/generate_pql', methods=['POST'])
def api_generate_pql():
    """
    普通 PQL 生成接口（POST，非流式）
    请求体：{"scene": "场景名", "question": "中文问题"}
    响应：生成完成后返回完整 JSON 结果
    """
    global scene_examples, scene_name_map
    data = request.json

    # 1. 参数校验（与流式接口一致）
    if not data or "scene" not in data or "question" not in data:
        return jsonify({
            "code": 400,
            "message": "缺少必填参数：scene（场景名）和 question（中文问题）",
            "result": None
        }), 400

    scene = data["scene"].strip()
    question = data["question"].strip()

    # 2. 场景合法性校验（与流式接口一致）
    if scene not in scene_name_map:
        valid_scenes = list(scene_name_map.keys())
        return jsonify({
            "code": 400,
            "message": f"无效场景名！可选场景：{valid_scenes}",
            "result": None
        }), 400

    # 3. 生成完整 PQL 并返回
    try:
        logger.info(f"接收普通请求：场景={scene}，问题={question[:50]}...")
        examples = scene_examples.get(scene, [])
        prompt = build_prompt(scene, question, examples)

        # 收集流式生成的完整结果（复用流式逻辑，确保一致性）
        full_pql = ""
        for char in generate_pql_stream(prompt):
            full_pql += char

        logger.info(f"普通生成完成：场景={scene}，PQL长度={len(full_pql)}")
        return jsonify({
            "code": 200,
            "message": "PQL 生成成功",
            "result": {
                "scene": scene,
                "scene_desc": scene_name_map[scene],
                "question": question,
                "pql_query": full_pql
            },
            "timestamp": datetime.datetime.now().isoformat()
        })

    except Exception as e:
        logger.error(f"普通生成失败：{str(e)}", exc_info=True)
        return jsonify({
            "code": 500,
            "message": f"生成失败：{str(e)}",
            "result": None
        }), 500


@app.route('/health', methods=['GET'])
def health_check():
    """
    服务健康检查接口（GET）
    用于监控服务是否正常运行（模型是否加载完成）
    """
    global model, tokenizer
    if model and tokenizer:
        return jsonify({
            "code": 200,
            "message": "服务运行正常",
            "status": "healthy",
            "timestamp": datetime.datetime.now().isoformat(),
            "supported_scenes": list(scene_name_map.keys())
        })
    else:
        return jsonify({
            "code": 503,
            "message": "服务未就绪（模型未加载）",
            "status": "unhealthy",
            "timestamp": datetime.datetime.now().isoformat()
        }), 503


# -------------------------- 服务启动入口 --------------------------
if __name__ == '__main__':
    # 解析启动参数（移除量化相关参数，只保留模型路径和服务配置）
    parser = argparse.ArgumentParser()
    parser.add_argument("--base_model_path", type=str, required=True,
                        help="基础模型路径（如本地文件夹或Hugging Face仓库名）")
    parser.add_argument("--lora_model_path", type=str, required=True, help="LoRA模型路径（本地微调后的LoRA文件夹）")
    parser.add_argument("--host", type=str, default="0.0.0.0", help="服务监听地址（默认0.0.0.0，支持外部访问）")
    parser.add_argument("--port", type=int, default=5010, help="服务监听端口（默认5010）")
    args = parser.parse_args()

    # 加载模型和资源（无量化）
    load_resources(args.base_model_path, args.lora_model_path)

    # 启动 Flask 服务（threaded=True 支持多线程并发）
    logger.info(f"PQL 生成服务启动成功：http://{args.host}:{args.port}")
    logger.info(f"支持场景：{list(scene_name_map.keys())}")
    app.run(
        host=args.host,
        port=args.port,
        threaded=True,  # 启用多线程处理请求
        use_reloader=False  # 禁用自动重载（生产环境推荐）
    )
