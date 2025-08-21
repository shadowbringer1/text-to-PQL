import json
import time
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed
from openai import OpenAI

# OpenAI API 配置
client = OpenAI(
    api_key="sk-cySsnCdGiI8jysaHtQjbuOkoCEch2JDxYbrLLbyJX7sJm8Mu",
    base_url="https://lonlie.plus7.plus/v1"
)

# 每条数据生成几条增强数据
AUG_NUM = 3


def build_prompt(en_question, zh_question, pql_query, num):
    return f"""你是一个中英文双语的数据增强助手，请你基于下面这条数据，生成 {num} 条语义等价但表达方式不同的新数据。

要求如下：

1. 改写中英文问题，语义保持一致，语言自然、清晰、有多样性；
2. 改写平台名和表名，要求如下：
   - 请自由命名，使用英文单词、拼音、项目代号、地区代号、数字等；
   - **不要使用下划线（`_`）**；
   - 命名风格需多样化，例如：
     - 驼峰式（如：`DataNode`, `CloudVault`）
     - 拼音词（如：`ShujuTai`, `YongHuBiao`）
     - 项目名（如：`SecureX`, `NetZone9`）
     - 组合名（如：`EastPlatform7`, `UserServiceX`）
   - **不要使用模板化命名（如 platformA, tableX 等）**；
3. SQL 查询语句（PQL_query）同步更新，保持字段名、平台名、表名和问题中一致；
4. **注意：在中文问题中，平台名、表名、列名和字段名请保持英文原样，不要翻译，不要加中文解释，并且使问句更符合中文的语言习惯，流畅通顺**；
5. 字段名如 `id`, `userID`, `balance`, `accountNumber` 可以适当变化，保持语义一致；
6. 返回结果是 JSON 数组，每条结构如下：
[
  {{
    "question": "英文改写",
    "Chinese_question": "中文改写（不翻译平台、表、字段名）",
    "PQL_query": "改写后的 SQL 查询语句"
  }},
  ...
]

原始数据如下：
英文问题: {en_question}
中文问题: {zh_question}
PQL_query: {pql_query}
"""


# 调用大模型生成增强数据
def augment_sample(en_question, zh_question, pql_query, num=5):
    prompt = build_prompt(en_question, zh_question, pql_query, num)
    try:
        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": "你是一个中英文双语数据增强助手"},
                {"role": "user", "content": prompt}
            ],
            temperature=1.0,
            max_tokens=1500
        )
        content = response.choices[0].message.content.strip()
        json_start = content.find("[")
        json_data = content[json_start:]
        samples = json.loads(json_data)
        for s in samples:
            s["source"] = "augmented"
        return samples
    except Exception as e:
        print(f"❌ 大模型调用失败: {e}")
        return []


# 单个场景处理函数
def process_scene(scene_name, samples):
    print(f"\n📂 开始处理场景：{scene_name}（共 {len(samples)} 条数据）")
    new_samples = []
    for sample in tqdm(samples, desc=f"🔁 {scene_name}"):
        en_q = sample["question"]
        zh_q = sample["Chinese_question"]
        pql = sample["PQL_query"]

        # 添加原始数据
        new_samples.append({
            "question": en_q,
            "Chinese_question": zh_q,
            "PQL_query": pql,
            "source": "original"
        })

        # 添加增强数据
        augmented = augment_sample(en_q, zh_q, pql, num=AUG_NUM)
        new_samples.extend(augmented)

    print(f"✅ 场景 {scene_name} 处理完成，共 {len(new_samples)} 条数据")
    return scene_name, new_samples


if __name__ == "__main__":
    # 加载原始数据
    with open("/home/csu/text2PQL/PQL_generate/pql_dataset.json", "r", encoding="utf-8") as f:
        raw_data = json.load(f)

    augmented_data = {}

    # 使用线程池并发处理每个场景
    with ThreadPoolExecutor(max_workers=8) as executor:
        futures = [
            executor.submit(process_scene, scene_name, samples)
            for scene_name, samples in raw_data.items()
        ]

        for future in as_completed(futures):
            scene_name, new_samples = future.result()
            augmented_data[scene_name] = new_samples

    # 保存增强数据集
    with open("augmented_dataset.json", "w", encoding="utf-8") as f:
        json.dump(augmented_data, f, ensure_ascii=False, indent=2)

    print("\n✅ 所有场景数据增强完成！已保存为 'augmented_dataset.json'")
