import requests
import json
import sys


def call_pql_service_stream(scene, question, service_url):
    """
    调用 PQL 生成服务流式接口
    :param scene: 场景名（如 software_PSI）
    :param question: 中文自然语言问题
    :param service_url: 服务接口地址
    :return: 生成器，逐块返回结果
    """
    payload = {
        "scene": scene,
        "question": question
    }

    try:
        response = requests.post(
            service_url,
            json=payload,
            headers={"Content-Type": "application/json"},
            timeout=1800,
            stream=True
        )

        if response.status_code != 200:
            yield {
                "success": False,
                "data": None,
                "message": f"请求失败，状态码：{response.status_code}，详情：{response.text}"
            }
            return

        for line in response.iter_lines(decode_unicode=True):
            if line:
                try:
                    result = json.loads(line)

                    if result.get("code") == 200:
                        yield {
                            "success": True,
                            "data": result["result"],
                            "message": result["message"],
                            "status": result["result"].get("status", "unknown")
                        }
                    else:
                        yield {
                            "success": False,
                            "data": None,
                            "message": f"服务返回错误：{result.get('message', '未知错误')}",
                            "status": "error"
                        }
                        return
                except json.JSONDecodeError:
                    yield {
                        "success": False,
                        "data": None,
                        "message": f"响应解析错误：{line}",
                        "status": "error"
                    }
                    return

    except requests.exceptions.Timeout:
        yield {
            "success": False,
            "data": None,
            "message": "请求超时，请检查服务是否正常运行或延长超时时间",
            "status": "error"
        }
    except requests.exceptions.ConnectionError:
        yield {
            "success": False,
            "data": None,
            "message": "连接失败，请检查服务地址和端口是否正确",
            "status": "error"
        }
    except Exception as e:
        yield {
            "success": False,
            "data": None,
            "message": f"调用过程出错：{str(e)}",
            "status": "error"
        }


def main():
    # 解析命令行参数
    scene_list = ['software_PSI', 'software_MPC', 'software_PIR', 'hardware_PSI', 'hardware_MPC',
                  'hardware_PIR', 'hardware_PIRMPC', 'Federated_learning']
    scene = "software_PSI"
    question = "我想要实现一个求交的pql.分别使用a表的id和b表的id,最后选出a表的x1，和b表的y1"
    url = "http://localhost:5010/generate_pql_stream"  # 修改为流式接口

    # 调用服务
    print(f"正在调用 PQL 流式服务...\n场景：{scene}\n问题：{question}")
    print("\n===== 开始生成 =====")

    for result in call_pql_service_stream(scene=scene, question=question, service_url=url):
        if not result["success"]:
            print(f"\n错误：{result['message']}")
            break

        status = result.get("status", "unknown")
        data = result["data"]

        if status == "start":
            print("开始生成PQL查询...")
            print("PQL: ", end="", flush=True)

        elif status == "generating":
            current_pql = data["pql_query"]
            print(current_pql, end="", flush=True)
            sys.stdout.flush()

    print("====================")


if __name__ == "__main__":
    main()
