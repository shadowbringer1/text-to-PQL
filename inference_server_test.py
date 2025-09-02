import requests
import json
import sys
import time


def time_compute(func):
    def warpper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()
        print(f"Function '{func.__name__}' executed in {end_time - start_time:.4f} seconds")
        return result

    return warpper


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


@time_compute
def main():
    scene_question_dict = {
        'software_PSI': "我想要实现一个求交的pql，分别使用a表的id和b表的id，最后选出a表的x1和b表的y1",
        'software_MPC': "使用软件MPC实现两表联合计算，分别使用a表的id和b表的id， 基于用户ID匹配，计算用户的平均消费金额和总订单数，两个表分别为a和b",
        'software_PIR': "通过软件PIR技术查询id为10086的用户信息，使用a表，但不泄露查询的具体id",
        'hardware_PSI': "基于硬件加速的PSI实现，对比本地客户表c和远程会员表d，找出共同用户的姓名name和联系方式tel",
        'hardware_MPC': "使用硬件MPC完成多方数据联合统计，分别使用a表的id和b表的id，计算不同地区的用户平均年龄和性别分布",
        'hardware_PIR': "通过软件PIR技术查询id为10086的用户信息，使用a表，但不泄露查询的具体id",
        'hardware_PIRMPC': "结合硬件PIR和MPC技术，分别使用a表的id和b表的id,在保护隐私的前提下计算两个部门的销售数据总和与平均值",
        'Federated_learning': "构建联邦学习模型逻辑回归，分别使用a表的id和b表的id，特征使用全量"
    }
    url = "http://localhost:5010/generate_pql_stream"  # 修改为流式接口
    for scene, question in scene_question_dict.items():
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

        print("\n===== 结束生成 =====")


if __name__ == "__main__":
    main()