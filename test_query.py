from flask import Flask, request, jsonify
import time

app = Flask(__name__)


@app.route("/v1/mira/openapi/Query", methods=["POST"])
def query():
    # 打印请求头（用于调试）
    print("Headers:", request.headers)

    # 打印请求体（JSON 数据）
    request_data = request.get_json()
    print("Request Body:", request_data)

    time.sleep(31)

    # 检查必要的请求头（模拟 Go 代码中的逻辑）
    required_headers = ["sign", "timestamp", "appId", "Authorization"]
    for header in required_headers:
        if header not in request.headers:
            return jsonify({"error": f"Missing header: {header}"}), 400

    # 模拟返回的响应数据
    response_data = {
        "status": "success",
        "result": {
            "input": request_data.get("input", {}),
            "async": request_data.get("async", False),
            "requestId": request_data.get("requestId", ""),
            "jobId": request_data.get("jobId", ""),
        },
        "message": "Query processed successfully"
    }

    return jsonify(response_data), 200


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5005, debug=True)