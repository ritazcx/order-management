# test_api.py
import pytest
from fastapi.testclient import TestClient
from unittest.mock import Mock, patch, MagicMock
import sys
import os

# 添加项目根目录到 Python 路径
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.api.main import app
from src.api.schema import TicketRequest, TicketResponse

# ========== Fixtures ==========
@pytest.fixture
def client():
    """创建测试客户端"""
    return TestClient(app)

@pytest.fixture
def mock_predictor():
    """创建模拟的 Predictor"""
    mock = Mock()
    mock.get_model_version.return_value = {"version": "v1.0"}
    mock.predict.return_value = {
        "category": "Hardware Problem",
        "severity": "High",
        "confidence": 0.95
    }
    return mock

# ========== 健康检查测试 ==========
def test_health_check(client):
    """测试健康检查端点"""
    response = client.get("/healthz")
    
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}

# ========== 版本端点测试 ==========
def test_get_version_success(client, mock_predictor):
    """测试成功获取版本信息"""
    # 模拟 predictor 已加载
    app.state.predictor = mock_predictor
    
    response = client.get("/version")
    
    assert response.status_code == 200
    assert response.json() == {"version": "v1.0"}
    
    # 验证调用了正确的方法
    mock_predictor.get_model_version.assert_called_once()

def test_get_version_predictor_not_loaded(client):
    """测试 predictor 未加载的情况"""
    # 确保 predictor 为 None
    app.state.predictor = None
    
    response = client.get("/version")
    
    assert response.status_code == 503
    assert response.json()["detail"] == "Predictor not loaded"

# ========== 预测端点测试 ==========
def test_predict_success(client, mock_predictor):
    """测试成功预测"""
    # 设置 predictor
    app.state.predictor = mock_predictor
    
    test_text = "我的电脑无法启动，显示蓝屏错误"
    request_data = {"text": test_text}
    
    response = client.post("/predict", json=request_data)
    
    assert response.status_code == 200
    data = response.json()
    assert data["category"] == "Hardware Problem"
    assert data["severity"] == "High"
    assert data["confidence"] == 0.95
    
    # 验证调用了 predict 方法
    mock_predictor.predict.assert_called_once_with(test_text)


def test_predict_empty_text(client, mock_predictor):
    """测试空文本输入"""
    app.state.predictor = mock_predictor
    
    # 空字符串
    request_data = {"text": ""}
    response = client.post("/predict", json=request_data)
    assert response.status_code == 400
    assert "cannot be empty" in response.json()["detail"]
    
    # 只有空格
    request_data = {"text": "   "}
    response = client.post("/predict", json=request_data)
    assert response.status_code == 400
    assert "cannot be empty" in response.json()["detail"]
    
    # None 值
    request_data = {"text": None}
    response = client.post("/predict", json=request_data)
    assert response.status_code == 422  # Pydantic 验证错误

def test_predict_predictor_not_loaded(client):
    """测试 predictor 未加载的情况"""
    app.state.predictor = None
    
    request_data = {"text": "测试文本"}
    response = client.post("/predict", json=request_data)
    
    assert response.status_code == 503
    assert response.json()["detail"] == "Predictor not loaded"

def test_predict_predictor_error(client, mock_predictor):
    """测试 predictor 抛出异常的情况"""
    app.state.predictor = mock_predictor
    
    # 模拟 predict 方法抛出异常
    mock_predictor.predict.side_effect = Exception("模型推理错误")
    
    request_data = {"text": "测试文本"}
    response = client.post("/predict", json=request_data)
    
    # 注意：你的代码中没有捕获 predictor.predict() 的异常
    # 所以会返回 500 错误
    assert response.status_code == 500

def test_predict_invalid_json(client):
    """测试无效的 JSON 输入"""
    response = client.post("/predict", data="invalid json")
    assert response.status_code == 422

# ========== 模型不同输出的测试 ==========
@pytest.mark.parametrize("input_text, expected_category, expected_severity", [
    ("无法登录邮箱", "Email Issue", "Medium"),
    ("网络连接失败", "Network Issue", "High"),
    ("Excel 文件无法保存", "Office/Excel Issue", "Low"),
    ("软件崩溃", "Software Bug", "High"),
])
def test_predict_different_inputs(client, mock_predictor, input_text, expected_category, expected_severity):
    """测试不同输入文本的预测"""
    app.state.predictor = mock_predictor
    
    # 为不同输入设置不同的返回值
    mock_predictor.predict.return_value = {
        "category": expected_category,
        "severity": expected_severity,
        "confidence": 0.92
    }
    
    request_data = {"text": input_text}
    response = client.post("/predict", json=request_data)
    
    assert response.status_code == 200
    data = response.json()
    assert data["category"] == expected_category
    assert data["severity"] == expected_severity

# ========== 启动和关闭事件测试 ==========
def test_startup_event():
    """测试应用启动事件"""
    from src.api.main import startup_event
    
    # 保存原始状态
    original_predictor = getattr(app.state, "predictor", None)
    
    try:
        # 模拟 Predictor 加载成功
        with patch('src.api.main.Predictor') as mock_predictor_class:
            mock_instance = Mock()
            mock_predictor_class.return_value = mock_instance
            
            startup_event()
            
            # 验证 Predictor 被实例化
            mock_predictor_class.assert_called_once()
            assert app.state.predictor == mock_instance
    finally:
        # 恢复原始状态
        app.state.predictor = original_predictor

def test_startup_event_with_error():
    """测试启动时 Predictor 加载失败"""
    from src.api.main import startup_event
    
    original_predictor = getattr(app.state, "predictor", None)
    
    try:
        # 模拟 Predictor 初始化抛出异常
        with patch('src.api.main.Predictor') as mock_predictor_class:
            mock_predictor_class.side_effect = Exception("加载模型失败")
            
            startup_event()
            
            # 验证 predictor 被设置为 None
            assert app.state.predictor is None
    finally:
        app.state.predictor = original_predictor

def test_shutdown_event():
    """测试应用关闭事件"""
    from src.api.main import shutdown_event
    
    # 设置一个 predictor
    mock_predictor = Mock()
    app.state.predictor = mock_predictor
    
    shutdown_event()
    
    # 验证 predictor 被清除
    assert app.state.predictor is None

# ========== 端点文档测试 ==========
def test_openapi_docs(client):
    """测试 API 文档端点"""
    # 测试 Swagger UI
    response = client.get("/docs")
    assert response.status_code == 200
    
    # 测试 ReDoc
    response = client.get("/redoc")
    assert response.status_code == 200
    
    # 测试 OpenAPI JSON
    response = client.get("/openapi.json")
    assert response.status_code == 200
    
    openapi_spec = response.json()
    assert "info" in openapi_spec
    assert openapi_spec["info"]["title"] == "Order Management ML API"
    assert "paths" in openapi_spec
    assert "/predict" in openapi_spec["paths"]
    assert "/healthz" in openapi_spec["paths"]

# ========== 模型验证测试 ==========
def test_ticket_request_validation():
    """测试请求模型的验证"""
    # 有效请求
    valid_request = TicketRequest(text="有效的请求文本")
    assert valid_request.text == "有效的请求文本"
    
    # 测试最小长度（如果需要）
    # 你可以在 TicketRequest 模型中添加约束
    # 例如：text: str = Field(..., min_length=1)

# ========== 边缘情况测试 ==========
def test_long_text_input(client, mock_predictor):
    """测试长文本输入"""
    app.state.predictor = mock_predictor
    
    long_text = "这是一个非常长的文本..." * 1000
    request_data = {"text": long_text}
    
    response = client.post("/predict", json=request_data)
    
    assert response.status_code == 200
    mock_predictor.predict.assert_called_once_with(long_text)

def test_special_characters(client, mock_predictor):
    """测试特殊字符输入"""
    app.state.predictor = mock_predictor
    
    special_text = "Test with special chars: @#$%^&*()_+{}|:\"<>?~`[]\\;',./"
    request_data = {"text": special_text}
    
    response = client.post("/predict", json=request_data)
    
    assert response.status_code == 200
    mock_predictor.predict.assert_called_once_with(special_text)

# ========== 并发测试 ==========
def test_concurrent_requests(client, mock_predictor):
    """测试并发请求"""
    import threading
    
    app.state.predictor = mock_predictor
    
    results = []
    errors = []
    
    def make_request():
        try:
            response = client.post("/predict", json={"text": "测试文本"})
            results.append(response.status_code)
        except Exception as e:
            errors.append(str(e))
    
    # 创建多个线程并发请求
    threads = []
    for _ in range(10):
        thread = threading.Thread(target=make_request)
        threads.append(thread)
        thread.start()
    
    for thread in threads:
        thread.join()
    
    # 验证所有请求都成功
    assert len(errors) == 0
    assert all(code == 200 for code in results)
    assert mock_predictor.predict.call_count == 10

# ========== 请求头测试 ==========
def test_cors_headers(client):
    """测试 CORS 头部"""
    response = client.get("/healthz")
    
    # 检查常见的 CORS 头部
    headers = response.headers
    # 你可以根据需要添加特定的 CORS 头部检查
    assert "content-type" in headers

def test_rate_limiting(client, mock_predictor):
    """测试速率限制（如果实现的话）"""
    app.state.predictor = mock_predictor
    
    # 快速连续发送多个请求
    for i in range(5):
        response = client.post("/predict", json={"text": f"请求 {i}"})
        assert response.status_code == 200