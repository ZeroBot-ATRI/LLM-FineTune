"""
Web应用配置文件
"""

import os
from pathlib import Path

class WebConfig:
    """Web应用配置类"""
    
    # 服务器配置
    HOST = "127.0.0.1"
    PORT = 8000
    RELOAD = True  # 开发模式
    LOG_LEVEL = "info"
    
    # 文件路径配置
    BASE_DIR = Path(__file__).parent
    WEB_DIR = BASE_DIR / "web"
    OUTPUTS_DIR = BASE_DIR / "outputs"
    LOGS_DIR = BASE_DIR
    
    # 静态文件配置
    STATIC_DIR = WEB_DIR
    STATIC_URL = "/static"
    
    # WebSocket配置
    WS_HEARTBEAT_INTERVAL = 30  # 秒
    WS_RECONNECT_INTERVAL = 5   # 秒
    
    # 训练配置
    MAX_TRAINING_HISTORY = 1000  # 最大保存的训练步数
    LOG_REFRESH_INTERVAL = 2     # 日志刷新间隔（秒）
    
    # 安全配置
    CORS_ORIGINS = ["*"]  # 生产环境中应该设置具体的域名
    MAX_MESSAGE_LENGTH = 2000
    MAX_HISTORY_LENGTH = 20
    
    # 模型配置
    DEFAULT_MAX_TOKENS = 100
    DEFAULT_TEMPERATURE = 0.3
    DEFAULT_TOP_P = 0.8
    
    # 文件上传配置
    MAX_FILE_SIZE = 10 * 1024 * 1024  # 10MB
    ALLOWED_EXTENSIONS = {'.json', '.txt', '.csv'}
    
    @classmethod
    def get_log_file(cls):
        """获取日志文件路径"""
        return cls.LOGS_DIR / "training.log"
    
    @classmethod
    def get_web_log_file(cls):
        """获取Web应用日志文件路径"""
        return cls.LOGS_DIR / "web_app.log"
    
    @classmethod
    def ensure_directories(cls):
        """确保必要的目录存在"""
        cls.WEB_DIR.mkdir(exist_ok=True)
        cls.OUTPUTS_DIR.mkdir(exist_ok=True)
        cls.LOGS_DIR.mkdir(exist_ok=True)