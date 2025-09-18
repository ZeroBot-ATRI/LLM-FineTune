#!/usr/bin/env python3
"""
Web应用启动脚本
提供训练、推理、管理的完整Web界面
"""

import os
import sys
import argparse
import logging
from pathlib import Path

def setup_logging():
    """设置日志"""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler('web_app.log'),
            logging.StreamHandler(sys.stdout)
        ]
    )

def check_dependencies():
    """检查依赖是否安装"""
    required_packages = [
        'fastapi', 'uvicorn', 'websockets', 'pydantic',
        'torch', 'transformers', 'peft', 'bitsandbytes'
    ]
    
    missing_packages = []
    for package in required_packages:
        try:
            __import__(package)
        except ImportError:
            missing_packages.append(package)
    
    if missing_packages:
        print(f"缺少以下依赖包: {', '.join(missing_packages)}")
        print("请运行: pip install -r requirements.txt")
        return False
    
    return True

def check_models():
    """检查是否有可用模型"""
    outputs_dir = Path("outputs")
    if not outputs_dir.exists():
        print("警告: outputs目录不存在，请先进行模型训练")
        return False
    
    model_dirs = [d for d in outputs_dir.iterdir() 
                  if d.is_dir() and (d / "adapter_config.json").exists()]
    
    if not model_dirs:
        print("警告: 未找到可用的微调模型，请先进行训练")
        return False
    
    print(f"找到 {len(model_dirs)} 个可用模型")
    return True

def main():
    """主函数"""
    parser = argparse.ArgumentParser(description='Qwen3-0.6B QLoRA Web应用')
    parser.add_argument('--host', default='127.0.0.1', help='服务器地址')
    parser.add_argument('--port', type=int, default=8000, help='服务器端口')
    parser.add_argument('--reload', action='store_true', help='开发模式，自动重载')
    parser.add_argument('--log-level', default='info', help='日志级别')
    
    args = parser.parse_args()
    
    # 设置日志
    setup_logging()
    logger = logging.getLogger(__name__)
    
    logger.info("正在启动Qwen3-0.6B QLoRA Web应用...")
    
    # 检查依赖
    if not check_dependencies():
        sys.exit(1)
    
    # 检查模型
    check_models()
    
    # 确保web目录存在
    web_dir = Path("web")
    if not web_dir.exists():
        logger.error("web目录不存在，请确保前端文件已正确安装")
        sys.exit(1)
    
    # 启动应用
    try:
        import uvicorn
        logger.info(f"启动Web服务器: http://{args.host}:{args.port}")
        uvicorn.run(
            "app:app",
            host=args.host,
            port=args.port,
            reload=args.reload,
            log_level=args.log_level
        )
    except KeyboardInterrupt:
        logger.info("应用已停止")
    except Exception as e:
        logger.error(f"启动失败: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()