"""
FastAPI Web应用主入口
提供模型推理、训练监控和管理接口
"""

import asyncio
import json
import logging
import os
from datetime import datetime
from typing import Dict, List, Optional, Any
from pathlib import Path

import uvicorn
from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect, BackgroundTasks
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse, FileResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from config import TrainingConfig
from inference import ChatBot
from model_utils import ModelUtils
from data_processor import LCCCDataProcessor

# 设置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# FastAPI应用实例
app = FastAPI(
    title="Qwen3-0.6B LCCC QLoRA 微调项目",
    description="基于QLoRA技术的中文对话模型训练和推理平台",
    version="1.0.0"
)

# 添加CORS中间件
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 全局变量
chatbot: Optional[ChatBot] = None
config = TrainingConfig()
training_status = {
    "is_training": False,
    "current_step": 0,
    "total_steps": 0,
    "loss": 0.0,
    "status": "idle"
}

# WebSocket连接管理
class ConnectionManager:
    def __init__(self):
        self.active_connections: List[WebSocket] = []

    async def connect(self, websocket: WebSocket):
        await websocket.accept()
        self.active_connections.append(websocket)

    def disconnect(self, websocket: WebSocket):
        self.active_connections.remove(websocket)

    async def send_personal_message(self, message: dict, websocket: WebSocket):
        await websocket.send_text(json.dumps(message, ensure_ascii=False))

    async def broadcast(self, message: dict):
        for connection in self.active_connections:
            try:
                await connection.send_text(json.dumps(message, ensure_ascii=False))
            except:
                pass

manager = ConnectionManager()

# Pydantic模型
class ChatMessage(BaseModel):
    message: str
    history: Optional[List[List[str]]] = []
    max_new_tokens: Optional[int] = 100
    temperature: Optional[float] = 0.3
    top_p: Optional[float] = 0.8

class ChatResponse(BaseModel):
    response: str
    history: List[List[str]]
    timestamp: str

class TrainingRequest(BaseModel):
    epochs: Optional[int] = 3
    batch_size: Optional[int] = 2
    learning_rate: Optional[float] = 2e-4

class ModelInfo(BaseModel):
    model_name: str
    status: str
    last_modified: str
    size_mb: float

# 启动时加载模型
@app.on_event("startup")
async def startup_event():
    """应用启动时初始化"""
    logger.info("正在启动FastAPI应用...")
    await load_model()

async def load_model():
    """加载预训练模型"""
    global chatbot
    try:
        final_model_path = os.path.join(config.output_dir, "final_model")
        if os.path.exists(final_model_path):
            logger.info("正在加载微调后的模型...")
            chatbot = ChatBot(final_model_path, use_peft=True)
            logger.info("模型加载成功！")
        else:
            logger.warning("未找到微调后的模型，请先进行训练")
            chatbot = None
    except Exception as e:
        logger.error(f"模型加载失败: {str(e)}")
        chatbot = None

# API路由
@app.get("/")
async def root():
    """根路径，返回前端页面"""
    return FileResponse("web/index.html")

@app.get("/api/health")
async def health_check():
    """健康检查"""
    return {
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "model_loaded": chatbot is not None
    }

@app.post("/api/chat", response_model=ChatResponse)
async def chat_endpoint(message: ChatMessage):
    """聊天接口"""
    if chatbot is None:
        raise HTTPException(status_code=503, detail="模型未加载")
    
    try:
        # 转换历史记录格式
        history = message.history or []
        
        # 生成回复
        response = chatbot.generate_response(
            message.message,
            history=history,
            max_new_tokens=message.max_new_tokens or 100,
            temperature=message.temperature or 0.3,
            top_p=message.top_p or 0.8
        )
        
        # 更新历史记录
        new_history = history + [[message.message, response]]
        
        # 广播消息给WebSocket客户端
        await manager.broadcast({
            "type": "chat_message",
            "user_message": message.message,
            "bot_response": response,
            "timestamp": datetime.now().isoformat()
        })
        
        return ChatResponse(
            response=response,
            history=new_history,
            timestamp=datetime.now().isoformat()
        )
    except Exception as e:
        logger.error(f"聊天处理错误: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/models")
async def get_models():
    """获取可用模型列表"""
    models = []
    output_dir = Path(config.output_dir)
    
    if output_dir.exists():
        for model_dir in output_dir.iterdir():
            if model_dir.is_dir() and (model_dir / "adapter_config.json").exists():
                try:
                    stat = model_dir.stat()
                    size = sum(f.stat().st_size for f in model_dir.rglob('*') if f.is_file()) / (1024 * 1024)
                    
                    models.append(ModelInfo(
                        model_name=model_dir.name,
                        status="available",
                        last_modified=datetime.fromtimestamp(stat.st_mtime).isoformat(),
                        size_mb=round(size, 2)
                    ))
                except Exception as e:
                    logger.error(f"读取模型信息失败 {model_dir}: {e}")
    
    return {"models": models}

@app.post("/api/load_model/{model_name}")
async def load_specific_model(model_name: str):
    """加载指定模型"""
    global chatbot
    
    model_path = os.path.join(config.output_dir, model_name)
    if not os.path.exists(model_path):
        raise HTTPException(status_code=404, detail="模型不存在")
    
    try:
        logger.info(f"正在加载模型: {model_name}")
        chatbot = ChatBot(model_path, use_peft=True)
        
        await manager.broadcast({
            "type": "model_loaded",
            "model_name": model_name,
            "timestamp": datetime.now().isoformat()
        })
        
        return {"status": "success", "message": f"模型 {model_name} 加载成功"}
    except Exception as e:
        logger.error(f"模型加载失败: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/training/status")
async def get_training_status():
    """获取训练状态"""
    return training_status

@app.post("/api/training/start")
async def start_training(request: TrainingRequest, background_tasks: BackgroundTasks):
    """开始训练"""
    if training_status["is_training"]:
        raise HTTPException(status_code=400, detail="训练已在进行中")
    
    # 在后台启动训练任务
    background_tasks.add_task(run_training, request)
    
    return {"status": "success", "message": "训练已开始"}

@app.post("/api/training/stop")
async def stop_training():
    """停止训练"""
    if not training_status["is_training"]:
        raise HTTPException(status_code=400, detail="没有正在进行的训练")
    
    training_status["is_training"] = False
    training_status["status"] = "stopped"
    
    await manager.broadcast({
        "type": "training_stopped",
        "timestamp": datetime.now().isoformat()
    })
    
    return {"status": "success", "message": "训练已停止"}

@app.get("/api/training/logs")
async def get_training_logs():
    """获取训练日志"""
    log_file = "training.log"
    if os.path.exists(log_file):
        with open(log_file, 'r', encoding='utf-8') as f:
            logs = f.readlines()
        return {"logs": logs[-100:]}  # 返回最后100行
    return {"logs": []}

async def run_training(request: TrainingRequest):
    """运行训练任务"""
    global training_status
    
    try:
        training_status.update({
            "is_training": True,
            "current_step": 0,
            "total_steps": 630,  # 根据实际配置计算
            "status": "training"
        })
        
        await manager.broadcast({
            "type": "training_started",
            "timestamp": datetime.now().isoformat()
        })
        
        # 这里应该调用实际的训练代码
        # 由于训练是长时间运行的任务，这里模拟训练过程
        for step in range(1, 631):
            if not training_status["is_training"]:
                break
                
            training_status["current_step"] = step
            training_status["loss"] = 2.0 - (step / 630) * 1.5  # 模拟损失下降
            
            # 每10步广播一次状态
            if step % 10 == 0:
                await manager.broadcast({
                    "type": "training_progress",
                    "step": step,
                    "total_steps": 630,
                    "loss": training_status["loss"],
                    "timestamp": datetime.now().isoformat()
                })
            
            await asyncio.sleep(0.1)  # 模拟训练时间
        
        if training_status["is_training"]:
            training_status.update({
                "is_training": False,
                "status": "completed"
            })
            
            await manager.broadcast({
                "type": "training_completed",
                "timestamp": datetime.now().isoformat()
            })
            
            # 训练完成后重新加载模型
            await load_model()
        
    except Exception as e:
        logger.error(f"训练过程中出错: {str(e)}")
        training_status.update({
            "is_training": False,
            "status": "error"
        })
        
        await manager.broadcast({
            "type": "training_error",
            "error": str(e),
            "timestamp": datetime.now().isoformat()
        })

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    """WebSocket连接"""
    await manager.connect(websocket)
    try:
        while True:
            data = await websocket.receive_text()
            message = json.loads(data)
            
            # 处理不同类型的WebSocket消息
            if message.get("type") == "ping":
                await manager.send_personal_message(
                    {"type": "pong", "timestamp": datetime.now().isoformat()},
                    websocket
                )
    except WebSocketDisconnect:
        manager.disconnect(websocket)

# 静态文件服务
app.mount("/static", StaticFiles(directory="web"), name="static")

if __name__ == "__main__":
    # 确保web目录存在
    os.makedirs("web", exist_ok=True)
    
    uvicorn.run(
        "app:app",
        host="127.0.0.1",
        port=8000,
        reload=True,
        log_level="info"
    )
