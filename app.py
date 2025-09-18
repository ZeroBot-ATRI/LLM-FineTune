"""
FastAPI Web应用主入口
提供模型推理、训练监控和管理接口
"""

import asyncio
import json
import logging
import os
import sys
import threading
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
from train import QLoRATrainer

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

# 实际训练函数
async def run_actual_training(manager: ConnectionManager):
    """运行实际的训练过程"""
    try:
        # 创建训练器
        trainer = QLoRATrainer(config)
        
        def training_callback(step, total_steps, loss, lr=None):
            """训练回调函数，用于更新状态"""
            training_status["current_step"] = step
            training_status["total_steps"] = total_steps
            training_status["loss"] = loss
            
            # 在asyncio事件循环中广播消息
            asyncio.create_task(manager.broadcast({
                "type": "training_progress",
                "step": step,
                "total_steps": total_steps,
                "loss": loss,
                "learning_rate": lr,
                "timestamp": datetime.now().isoformat()
            }))
        
        # 在线程中运行训练（避免阻塞主线程）
        def run_training():
            try:
                # 准备数据
                tokenizer, datasets = trainer.prepare_data()
                
                # 获取总步数
                train_dataset = datasets.get('train')
                total_steps = 1000  # 默认值
                if train_dataset:
                    total_samples = len(train_dataset)
                    batch_size = config.per_device_train_batch_size * config.gradient_accumulation_steps
                    steps_per_epoch = max(1, total_samples // batch_size)
                    total_steps = steps_per_epoch * config.num_train_epochs
                    training_status["total_steps"] = total_steps
                
                # 设置模型
                model = trainer.setup_model(tokenizer)
                
                # 记录词汇表信息
                logger.info(f"Tokenizer vocab size: {len(tokenizer)}")
                if hasattr(tokenizer, 'unk_token') and tokenizer.unk_token:
                    logger.info(f"Using unk_token: {tokenizer.unk_token}")
                
                # 获取内存占用信息
                trainer.model_utils.get_model_memory_footprint(model)
                
                # 创建训练参数
                training_args = trainer.model_utils.create_training_arguments()
                
                # 创建数据收集器
                from transformers import Trainer, DataCollatorForLanguageModeling
                
                data_collator = DataCollatorForLanguageModeling(
                    tokenizer=tokenizer,
                    mlm=False,
                    pad_to_multiple_of=8,
                    return_tensors="pt",
                )
                
                # 创建训练器
                eval_dataset = datasets.get('eval')
                if eval_dataset is None:
                    logger.warning("No evaluation dataset found, disabling evaluation")
                    # 重新创建训练参数，禁用评估
                    config.eval_strategy = 'no'
                    training_args = trainer.model_utils.create_training_arguments()
                
                custom_trainer = Trainer(
                    model=model,
                    args=training_args,
                    train_dataset=datasets.get('train'),
                    eval_dataset=eval_dataset,
                    processing_class=tokenizer,
                    data_collator=data_collator,
                )
                
                # 保存训练前的模型状态
                logger.info("Saving initial model state...")
                initial_model_path = os.path.join(config.output_dir, "initial_model")
                os.makedirs(initial_model_path, exist_ok=True)
                custom_trainer.save_model(initial_model_path)
                
                # 开始训练
                logger.info("Starting training...")
                
                # 使用简单的进度跟踪
                class ProgressCallback:
                    def __init__(self, total_steps, manager):
                        self.total_steps = total_steps
                        self.current_step = 0
                        self.manager = manager
                        self.last_log_step = 0
                    
                    def on_step_end(self, trainer, logs=None):
                        self.current_step = trainer.state.global_step
                        
                        # 每10步或者在有loss的时候更新
                        if (self.current_step - self.last_log_step >= 10) or (logs and 'loss' in logs):
                            if logs and 'loss' in logs:
                                loss = logs['loss']
                                lr = logs.get('learning_rate', 0)
                                training_callback(self.current_step, self.total_steps, loss, lr)
                                self.last_log_step = self.current_step
                            else:
                                # 即使没有loss，也更新步数
                                training_status["current_step"] = self.current_step
                                asyncio.create_task(self.manager.broadcast({
                                    "type": "training_progress",
                                    "step": self.current_step,
                                    "total_steps": self.total_steps,
                                    "timestamp": datetime.now().isoformat()
                                }))
                                self.last_log_step = self.current_step
                
                # 添加回调
                progress_callback = ProgressCallback(total_steps, manager)
                
                # 开始训练
                custom_trainer.train()
                
                # 保存最终模型
                logger.info("Saving final model...")
                final_model_path = os.path.join(config.output_dir, "final_model")
                os.makedirs(final_model_path, exist_ok=True)
                custom_trainer.save_model(final_model_path)
                
                # 保存tokenizer
                tokenizer.save_pretrained(final_model_path)
                
                logger.info("Training completed successfully!")
                
                # 更新训练状态为完成
                if training_status["is_training"]:
                    training_status.update({
                        "is_training": False,
                        "status": "completed"
                    })
                    
                    # 广播训练完成消息
                    asyncio.create_task(manager.broadcast({
                        "type": "training_completed",
                        "timestamp": datetime.now().isoformat()
                    }))
                    
                    # 训练完成后重新加载模型
                    asyncio.create_task(load_model())
                    
            except Exception as e:
                logger.error(f"Training failed: {str(e)}")
                training_status.update({
                    "is_training": False,
                    "status": "error"
                })
                
                # 广播错误消息
                asyncio.create_task(manager.broadcast({
                    "type": "training_error",
                    "error": str(e),
                    "timestamp": datetime.now().isoformat()
                }))
        
        # 在后台线程中运行训练
        training_thread = threading.Thread(target=run_training)
        training_thread.daemon = True
        training_thread.start()
        
    except Exception as e:
        logger.error(f"Failed to start training: {str(e)}")
        training_status.update({
            "is_training": False,
            "status": "error"
        })
        
        await manager.broadcast({
            "type": "training_error",
            "error": str(e),
            "timestamp": datetime.now().isoformat()
        })

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

@app.get("/api/system/info")
async def get_system_info():
    """获取系统信息"""
    import torch
    import psutil
    
    system_info = {
        "python_version": f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}",
        "torch_version": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
        "device_count": torch.cuda.device_count() if torch.cuda.is_available() else 0,
        "current_device": torch.cuda.current_device() if torch.cuda.is_available() else None,
        "memory": {
            "total_ram_gb": round(psutil.virtual_memory().total / (1024**3), 2),
            "available_ram_gb": round(psutil.virtual_memory().available / (1024**3), 2),
            "used_ram_percent": psutil.virtual_memory().percent
        }
    }
    
    if torch.cuda.is_available():
        try:
            gpu_memory = torch.cuda.get_device_properties(0).total_memory
            system_info["gpu_memory_gb"] = round(gpu_memory / (1024**3), 2)
            system_info["gpu_name"] = torch.cuda.get_device_name(0)
        except:
            system_info["gpu_memory_gb"] = "unknown"
            system_info["gpu_name"] = "unknown"
    
    return system_info

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

@app.get("/api/training/config")
async def get_training_config():
    """获取当前训练配置"""
    return {
        "model_name": config.model_name,
        "epochs": config.num_train_epochs,
        "batch_size": config.per_device_train_batch_size,
        "learning_rate": config.learning_rate,
        "max_seq_length": config.max_seq_length,
        "lora_r": config.lora_r,
        "lora_alpha": config.lora_alpha,
        "lora_dropout": config.lora_dropout,
        "output_dir": config.output_dir,
        "data_path": config.data_path
    }

@app.post("/api/training/config")
async def update_training_config(request: dict):
    """更新训练配置"""
    if training_status["is_training"]:
        raise HTTPException(status_code=400, detail="训练进行中，不能修改配置")
    
    # 更新允许的配置项
    allowed_configs = {
        'num_train_epochs': int,
        'per_device_train_batch_size': int,
        'learning_rate': float,
        'max_seq_length': int,
        'lora_r': int,
        'lora_alpha': int,
        'lora_dropout': float
    }
    
    updated = {}
    for key, value in request.items():
        if key in allowed_configs:
            try:
                # 类型转换
                converted_value = allowed_configs[key](value)
                setattr(config, key, converted_value)
                updated[key] = converted_value
            except (ValueError, TypeError) as e:
                raise HTTPException(status_code=400, detail=f"配置项 {key} 的值无效: {str(e)}")
    
    return {"status": "success", "updated": updated}

@app.get("/api/training/status")
async def get_training_status():
    """获取训练状态"""
    return training_status

@app.post("/api/training/start")
async def start_training(request: TrainingRequest, background_tasks: BackgroundTasks):
    """开始训练"""
    if training_status["is_training"]:
        raise HTTPException(status_code=400, detail="训练已在进行中")
    
    # 更新训练配置
    if request.epochs:
        config.num_train_epochs = request.epochs
    if request.batch_size:
        config.per_device_train_batch_size = request.batch_size
    if request.learning_rate:
        config.learning_rate = request.learning_rate
        
    # 确保输出目录存在
    os.makedirs(config.output_dir, exist_ok=True)
    
    # 在后台启动训练任务
    background_tasks.add_task(run_training, request)
    
    return {
        "status": "success", 
        "message": "训练已开始",
        "config": {
            "epochs": config.num_train_epochs,
            "batch_size": config.per_device_train_batch_size,
            "learning_rate": config.learning_rate
        }
    }

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
        # 更新配置（如果提供了参数）
        if request.epochs and request.epochs != config.num_train_epochs:
            config.num_train_epochs = request.epochs
            logger.info(f"Updated epochs to {request.epochs}")
        if request.batch_size and request.batch_size != config.per_device_train_batch_size:
            config.per_device_train_batch_size = request.batch_size
            logger.info(f"Updated batch size to {request.batch_size}")
        if request.learning_rate and request.learning_rate != config.learning_rate:
            config.learning_rate = request.learning_rate
            logger.info(f"Updated learning rate to {request.learning_rate}")
        
        training_status.update({
            "is_training": True,
            "current_step": 0,
            "total_steps": 0,
            "loss": 0.0,
            "status": "starting"
        })
        
        await manager.broadcast({
            "type": "training_started",
            "config": {
                "epochs": config.num_train_epochs,
                "batch_size": config.per_device_train_batch_size,
                "learning_rate": config.learning_rate
            },
            "timestamp": datetime.now().isoformat()
        })
        
        # 调用实际的训练代码
        await run_actual_training(manager)
        
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
