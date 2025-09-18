# Qwen3-0.6B LCCC QLoRA 微调项目

本项目使用QLoRA技术对Qwen3-0.6B模型在LCCC（Large-scale Chinese Conversation Collection）数据集上进行微调，以提升中文对话能力。项目提供了完整的命令行工具和现代化的Web应用界面。

## ✨ 特性亮点

- 🧠 **QLoRA高效微调**: 4bit量化 + LoRA技术，大幅降低显存占用
- 💬 **中文对话优化**: 专门针对LCCC中文对话数据集进行优化
- 🌐 **现代化Web界面**: 响应式设计，支持实时对话和训练监控
- ⚙️ **可视化管理**: 模型训练、参数调优、状态监控一站式服务
- 🚀 **一键部署**: 支持命令行和Web两种使用方式

## 项目结构

```
llm_ft/
├── datasets/                  # 数据集目录
│   ├── LCCC-base_train.json  # 训练数据
│   ├── LCCC-base_valid.json  # 验证数据
│   └── LCCC-base_test.json   # 测试数据
├── web/                      # Web应用目录
│   ├── index.html            # 主页面
│   ├── style.css             # 样式文件
│   └── script.js             # 交互逻辑
├── config.py                 # 训练配置文件
├── data_processor.py         # 数据处理模块
├── model_utils.py            # 模型工具模块
├── train.py                  # 主训练脚本
├── inference.py              # 推理测试脚本
├── app.py                    # FastAPI Web应用
├── web_config.py             # Web配置文件
├── start_web.py              # Web启动脚本
├── start_web.bat             # Windows启动脚本
├── requirements.txt          # 依赖包列表
├── WEB使用说明.md           # Web应用使用说明
└── README.md                 # 说明文档
```

## 环境要求

- Python 3.8+
- CUDA 11.8+ (推荐)
- GPU内存 8GB+ (推荐)

## 安装依赖

```bash
pip install -r requirements.txt
```

## 主要特性

### QLoRA技术
- 使用4bit量化减少显存占用
- LoRA参数高效微调
- 支持多GPU训练

### 数据处理
- 自动处理LCCC对话格式
- 支持多轮对话训练
- 智能截断和填充

### 模型配置
- 基于Qwen3-0.6B
- 优化的超参数设置
- 自动保存检查点

### Web应用特性 🎆
- **实时对话界面**: 支持多轮对话，可调节生成参数
- **训练监控仪表板**: 实时方终数据可视化，进度跟踪
- **模型管理中心**: 一键切换不同检查点，模型对比
- **日志实时查看**: 训练过程实时反馈，错误诊断
- **响应式设计**: 支持桌面和移动设备访问

## 快速开始

### 🛠️ 环境准备

确保LCCC数据集文件在`datasets/`目录下：
- `LCCC-base_train.json` - 训练数据
- `LCCC-base_valid.json` - 验证数据
- `LCCC-base_test.json` - 测试数据

### 💻 命令行使用

#### 1. 安装依赖

```bash
pip install -r requirements.txt
```

#### 2. 配置调整

编辑`config.py`文件中的参数：

```python
class TrainingConfig:
    # 模型配置
    model_name = "Qwen/Qwen3-0.6B"
    
    # 训练参数
    num_train_epochs = 3
    per_device_train_batch_size = 2
    gradient_accumulation_steps = 4
    learning_rate = 2e-4
    
    # QLoRA配置
    lora_r = 16
    lora_alpha = 32
    lora_dropout = 0.1
```

#### 3. 开始训练

```bash
python train.py
```

训练日志将保存在`training.log`文件中，模型检查点保存在`outputs/`目录。

#### 4. 推理测试

**交互式对话**
```bash
python inference.py
```

**批量测试**
```bash
python inference.py --test
```

### 🌐 Web应用使用

#### 1. 快速启动

**Windows 用户**：
```bash
# 双击运行
start_web.bat
```

**或手动启动**：
```bash
python start_web.py
```

#### 2. 访问应用

打开浏览器访问：`http://127.0.0.1:8000`

#### 3. 功能使用

- **💬 对话模块**: 与 AI 进行实时对话，可调节生成参数
- **⚙️ 训练模块**: 配置训练参数，实时监控训练进度
- **🧠 模型管理**: 查看和切换不同的微调模型
- **📝 日志查看**: 实时查看训练和系统日志

## 配置说明

### 关键参数

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `lora_r` | 16 | LoRA的秩，控制适配器的复杂度 |
| `lora_alpha` | 32 | LoRA的缩放参数 |
| `max_seq_length` | 512 | 最大序列长度 |
| `per_device_train_batch_size` | 2 | 每设备批量大小 |
| `gradient_accumulation_steps` | 4 | 梯度累积步数 |
| `learning_rate` | 2e-4 | 学习率 |

### 显存优化

- 使用4bit量化：节省约50%显存
- 梯度检查点：减少激活值显存占用
- 小批量大小：适应有限GPU内存

### Web应用配置

在 `web_config.py` 中可以修改：

```python
class WebConfig:
    HOST = "127.0.0.1"      # 服务器地址
    PORT = 8000              # 服务器端口
    RELOAD = True            # 开发模式
    
    # 安全配置
    CORS_ORIGINS = ["*"]              # CORS设置
    MAX_MESSAGE_LENGTH = 2000         # 最大消息长度
    MAX_HISTORY_LENGTH = 20           # 最大历史记录长度
    
    # 默认参数
    DEFAULT_MAX_TOKENS = 100          # 默认最大token数
    DEFAULT_TEMPERATURE = 0.3         # 默认温度
    DEFAULT_TOP_P = 0.8              # 默认top-p值
```

## 数据格式

LCCC数据集格式示例：

```json
{
  "conversation": [
    "你好",
    "你好！有什么可以帮助你的吗？",
    "今天天气怎么样？",
    "今天天气很好，阳光明媚。"
  ]
}
```

处理后的训练格式：
```
<|im_start|>system
你是一个乐于助人的AI助手。<|im_end|>
<|im_start|>user
你好<|im_end|>
<|im_start|>assistant
你好！有什么可以帮助你的吗？<|im_end|>
```

## 训练监控

### 查看训练日志
```bash
tail -f training.log
```

### 关键指标
- `train_loss`: 训练损失
- `eval_loss`: 验证损失
- `learning_rate`: 当前学习率
- `epoch`: 训练轮次

### 内存监控
```bash
nvidia-smi
```

## 常见问题

### Q1: 显存不足怎么办？
A: 尝试以下方法：
- 减小`per_device_train_batch_size`
- 增加`gradient_accumulation_steps`
- 减小`max_seq_length`
- 启用梯度检查点

### Q2: 训练速度慢怎么办？
A: 
- 确保使用GPU训练
- 增加`per_device_train_batch_size`
- 使用多GPU训练
- 减小数据集大小进行测试

### Q3: 如何调整模型效果？
A:
- 增加训练轮次`num_train_epochs`
- 调整学习率`learning_rate`
- 修改LoRA参数`lora_r`和`lora_alpha`
- 使用更大的数据集

### Q4: 如何保存最终模型？
A: 训练完成后可以合并LoRA权重：

```python
from peft import PeftModel
from transformers import AutoModelForCausalLM

# 加载基础模型和LoRA权重
base_model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen3-0.6B")
model = PeftModel.from_pretrained(base_model, "./outputs/final_model")

# 合并权重
merged_model = model.merge_and_unload()

# 保存合并后的模型
merged_model.save_pretrained("./merged_model")
```

### Q5: Web应用无法访问怎么办？
A:
- 检查端口是否被占用，修改`web_config.py`中的端口号
- 确保防火墙允许访问对应端口
- 检查是否正确安装了FastAPI相关依赖

### Q6: 模型加载失败怎么办？
A:
- 确保`outputs/`目录下有微调模型
- 检查模型文件是否完整
- 如果是显存不足，减小batch size或使用更小的模型

## 性能优化建议

1. **硬件优化**
   - 使用V100/A100等高端GPU
   - 增加GPU内存
   - 使用NVMe SSD存储数据

2. **软件优化**
   - 启用混合精度训练
   - 使用更高效的优化器
   - 调整DataLoader参数

3. **数据优化**
   - 预处理数据并缓存
   - 使用更高效的数据格式
   - 并行数据加载

## 许可证

本项目遵循MIT许可证。

## 贡献指南

欢迎提交Issue和Pull Request来改进项目。

## 更新日志

- **v1.0.0**: 初始版本，支持基础QLoRA微调
- **v1.1.0**: 新增完整Web应用界面
  - 实时对话功能
  - 训练进度监控
  - 模型管理中心
  - 日志实时查看
  - 响应式设计支持
- 计划中: 支持更多模型和数据集

## 相关资源

- 📄 **[Web使用说明](WEB使用说明.md)**: 详细的Web应用使用指南
- 📉 **[项目报告](项目报告.md)**: 完整的项目技术报告
- 🧠 **[Qwen3-0.6B](https://huggingface.co/Qwen/Qwen3-0.6B)**: 基础模型
- 📊 **[LCCC数据集](https://github.com/thu-coai/CDial-GPT)**: 中文对话数据集

---

**享受您的AI对话体验！** 🤖✨