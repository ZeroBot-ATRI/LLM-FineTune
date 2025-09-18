"""
模型加载和QLoRA配置工具
"""

import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    TrainingArguments
)
from peft import (
    LoraConfig,
    get_peft_model,
    TaskType,
    prepare_model_for_kbit_training
)
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ModelUtils:
    def __init__(self, config):
        self.config = config
        
    def create_bnb_config(self):
        """创建BitsAndBytesConfig用于4bit量化"""
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=self.config.load_in_4bit,
            bnb_4bit_quant_type=self.config.bnb_4bit_quant_type,
            bnb_4bit_compute_dtype=getattr(torch, self.config.bnb_4bit_compute_dtype),
            bnb_4bit_use_double_quant=self.config.bnb_4bit_use_double_quant,
        )
        return bnb_config
    
    def load_tokenizer(self):
        """加载并配置tokenizer"""
        logger.info(f"Loading tokenizer from {self.config.tokenizer_name}")
        
        tokenizer = AutoTokenizer.from_pretrained(
            self.config.tokenizer_name,
            trust_remote_code=True,
            use_fast=False
        )
        
        # 设置特殊token
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        
        # 确保有unk_token，用于处理未知词汇
        if tokenizer.unk_token is None:
            # 检查常见的UNK token
            vocab = tokenizer.get_vocab()
            if "<unk>" in vocab:
                tokenizer.unk_token = "<unk>"
                logger.info("Set unk_token to <unk>")
            elif "[UNK]" in vocab:
                tokenizer.unk_token = "[UNK]"
                logger.info("Set unk_token to [UNK]")
            elif "<|endoftext|>" in vocab:
                tokenizer.unk_token = "<|endoftext|>"
                logger.info("Set unk_token to <|endoftext|>")
            else:
                # 如果都没有，使用eos_token作为unk_token
                tokenizer.unk_token = tokenizer.eos_token
                logger.warning(f"No standard unk_token found, using eos_token: {tokenizer.eos_token}")
        else:
            logger.info(f"Found existing unk_token: {tokenizer.unk_token}")
            
        # 添加聊天模板（如果需要）
        if not hasattr(tokenizer, 'chat_template') or tokenizer.chat_template is None:
            tokenizer.chat_template = (
                "{% for message in messages %}"
                "{% if message['role'] == 'system' %}"
                "<|im_start|>system\n{{ message['content'] }}<|im_end|>\n"
                "{% elif message['role'] == 'user' %}"
                "<|im_start|>user\n{{ message['content'] }}<|im_end|>\n"
                "{% elif message['role'] == 'assistant' %}"
                "<|im_start|>assistant\n{{ message['content'] }}<|im_end|>"
                "{% endif %}"
                "{% endfor %}"
            )
        
        logger.info(f"Tokenizer配置完成:")
        logger.info(f"  - 词汇表大小: {len(tokenizer)}")
        logger.info(f"  - pad_token: {tokenizer.pad_token}")
        logger.info(f"  - eos_token: {tokenizer.eos_token}")
        logger.info(f"  - unk_token: {tokenizer.unk_token}")
        logger.info(f"  - unk_token_id: {getattr(tokenizer, 'unk_token_id', 'None')}")
        
        return tokenizer
    
    def load_model(self):
        """加载并配置模型"""
        logger.info(f"Loading model from {self.config.model_name}")
        
        # 创建量化配置
        bnb_config = self.create_bnb_config() if self.config.use_qlora else None
        
        # 加载模型
        model = AutoModelForCausalLM.from_pretrained(
            self.config.model_name,
            quantization_config=bnb_config,
            device_map="auto",
            trust_remote_code=True,
            torch_dtype=torch.float16 if self.config.fp16 else torch.bfloat16,
            use_cache=False  # 训练时禁用缓存
        )
        
        # 如果使用QLoRA，准备模型进行kbit训练
        if self.config.use_qlora:
            model = prepare_model_for_kbit_training(model)
        
        logger.info("Model loaded successfully")
        return model
    
    def create_lora_config(self):
        """创建LoRA配置"""
        lora_config = LoraConfig(
            r=self.config.lora_r,
            lora_alpha=self.config.lora_alpha,
            target_modules=self.config.target_modules,
            lora_dropout=self.config.lora_dropout,
            bias=self.config.lora_bias,
            task_type=TaskType.CAUSAL_LM,
        )
        return lora_config
    
    def setup_peft_model(self, model):
        """设置PEFT模型"""
        if not self.config.use_qlora:
            return model
            
        logger.info("Setting up PEFT model with LoRA")
        lora_config = self.create_lora_config()
        model = get_peft_model(model, lora_config)
        
        # 打印可训练参数
        model.print_trainable_parameters()
        
        return model
    
    def create_training_arguments(self):
        """创建训练参数"""
        training_args = TrainingArguments(
            output_dir=self.config.output_dir,
            num_train_epochs=self.config.num_train_epochs,
            per_device_train_batch_size=self.config.per_device_train_batch_size,
            per_device_eval_batch_size=self.config.per_device_eval_batch_size,
            gradient_accumulation_steps=self.config.gradient_accumulation_steps,
            optim=self.config.optim,
            save_steps=self.config.save_steps,
            logging_steps=self.config.logging_steps,
            learning_rate=self.config.learning_rate,
            weight_decay=self.config.weight_decay,
            fp16=self.config.fp16,
            bf16=self.config.bf16,
            max_grad_norm=self.config.max_grad_norm,
            max_steps=self.config.max_steps,
            warmup_ratio=self.config.warmup_ratio,
            group_by_length=self.config.group_by_length,
            lr_scheduler_type=self.config.lr_scheduler_type,
            eval_strategy=self.config.eval_strategy,
            eval_steps=self.config.eval_steps,
            save_strategy=self.config.save_strategy,
            load_best_model_at_end=self.config.load_best_model_at_end,
            metric_for_best_model=self.config.metric_for_best_model,
            greater_is_better=self.config.greater_is_better,
            report_to=self.config.report_to,
            remove_unused_columns=self.config.remove_unused_columns,
            dataloader_pin_memory=self.config.dataloader_pin_memory,
        )
        return training_args
    
    def get_model_memory_footprint(self, model):
        """获取模型内存占用"""
        param_count = sum(p.numel() for p in model.parameters())
        param_size = sum(p.numel() * p.element_size() for p in model.parameters())
        buffer_size = sum(buf.numel() * buf.element_size() for buf in model.buffers())
        
        logger.info(f"Model parameters: {param_count:,}")
        logger.info(f"Parameter memory: {param_size / 1024**2:.2f} MB")
        logger.info(f"Buffer memory: {buffer_size / 1024**2:.2f} MB")
        logger.info(f"Total memory: {(param_size + buffer_size) / 1024**2:.2f} MB")
        
        return {
            'param_count': param_count,
            'param_size_mb': param_size / 1024**2,
            'buffer_size_mb': buffer_size / 1024**2,
            'total_size_mb': (param_size + buffer_size) / 1024**2
        }

def test_model_loading():
    """测试模型加载"""
    from config import TrainingConfig
    
    config = TrainingConfig()
    model_utils = ModelUtils(config)
    
    # 加载tokenizer
    tokenizer = model_utils.load_tokenizer()
    
    # 加载模型
    model = model_utils.load_model()
    
    # 设置PEFT
    model = model_utils.setup_peft_model(model)
    
    # 获取内存占用
    memory_info = model_utils.get_model_memory_footprint(model)
    
    print("Model loading test completed successfully!")
    return model, tokenizer

if __name__ == "__main__":
    test_model_loading()