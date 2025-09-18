"""
主训练脚本
使用QLoRA对Qwen3-0.6B进行微调
"""

import os
import sys
import torch
import logging
from transformers import Trainer, DataCollatorForLanguageModeling
from config import TrainingConfig
from model_utils import ModelUtils
from data_processor import LCCCDataProcessor

# 设置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('training.log'),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

class QLoRATrainer:
    def __init__(self, config):
        self.config = config
        self.model_utils = ModelUtils(config)
        
    def prepare_data(self):
        """准备训练数据"""
        logger.info("Preparing training data...")
        
        # 加载tokenizer
        tokenizer = self.model_utils.load_tokenizer()
        
        # 创建数据处理器
        data_processor = LCCCDataProcessor(tokenizer, self.config.max_seq_length)
        
        # 准备数据集
        datasets = data_processor.prepare_datasets(
            self.config.data_path,
            self.config.train_file,
            self.config.valid_file,
            self.config.test_file
        )
        
        return tokenizer, datasets
    
    def setup_model(self, tokenizer):
        """设置模型"""
        logger.info("Setting up model...")
        
        # 加载模型
        model = self.model_utils.load_model()
        
        # 调整embedding大小（如果需要）
        if len(tokenizer) != model.get_input_embeddings().num_embeddings:
            model.resize_token_embeddings(len(tokenizer))
            logger.info(f"Resized token embeddings to {len(tokenizer)}")
        
        # 设置PEFT
        model = self.model_utils.setup_peft_model(model)
        
        return model
    
    def train(self):
        """执行训练"""
        logger.info("Starting QLoRA training...")
        
        # 检查GPU
        if torch.cuda.is_available():
            logger.info(f"CUDA available. GPU count: {torch.cuda.device_count()}")
            logger.info(f"Current device: {torch.cuda.current_device()}")
            logger.info(f"GPU memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f} GB")
        else:
            logger.warning("CUDA not available. Training will be very slow!")
        
        # 准备数据
        tokenizer, datasets = self.prepare_data()
        
        # 设置模型
        model = self.setup_model(tokenizer)
        
        # 获取内存占用信息
        self.model_utils.get_model_memory_footprint(model)
        
        # 创建训练参数
        training_args = self.model_utils.create_training_arguments()
        
        # 创建数据收集器
        data_collator = DataCollatorForLanguageModeling(
            tokenizer=tokenizer,
            mlm=False,  # 因果语言模型不使用MLM
            pad_to_multiple_of=8,  # 优化GPU性能
            return_tensors="pt",
        )
        
        # 创建训练器
        # 检查是否有验证数据集
        eval_dataset = datasets.get('eval')
        if eval_dataset is None:
            logger.warning("No evaluation dataset found, disabling evaluation")
            # 如果没有验证集，修改配置禁用评估
            self.config.evaluation_strategy = 'no'
            # 重新创建训练参数
            training_args = self.model_utils.create_training_arguments()
        
        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=datasets.get('train'),
            eval_dataset=eval_dataset,
            processing_class=tokenizer,  # 使用新的 processing_class 替代 tokenizer
            data_collator=data_collator,
        )
        
        # 保存训练前的模型状态
        logger.info("Saving initial model state...")
        trainer.save_model(os.path.join(self.config.output_dir, "initial_model"))
        
        # 开始训练
        logger.info("Starting training...")
        try:
            trainer.train()
            
            # 保存最终模型
            logger.info("Saving final model...")
            trainer.save_model(os.path.join(self.config.output_dir, "final_model"))
            
            # 保存tokenizer
            tokenizer.save_pretrained(os.path.join(self.config.output_dir, "final_model"))
            
            logger.info("Training completed successfully!")
            
        except Exception as e:
            logger.error(f"Training failed: {str(e)}")
            raise
        
        return trainer

def main():
    """主函数"""
    # 设置环境变量
    os.environ['TOKENIZERS_PARALLELISM'] = 'false'
    
    # 创建输出目录
    config = TrainingConfig()
    os.makedirs(config.output_dir, exist_ok=True)
    
    # 创建训练器并开始训练
    trainer = QLoRATrainer(config)
    trainer.train()

if __name__ == "__main__":
    main()