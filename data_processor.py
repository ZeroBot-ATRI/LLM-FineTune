"""
数据处理模块
处理LCCC对话数据集，支持多轮对话训练
"""

import os
import json
import logging
from typing import Dict, List, Any, Optional
from datasets import Dataset
import torch

logger = logging.getLogger(__name__)

class LCCCDataProcessor:
    """LCCC数据集处理器"""
    
    def __init__(self, tokenizer, max_seq_length: int = 512):
        """
        初始化数据处理器
        
        Args:
            tokenizer: 分词器
            max_seq_length: 最大序列长度
        """
        self.tokenizer = tokenizer
        self.max_seq_length = max_seq_length
        
        # 设置特殊tokens
        self.pad_token_id = tokenizer.pad_token_id
        if self.pad_token_id is None:
            self.pad_token_id = tokenizer.eos_token_id
            
        # 聊天模板
        self.system_message = "你是一个乐于助人的AI助手。"
        self.user_start = "<|im_start|>user\n"
        self.user_end = "<|im_end|>\n"
        self.assistant_start = "<|im_start|>assistant\n"
        self.assistant_end = "<|im_end|>"
        
    def load_json_data(self, file_path: str) -> List[Dict]:
        """加载JSON数据文件"""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            logger.info(f"Loaded {len(data)} conversations from {file_path}")
            return data
        except Exception as e:
            logger.error(f"Error loading {file_path}: {str(e)}")
            return []
    
    def format_conversation(self, conversation: List[str]) -> str:
        """
        将对话列表格式化为训练文本
        
        Args:
            conversation: 对话轮次列表
            
        Returns:
            格式化后的训练文本
        """
        if len(conversation) < 2:
            return ""
            
        formatted_text = f"<|im_start|>system\n{self.system_message}<|im_end|>\n"
        
        for i in range(0, len(conversation), 2):
            # 用户消息
            if i < len(conversation):
                user_msg = conversation[i].strip()
                formatted_text += f"{self.user_start}{user_msg}{self.user_end}"
            
            # 助手回复
            if i + 1 < len(conversation):
                assistant_msg = conversation[i + 1].strip()
                formatted_text += f"{self.assistant_start}{assistant_msg}{self.assistant_end}"
        
        return formatted_text
    
    def tokenize_function(self, examples: Dict[str, List]) -> Dict[str, List]:
        """
        对文本进行分词
        
        Args:
            examples: 包含文本的字典
            
        Returns:
            分词后的结果
        """
        # 分词，确保使用unk_token处理未知词汇
        model_inputs = self.tokenizer(
            examples["text"],
            truncation=True,
            padding=False,  # 不在这里padding，后面用DataCollator
            max_length=self.max_seq_length,
            return_tensors=None,
            # 确保使用unk_token处理未知词汇
            add_special_tokens=True
        )
        
        # 检查是否有unk_token_id出现
        if hasattr(self.tokenizer, 'unk_token_id') and self.tokenizer.unk_token_id is not None:
            for i, input_ids in enumerate(model_inputs["input_ids"]):
                unk_count = input_ids.count(self.tokenizer.unk_token_id)
                if unk_count > 0:
                    logger.debug(f"Sample {i}: Found {unk_count} unknown tokens")
        
        # 对于因果语言模型，labels就是input_ids
        # DataCollatorForLanguageModeling会自动处理labels，所以不需要手动设置
        # model_inputs["labels"] = [ids[:] for ids in model_inputs["input_ids"]]
        
        return model_inputs
    
    def process_dataset_file(self, file_path: str) -> Optional[Dataset]:
        """
        处理单个数据集文件
        
        Args:
            file_path: 数据文件路径
            
        Returns:
            处理后的Dataset对象
        """
        if not os.path.exists(file_path):
            logger.warning(f"File not found: {file_path}")
            return None
            
        # 加载原始数据
        raw_data = self.load_json_data(file_path)
        if not raw_data:
            return None
        
        # 格式化对话
        formatted_texts = []
        for item in raw_data:
            # 支持两种数据格式：
            # 1. 字典格式：{"conversation": [...]}
            # 2. 直接列表格式：[...]
            conversation = None
            if isinstance(item, dict) and "conversation" in item:
                conversation = item["conversation"]
            elif isinstance(item, list):
                conversation = item
            
            if conversation and isinstance(conversation, list):
                formatted_text = self.format_conversation(conversation)
                if formatted_text:
                    formatted_texts.append(formatted_text)
        
        if not formatted_texts:
            logger.warning(f"No valid conversations found in {file_path}")
            return None
        
        logger.info(f"Processed {len(formatted_texts)} conversations from {file_path}")
        
        # 创建Dataset
        dataset = Dataset.from_dict({"text": formatted_texts})
        
        # 分词
        tokenized_dataset = dataset.map(
            self.tokenize_function,
            batched=True,
            remove_columns=dataset.column_names,
            desc="Tokenizing"
        )
        
        return tokenized_dataset
    
    def prepare_datasets(self, 
                        data_path: str,
                        train_file: str,
                        valid_file: str,
                        test_file: Optional[str] = None) -> Dict[str, Dataset]:
        """
        准备训练、验证和测试数据集
        
        Args:
            data_path: 数据目录路径
            train_file: 训练文件名
            valid_file: 验证文件名
            test_file: 测试文件名（可选）
            
        Returns:
            包含数据集的字典
        """
        datasets = {}
        
        # 处理训练集
        train_path = os.path.join(data_path, train_file)
        train_dataset = self.process_dataset_file(train_path)
        if train_dataset:
            datasets["train"] = train_dataset
            logger.info(f"Training dataset: {len(train_dataset)} samples")
        
        # 处理验证集
        valid_path = os.path.join(data_path, valid_file)
        valid_dataset = self.process_dataset_file(valid_path)
        if valid_dataset:
            datasets["eval"] = valid_dataset
            logger.info(f"Validation dataset: {len(valid_dataset)} samples")
        
        # 处理测试集（如果提供）
        if test_file:
            test_path = os.path.join(data_path, test_file)
            test_dataset = self.process_dataset_file(test_path)
            if test_dataset:
                datasets["test"] = test_dataset
                logger.info(f"Test dataset: {len(test_dataset)} samples")
        
        return datasets
    
    def get_sample_text(self, conversation: List[str]) -> str:
        """
        获取样本文本（用于调试）
        
        Args:
            conversation: 对话列表
            
        Returns:
            格式化后的文本
        """
        return self.format_conversation(conversation)