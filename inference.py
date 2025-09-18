"""
推理测试脚本
加载微调后的模型进行对话测试
"""

import torch
import os
import logging
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel
from config import TrainingConfig

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ChatBot:
    def __init__(self, model_path: str, use_peft: bool = True):
        self.model_path = model_path
        self.use_peft = use_peft
        self.tokenizer = None
        self.model = None
        self.load_model()
    
    def load_model(self):
        """加载模型和tokenizer"""
        logger.info(f"Loading model from {self.model_path}")
        
        # 加载tokenizer，使用基础模型的tokenizer
        config = TrainingConfig()
        self.tokenizer = AutoTokenizer.from_pretrained(
            config.model_name,  # 使用基础模型的tokenizer
            trust_remote_code=True,
            use_fast=False
        )
        
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        # 加载基础模型
        if self.use_peft:
            # 从配置获取基础模型路径
            config = TrainingConfig()
            base_model = AutoModelForCausalLM.from_pretrained(
                config.model_name,
                torch_dtype=torch.float16,
                device_map="auto",
                trust_remote_code=True
            )
            
            # 加载PEFT权重
            self.model = PeftModel.from_pretrained(base_model, self.model_path)
        else:
            # 直接加载合并后的模型
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_path,
                torch_dtype=torch.float16,
                device_map="auto",
                trust_remote_code=True
            )
        
        self.model.eval()
        logger.info("Model loaded successfully")
    
    def format_message(self, user_input: str, history: list = None) -> str:
        """格式化消息为聊天模板"""
        # 手动构建聊天模板，与训练时使用的格式一致
        formatted_text = "<|im_start|>system\n你是一个乐于助人的AI助手。<|im_end|>\n"
        
        # 添加历史对话
        if history:
            for user_msg, assistant_msg in history:
                formatted_text += f"<|im_start|>user\n{user_msg}<|im_end|>\n"
                formatted_text += f"<|im_start|>assistant\n{assistant_msg}<|im_end|>\n"
        
        # 添加当前用户输入
        formatted_text += f"<|im_start|>user\n{user_input}<|im_end|>\n"
        formatted_text += "<|im_start|>assistant\n"
        
        return formatted_text
    
    def generate_response(self, user_input: str, history: list = None, 
                         max_new_tokens: int = 100, temperature: float = 0.3,
                         top_p: float = 0.8, do_sample: bool = True) -> str:
        """生成回复"""
        # 格式化输入
        formatted_input = self.format_message(user_input, history)
        
        # Tokenize
        inputs = self.tokenizer(
            formatted_input,
            return_tensors="pt",
            truncation=True,
            max_length=1024
        ).to(self.model.device)
        
        # 生成
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                top_p=top_p,
                do_sample=do_sample,
                eos_token_id=self.tokenizer.eos_token_id,
                pad_token_id=self.tokenizer.pad_token_id,
                repetition_penalty=1.1
            )
        
        # 解码响应（保留特殊标记）
        full_response = self.tokenizer.decode(outputs[0], skip_special_tokens=False)
        
        # 提取新生成的部分
        # 查找最后一个 assistant 标签后的内容
        if "<|im_start|>assistant\n" in full_response:
            # 找到最后一个 assistant 标签
            assistant_start = full_response.rfind("<|im_start|>assistant\n")
            if assistant_start != -1:
                response = full_response[assistant_start + len("<|im_start|>assistant\n"):]
                # 移除结束标签
                if "<|im_end|>" in response:
                    response = response[:response.find("<|im_end|>")]
                response = response.strip()
            else:
                response = ""
        else:
            # 如果没有找到模板，尝试直接截取
            response = full_response[len(formatted_input):].strip()
            if response.endswith("<|im_end|>"):
                response = response[:-10].strip()
        
        return response
    
    def chat(self):
        """交互式聊天"""
        logger.info("Starting interactive chat. Type 'quit' to exit.")
        history = []
        
        while True:
            try:
                user_input = input("\n用户: ").strip()
                
                if user_input.lower() in ['quit', 'exit', '退出']:
                    print("再见！")
                    break
                
                if not user_input:
                    continue
                
                # 生成回复
                response = self.generate_response(user_input, history)
                print(f"助手: {response}")
                
                # 更新历史
                history.append((user_input, response))
                
                # 限制历史长度
                if len(history) > 5:
                    history = history[-5:]
                    
            except KeyboardInterrupt:
                print("\n再见！")
                break
            except Exception as e:
                logger.error(f"Error during chat: {e}")
                print("抱歉，出现了错误，请重试。")

def test_model(model_path: str):
    """测试模型"""
    chatbot = ChatBot(model_path)
    
    # 测试案例
    test_cases = [
        "你好，请介绍一下自己",
        "什么是人工智能？",
        "给我讲个笑话",
        "今天天气怎么样？",
        "帮我写一首诗"
    ]
    
    print("=" * 50)
    print("模型测试")
    print("=" * 50)
    
    for i, test_input in enumerate(test_cases, 1):
        print(f"\n测试 {i}: {test_input}")
        
        # 添加调试信息
        formatted_input = chatbot.format_message(test_input)
        print(f"格式化输入 (前100字符): {formatted_input[:100]}...")
        
        response = chatbot.generate_response(test_input)
        print(f"回复: {response}")
        print(f"回复长度: {len(response)}")
        print("-" * 30)

def main():
    """主函数"""
    config = TrainingConfig()
    
    # 检查模型路径
    final_model_path = os.path.join(config.output_dir, "final_model")
    
    if not os.path.exists(final_model_path):
        logger.error(f"Model not found at {final_model_path}")
        logger.info("Please run training first: python train.py")
        return
    
    # 选择运行模式
    import sys
    if len(sys.argv) > 1 and sys.argv[1] == "--test":
        # 测试模式
        test_model(final_model_path)
    else:
        # 交互模式
        chatbot = ChatBot(final_model_path)
        chatbot.chat()

if __name__ == "__main__":
    main()