"""
训练配置文件
包含模型、数据、训练超参数等配置
"""

class TrainingConfig:
    # 模型配置
    model_name = "Qwen/Qwen3-0.6B"  # 使用Qwen3-0.6B作为基础模型
    tokenizer_name = "Qwen/Qwen3-0.6B"
    
    # 数据配置
    data_path = "./datasets"
    train_file = "LCCC-base_train.json"
    valid_file = "LCCC-base_valid.json"
    test_file = "LCCC-base_test.json"
    max_seq_length = 512
    
    # QLoRA配置
    use_qlora = True
    lora_r = 16
    lora_alpha = 32
    lora_dropout = 0.1
    lora_bias = "none"
    target_modules = [
        "q_proj",
        "k_proj", 
        "v_proj",
        "o_proj",
        "gate_proj",
        "up_proj",
        "down_proj"
    ]
    
    # 量化配置
    load_in_4bit = True
    bnb_4bit_compute_dtype = "float16"
    bnb_4bit_quant_type = "nf4"
    bnb_4bit_use_double_quant = True
    
    # 训练超参数
    output_dir = "./outputs"
    num_train_epochs = 3
    per_device_train_batch_size = 2
    per_device_eval_batch_size = 2
    gradient_accumulation_steps = 4
    optim = "paged_adamw_32bit"
    save_steps = 500
    logging_steps = 10
    learning_rate = 2e-4
    weight_decay = 0.001
    fp16 = False
    bf16 = True
    max_grad_norm = 0.3
    max_steps = -1
    warmup_ratio = 0.03
    group_by_length = True
    lr_scheduler_type = "constant"
    
    # 评估配置
    eval_strategy = "steps"
    eval_steps = 500
    save_strategy = "steps"
    load_best_model_at_end = True
    metric_for_best_model = "eval_loss"
    greater_is_better = False
    
    # 其他配置
    report_to = "none"  # 可设置为"wandb"启用wandb日志
    remove_unused_columns = False
    dataloader_pin_memory = False
    
    # 聊天模板配置
    chat_template = {
        "system": "你是一个乐于助人的AI助手。",
        "user_start": "<|im_start|>user\n",
        "user_end": "<|im_end|>\n",
        "assistant_start": "<|im_start|>assistant\n", 
        "assistant_end": "<|im_end|>"
    }