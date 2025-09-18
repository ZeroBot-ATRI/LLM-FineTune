#!/usr/bin/env python3
"""
测试训练功能的脚本
"""

import asyncio
import json
import requests
import time
from datetime import datetime

class TrainingTester:
    def __init__(self, base_url="http://127.0.0.1:8000"):
        self.base_url = base_url
        
    def test_health(self):
        """测试健康检查"""
        print("🔍 检查服务健康状态...")
        try:
            response = requests.get(f"{self.base_url}/api/health")
            if response.status_code == 200:
                data = response.json()
                print(f"✅ 服务正常运行，模型加载状态: {data.get('model_loaded', False)}")
                return True
            else:
                print(f"❌ 健康检查失败: {response.status_code}")
                return False
        except Exception as e:
            print(f"❌ 连接失败: {str(e)}")
            return False
    
    def test_system_info(self):
        """测试系统信息"""
        print("\n🖥️  获取系统信息...")
        try:
            response = requests.get(f"{self.base_url}/api/system/info")
            if response.status_code == 200:
                data = response.json()
                print(f"✅ Python版本: {data.get('python_version')}")
                print(f"✅ PyTorch版本: {data.get('torch_version')}")
                print(f"✅ CUDA可用: {data.get('cuda_available')}")
                if data.get('cuda_available'):
                    print(f"✅ GPU设备数: {data.get('device_count')}")
                    print(f"✅ GPU内存: {data.get('gpu_memory_gb', 'unknown')} GB")
                print(f"✅ 系统内存: {data.get('memory', {}).get('total_ram_gb', 'unknown')} GB")
                return True
            else:
                print(f"❌ 获取系统信息失败: {response.status_code}")
                return False
        except Exception as e:
            print(f"❌ 获取系统信息失败: {str(e)}")
            return False
    
    def test_training_config(self):
        """测试训练配置"""
        print("\n⚙️  获取训练配置...")
        try:
            response = requests.get(f"{self.base_url}/api/training/config")
            if response.status_code == 200:
                data = response.json()
                print(f"✅ 模型名称: {data.get('model_name')}")
                print(f"✅ 训练轮数: {data.get('epochs')}")
                print(f"✅ 批次大小: {data.get('batch_size')}")
                print(f"✅ 学习率: {data.get('learning_rate')}")
                print(f"✅ 最大序列长度: {data.get('max_seq_length')}")
                print(f"✅ LoRA rank: {data.get('lora_r')}")
                return True
            else:
                print(f"❌ 获取训练配置失败: {response.status_code}")
                return False
        except Exception as e:
            print(f"❌ 获取训练配置失败: {str(e)}")
            return False
    
    def test_training_status(self):
        """测试训练状态"""
        print("\n📊 获取训练状态...")
        try:
            response = requests.get(f"{self.base_url}/api/training/status")
            if response.status_code == 200:
                data = response.json()
                print(f"✅ 训练状态: {data.get('status')}")
                print(f"✅ 是否训练中: {data.get('is_training')}")
                print(f"✅ 当前步数: {data.get('current_step')}")
                print(f"✅ 总步数: {data.get('total_steps')}")
                print(f"✅ 当前损失: {data.get('loss')}")
                return True
            else:
                print(f"❌ 获取训练状态失败: {response.status_code}")
                return False
        except Exception as e:
            print(f"❌ 获取训练状态失败: {str(e)}")
            return False
    
    def test_start_training(self, epochs=1, batch_size=1, learning_rate=2e-5):
        """测试开始训练"""
        print(f"\n🚀 开始训练测试（epochs={epochs}, batch_size={batch_size}, lr={learning_rate}）...")
        try:
            payload = {
                "epochs": epochs,
                "batch_size": batch_size,
                "learning_rate": learning_rate
            }
            response = requests.post(f"{self.base_url}/api/training/start", json=payload)
            if response.status_code == 200:
                data = response.json()
                print(f"✅ 训练启动成功: {data.get('message')}")
                print(f"✅ 使用配置: {data.get('config', {})}")
                return True
            else:
                print(f"❌ 启动训练失败: {response.status_code} - {response.text}")
                return False
        except Exception as e:
            print(f"❌ 启动训练失败: {str(e)}")
            return False
    
    def monitor_training(self, timeout=300):
        """监控训练进度"""
        print(f"\n👀 监控训练进度（超时 {timeout} 秒）...")
        start_time = time.time()
        
        while time.time() - start_time < timeout:
            try:
                response = requests.get(f"{self.base_url}/api/training/status")
                if response.status_code == 200:
                    data = response.json()
                    status = data.get('status', 'unknown')
                    is_training = data.get('is_training', False)
                    current_step = data.get('current_step', 0)
                    total_steps = data.get('total_steps', 0)
                    loss = data.get('loss', 0.0)
                    
                    print(f"📈 状态: {status} | 步数: {current_step}/{total_steps} | 损失: {loss:.4f}")
                    
                    if not is_training and status in ['completed', 'error', 'stopped']:
                        print(f"✅ 训练结束，最终状态: {status}")
                        return status == 'completed'
                    
                    time.sleep(5)  # 每5秒检查一次
                else:
                    print(f"❌ 获取状态失败: {response.status_code}")
                    time.sleep(5)
            except Exception as e:
                print(f"❌ 监控出错: {str(e)}")
                time.sleep(5)
        
        print(f"⏰ 监控超时（{timeout}秒）")
        return False
    
    def run_comprehensive_test(self):
        """运行综合测试"""
        print("🧪 开始LLM微调训练功能综合测试")
        print("=" * 60)
        
        # 测试序列
        tests = [
            ("健康检查", self.test_health),
            ("系统信息", self.test_system_info),
            ("训练配置", self.test_training_config),
            ("训练状态", self.test_training_status),
        ]
        
        # 运行基础测试
        for test_name, test_func in tests:
            if not test_func():
                print(f"\n❌ {test_name}测试失败，终止测试")
                return False
        
        # 询问是否进行实际训练测试
        print("\n" + "=" * 60)
        print("⚠️  即将进行实际训练测试，这将消耗计算资源")
        
        # 在脚本环境中，我们设置一个较小的训练配置用于测试
        if self.test_start_training(epochs=1, batch_size=1, learning_rate=2e-5):
            print("\n📊 开始监控训练过程...")
            success = self.monitor_training(timeout=600)  # 10分钟超时
            
            if success:
                print("\n🎉 训练测试完成！")
            else:
                print("\n⚠️  训练测试未完成或失败")
        
        print("\n" + "=" * 60)
        print("🏁 测试结束")

def main():
    """主函数"""
    tester = TrainingTester()
    tester.run_comprehensive_test()

if __name__ == "__main__":
    main()