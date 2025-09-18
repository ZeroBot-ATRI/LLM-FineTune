// 全局变量
let websocket = null;
let chatHistory = [];
let lossChart = null;
let lossData = [];

// DOM元素
const elements = {
    // 导航
    navItems: document.querySelectorAll('.nav-item'),
    tabContents: document.querySelectorAll('.tab-content'),
    
    // 状态显示
    modelStatus: document.getElementById('model-status'),
    trainingStatus: document.getElementById('training-status'),
    
    // 对话界面
    chatMessages: document.getElementById('chat-messages'),
    chatInput: document.getElementById('chat-input'),
    sendBtn: document.getElementById('send-btn'),
    clearChatBtn: document.getElementById('clear-chat'),
    exportChatBtn: document.getElementById('export-chat'),
    
    // 生成设置
    maxTokensSlider: document.getElementById('max-tokens'),
    maxTokensValue: document.getElementById('max-tokens-value'),
    temperatureSlider: document.getElementById('temperature'),
    temperatureValue: document.getElementById('temperature-value'),
    topPSlider: document.getElementById('top-p'),
    topPValue: document.getElementById('top-p-value'),
    
    // 训练界面
    epochsInput: document.getElementById('epochs'),
    batchSizeInput: document.getElementById('batch-size'),
    learningRateInput: document.getElementById('learning-rate'),
    startTrainingBtn: document.getElementById('start-training'),
    stopTrainingBtn: document.getElementById('stop-training'),
    
    // 训练进度
    currentStep: document.getElementById('current-step'),
    totalSteps: document.getElementById('total-steps'),
    currentLoss: document.getElementById('current-loss'),
    progressBar: document.getElementById('progress-bar'),
    lossCanvas: document.getElementById('loss-canvas'),
    
    // 模型管理
    refreshModelsBtn: document.getElementById('refresh-models'),
    modelsList: document.getElementById('models-list'),
    
    // 日志
    refreshLogsBtn: document.getElementById('refresh-logs'),
    clearLogsBtn: document.getElementById('clear-logs'),
    logsContent: document.getElementById('logs-content'),
    
    // 通用
    loadingOverlay: document.getElementById('loading-overlay'),
    notification: document.getElementById('notification'),
    notificationText: document.getElementById('notification-text'),
    closeNotification: document.getElementById('close-notification')
};

// 初始化
document.addEventListener('DOMContentLoaded', function() {
    initializeWebSocket();
    initializeEventListeners();
    initializeChart();
    checkModelStatus();
    loadModels();
    loadLogs();
});

// WebSocket连接
function initializeWebSocket() {
    const protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
    const wsUrl = `${protocol}//${window.location.host}/ws`;
    
    websocket = new WebSocket(wsUrl);
    
    websocket.onopen = function() {
        console.log('WebSocket连接已建立');
        showNotification('WebSocket连接成功', 'success');
    };
    
    websocket.onmessage = function(event) {
        const data = JSON.parse(event.data);
        handleWebSocketMessage(data);
    };
    
    websocket.onclose = function() {
        console.log('WebSocket连接已关闭');
        showNotification('WebSocket连接断开，尝试重连...', 'warning');
        setTimeout(initializeWebSocket, 5000);
    };
    
    websocket.onerror = function(error) {
        console.error('WebSocket错误:', error);
        showNotification('WebSocket连接错误', 'error');
    };
}

// 处理WebSocket消息
function handleWebSocketMessage(data) {
    switch(data.type) {
        case 'chat_message':
            // 对话消息已通过HTTP API处理
            break;
            
        case 'training_started':
            elements.startTrainingBtn.disabled = true;
            elements.stopTrainingBtn.disabled = false;
            elements.trainingStatus.textContent = '训练中';
            showNotification('训练已开始', 'success');
            break;
            
        case 'training_progress':
            updateTrainingProgress(data);
            break;
            
        case 'training_completed':
            elements.startTrainingBtn.disabled = false;
            elements.stopTrainingBtn.disabled = true;
            elements.trainingStatus.textContent = '已完成';
            showNotification('训练已完成', 'success');
            break;
            
        case 'training_stopped':
            elements.startTrainingBtn.disabled = false;
            elements.stopTrainingBtn.disabled = true;
            elements.trainingStatus.textContent = '已停止';
            showNotification('训练已停止', 'warning');
            break;
            
        case 'training_error':
            elements.startTrainingBtn.disabled = false;
            elements.stopTrainingBtn.disabled = true;
            elements.trainingStatus.textContent = '错误';
            showNotification(`训练错误: ${data.error}`, 'error');
            break;
            
        case 'model_loaded':
            elements.modelStatus.textContent = '已加载';
            showNotification(`模型 ${data.model_name} 加载成功`, 'success');
            break;
            
        case 'pong':
            // 心跳响应
            break;
    }
}

// 事件监听器
function initializeEventListeners() {
    // 导航切换
    elements.navItems.forEach(item => {
        item.addEventListener('click', function() {
            const targetTab = this.dataset.tab;
            switchTab(targetTab);
        });
    });
    
    // 滑块值显示
    elements.maxTokensSlider.addEventListener('input', function() {
        elements.maxTokensValue.textContent = this.value;
    });
    
    elements.temperatureSlider.addEventListener('input', function() {
        elements.temperatureValue.textContent = this.value;
    });
    
    elements.topPSlider.addEventListener('input', function() {
        elements.topPValue.textContent = this.value;
    });
    
    // 对话功能
    elements.sendBtn.addEventListener('click', sendMessage);
    elements.chatInput.addEventListener('keypress', function(e) {
        if (e.key === 'Enter' && !e.shiftKey) {
            e.preventDefault();
            sendMessage();
        }
    });
    
    elements.clearChatBtn.addEventListener('click', clearChat);
    elements.exportChatBtn.addEventListener('click', exportChat);
    
    // 训练功能
    elements.startTrainingBtn.addEventListener('click', startTraining);
    elements.stopTrainingBtn.addEventListener('click', stopTraining);
    
    // 模型管理
    elements.refreshModelsBtn.addEventListener('click', loadModels);
    
    // 日志功能
    elements.refreshLogsBtn.addEventListener('click', loadLogs);
    elements.clearLogsBtn.addEventListener('click', clearLogs);
    
    // 通知关闭
    elements.closeNotification.addEventListener('click', hideNotification);
}

// 切换标签页
function switchTab(tabName) {
    // 更新导航样式
    elements.navItems.forEach(item => {
        if (item.dataset.tab === tabName) {
            item.classList.add('active');
        } else {
            item.classList.remove('active');
        }
    });
    
    // 切换内容显示
    elements.tabContents.forEach(content => {
        if (content.id === `${tabName}-tab`) {
            content.classList.add('active');
        } else {
            content.classList.remove('active');
        }
    });
}

// 检查模型状态
async function checkModelStatus() {
    try {
        const response = await fetch('/api/health');
        const data = await response.json();
        
        if (data.model_loaded) {
            elements.modelStatus.textContent = '已加载';
            elements.modelStatus.style.color = '#51cf66';
        } else {
            elements.modelStatus.textContent = '未加载';
            elements.modelStatus.style.color = '#ff6b6b';
        }
    } catch (error) {
        console.error('检查模型状态失败:', error);
        elements.modelStatus.textContent = '错误';
        elements.modelStatus.style.color = '#ff6b6b';
    }
}

// 发送消息
async function sendMessage() {
    const message = elements.chatInput.value.trim();
    if (!message) return;
    
    // 添加用户消息到界面
    addMessage(message, 'user');
    elements.chatInput.value = '';
    
    // 显示加载状态
    showLoading();
    
    try {
        const response = await fetch('/api/chat', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify({
                message: message,
                history: chatHistory,
                max_new_tokens: parseInt(elements.maxTokensSlider.value),
                temperature: parseFloat(elements.temperatureSlider.value),
                top_p: parseFloat(elements.topPSlider.value)
            })
        });
        
        if (!response.ok) {
            throw new Error(`HTTP ${response.status}: ${response.statusText}`);
        }
        
        const data = await response.json();
        
        // 添加助手回复到界面
        addMessage(data.response, 'bot');
        
        // 更新历史记录
        chatHistory = data.history;
        
    } catch (error) {
        console.error('发送消息失败:', error);
        addMessage('抱歉，发生了错误。请稍后重试。', 'bot');
        showNotification('发送消息失败', 'error');
    } finally {
        hideLoading();
    }
}

// 添加消息到界面
function addMessage(text, type) {
    const messageDiv = document.createElement('div');
    messageDiv.className = `message ${type}-message`;
    
    const time = new Date().toLocaleTimeString('zh-CN', {
        hour: '2-digit',
        minute: '2-digit'
    });
    
    messageDiv.innerHTML = `
        <div class="message-avatar">
            <i class="fas fa-${type === 'user' ? 'user' : 'robot'}"></i>
        </div>
        <div class="message-content">
            <div class="message-text">${text}</div>
            <div class="message-time">${time}</div>
        </div>
    `;
    
    elements.chatMessages.appendChild(messageDiv);
    elements.chatMessages.scrollTop = elements.chatMessages.scrollHeight;
}

// 清空对话
function clearChat() {
    if (confirm('确定要清空所有对话吗？')) {
        elements.chatMessages.innerHTML = `
            <div class="message bot-message">
                <div class="message-avatar">
                    <i class="fas fa-robot"></i>
                </div>
                <div class="message-content">
                    <div class="message-text">你好！我是基于Qwen3-0.6B微调的中文对话助手。有什么可以帮助你的吗？</div>
                    <div class="message-time">刚刚</div>
                </div>
            </div>
        `;
        chatHistory = [];
        showNotification('对话已清空', 'success');
    }
}

// 导出对话
function exportChat() {
    if (chatHistory.length === 0) {
        showNotification('没有对话记录可导出', 'warning');
        return;
    }
    
    const chatText = chatHistory.map(([user, bot]) => 
        `用户: ${user}\n助手: ${bot}\n\n`
    ).join('');
    
    const blob = new Blob([chatText], { type: 'text/plain;charset=utf-8' });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = `chat_export_${new Date().toISOString().slice(0, 10)}.txt`;
    a.click();
    URL.revokeObjectURL(url);
    
    showNotification('对话已导出', 'success');
}

// 开始训练
async function startTraining() {
    if (!confirm('确定要开始训练吗？这可能需要较长时间。')) {
        return;
    }
    
    const trainingConfig = {
        epochs: parseInt(elements.epochsInput.value),
        batch_size: parseInt(elements.batchSizeInput.value),
        learning_rate: parseFloat(elements.learningRateInput.value)
    };
    
    try {
        showLoading();
        
        const response = await fetch('/api/training/start', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify(trainingConfig)
        });
        
        if (!response.ok) {
            throw new Error(`HTTP ${response.status}: ${response.statusText}`);
        }
        
        const data = await response.json();
        showNotification(data.message, 'success');
        
    } catch (error) {
        console.error('开始训练失败:', error);
        showNotification('开始训练失败', 'error');
    } finally {
        hideLoading();
    }
}

// 停止训练
async function stopTraining() {
    if (!confirm('确定要停止训练吗？')) {
        return;
    }
    
    try {
        const response = await fetch('/api/training/stop', {
            method: 'POST'
        });
        
        if (!response.ok) {
            throw new Error(`HTTP ${response.status}: ${response.statusText}`);
        }
        
        const data = await response.json();
        showNotification(data.message, 'success');
        
    } catch (error) {
        console.error('停止训练失败:', error);
        showNotification('停止训练失败', 'error');
    }
}

// 更新训练进度
function updateTrainingProgress(data) {
    elements.currentStep.textContent = data.step;
    elements.totalSteps.textContent = data.total_steps;
    elements.currentLoss.textContent = data.loss.toFixed(4);
    
    const progress = (data.step / data.total_steps) * 100;
    elements.progressBar.style.width = `${progress}%`;
    
    // 更新损失图表
    lossData.push({
        step: data.step,
        loss: data.loss
    });
    
    updateLossChart();
}

// 初始化图表
function initializeChart() {
    const ctx = elements.lossCanvas.getContext('2d');
    lossChart = new Chart(ctx, {
        type: 'line',
        data: {
            labels: [],
            datasets: [{
                label: '训练损失',
                data: [],
                borderColor: '#667eea',
                backgroundColor: 'rgba(102, 126, 234, 0.1)',
                tension: 0.1
            }]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            scales: {
                y: {
                    beginAtZero: true,
                    title: {
                        display: true,
                        text: '损失值'
                    }
                },
                x: {
                    title: {
                        display: true,
                        text: '训练步数'
                    }
                }
            },
            plugins: {
                legend: {
                    display: true,
                    position: 'top'
                }
            }
        }
    });
}

// 更新损失图表
function updateLossChart() {
    const labels = lossData.map(d => d.step);
    const data = lossData.map(d => d.loss);
    
    lossChart.data.labels = labels;
    lossChart.data.datasets[0].data = data;
    lossChart.update();
}

// 加载模型列表
async function loadModels() {
    try {
        showLoading();
        
        const response = await fetch('/api/models');
        if (!response.ok) {
            throw new Error(`HTTP ${response.status}: ${response.statusText}`);
        }
        
        const data = await response.json();
        displayModels(data.models);
        
    } catch (error) {
        console.error('加载模型列表失败:', error);
        showNotification('加载模型列表失败', 'error');
    } finally {
        hideLoading();
    }
}

// 显示模型列表
function displayModels(models) {
    if (models.length === 0) {
        elements.modelsList.innerHTML = '<p>暂无可用模型</p>';
        return;
    }
    
    elements.modelsList.innerHTML = models.map(model => `
        <div class="model-card">
            <h4>${model.model_name}</h4>
            <div class="model-info">
                <div class="model-info-item">
                    <span>状态:</span>
                    <span>${model.status}</span>
                </div>
                <div class="model-info-item">
                    <span>大小:</span>
                    <span>${model.size_mb} MB</span>
                </div>
                <div class="model-info-item">
                    <span>修改时间:</span>
                    <span>${new Date(model.last_modified).toLocaleString('zh-CN')}</span>
                </div>
            </div>
            <div class="model-actions">
                <button class="btn btn-primary" onclick="loadModel('${model.model_name}')">
                    <i class="fas fa-play"></i> 加载
                </button>
            </div>
        </div>
    `).join('');
}

// 加载指定模型
async function loadModel(modelName) {
    try {
        showLoading();
        
        const response = await fetch(`/api/load_model/${modelName}`, {
            method: 'POST'
        });
        
        if (!response.ok) {
            throw new Error(`HTTP ${response.status}: ${response.statusText}`);
        }
        
        const data = await response.json();
        showNotification(data.message, 'success');
        
    } catch (error) {
        console.error('加载模型失败:', error);
        showNotification('加载模型失败', 'error');
    } finally {
        hideLoading();
    }
}

// 加载日志
async function loadLogs() {
    try {
        const response = await fetch('/api/training/logs');
        if (!response.ok) {
            throw new Error(`HTTP ${response.status}: ${response.statusText}`);
        }
        
        const data = await response.json();
        elements.logsContent.textContent = data.logs.join('');
        elements.logsContent.scrollTop = elements.logsContent.scrollHeight;
        
    } catch (error) {
        console.error('加载日志失败:', error);
        showNotification('加载日志失败', 'error');
    }
}

// 清空日志显示
function clearLogs() {
    elements.logsContent.textContent = '';
    showNotification('日志显示已清空', 'success');
}

// 显示加载状态
function showLoading() {
    elements.loadingOverlay.style.display = 'flex';
}

// 隐藏加载状态
function hideLoading() {
    elements.loadingOverlay.style.display = 'none';
}

// 显示通知
function showNotification(message, type = 'info') {
    elements.notificationText.textContent = message;
    elements.notification.className = `notification show ${type}`;
    
    setTimeout(() => {
        hideNotification();
    }, 5000);
}

// 隐藏通知
function hideNotification() {
    elements.notification.classList.remove('show');
}

// 心跳检测
setInterval(() => {
    if (websocket && websocket.readyState === WebSocket.OPEN) {
        websocket.send(JSON.stringify({ type: 'ping' }));
    }
}, 30000);