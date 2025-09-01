"""
隐私号平台Web服务器
基于Flask的后端API服务，连接前端界面和LangGraph Agent
"""

import asyncio
import json
from typing import Optional
from flask import Flask, request, jsonify, send_from_directory, Response
from flask_cors import CORS
import json
from privacy_agent import PrivacyNumberAgent

app = Flask(__name__)
CORS(app)

# 全局Agent实例
agent: Optional[PrivacyNumberAgent] = None

def initialize_agent():
    """初始化Agent实例"""
    global agent
    agent = PrivacyNumberAgent()
    print(" 隐私号AI Agent初始化完成")


@app.route('/')
def index():
    """提供主页面"""
    return send_from_directory('.', 'index.html')

@app.route('/api/process', methods=['POST'])
def process_request():
    """处理用户请求的API端点"""
    try:
        data = request.get_json()
        user_input = data.get('user_input', '')
        thread_id = data.get('thread_id', 'default')
        
        if not user_input:
            return jsonify({"error": "用户输入不能为空"}), 400
        
        if not agent:
            return jsonify({"error": "Agent未初始化"}), 500
            
        # 异步调用Agent处理
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        
        try:
            result = loop.run_until_complete(
                agent.process_request(user_input, thread_id)
            )
            
            # 转换结果为JSON可序列化格式
            response_data = {
                "success": True,
                "user_input": result.get("user_input"),
                "phone_number": result.get("phone_number"),
                "phone_confirmed": result.get("phone_confirmed", False),
                "privacy_type": result.get("privacy_type"),
                "type_confirmed": result.get("type_confirmed", False),
                "reasoning_process": result.get("reasoning_process", []),
                "privacy_number": result.get("privacy_number"),
                "need_human_confirmation": result.get("need_human_confirmation", False),
                "confirmation_message": result.get("confirmation_message"),
                "step": result.get("step", ""),
                "messages": result.get("messages", []),
                # 新增流式数据字段
                "need_reinput": result.get("need_reinput", False),
                "reinput_reason": result.get("reinput_reason"),
                "thinking_process": result.get("thinking_process", []),
                "final_results": result.get("final_results", []),
                "stream_data": result.get("stream_data", [])
            }
            
            return jsonify(response_data)
            
        finally:
            loop.close()
            
    except Exception as e:
        print(f"处理请求时发生错误: {str(e)}")
        return jsonify({
            "success": False,
            "error": str(e),
            "step": "error"
        }), 500

@app.route('/api/continue', methods=['POST'])
def continue_conversation():
    """继续对话的API端点（处理用户确认）"""
    try:
        data = request.get_json()
        user_response = data.get('user_response', '')
        thread_id = data.get('thread_id', 'default')
        
        if not user_response:
            return jsonify({"error": "用户回复不能为空"}), 400
        
        if not agent:
            return jsonify({"error": "Agent未初始化"}), 500
            
        # 异步调用Agent继续处理
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        
        try:
            result = loop.run_until_complete(
                agent.continue_conversation(user_response, thread_id)
            )
            
            # 转换结果为JSON可序列化格式
            response_data = {
                "success": True,
                "user_input": result.get("user_input"),
                "phone_number": result.get("phone_number"),
                "phone_confirmed": result.get("phone_confirmed", False),
                "privacy_type": result.get("privacy_type"),
                "type_confirmed": result.get("type_confirmed", False),
                "reasoning_process": result.get("reasoning_process", []),
                "privacy_number": result.get("privacy_number"),
                "need_human_confirmation": result.get("need_human_confirmation", False),
                "confirmation_message": result.get("confirmation_message"),
                "step": result.get("step", ""),
                "messages": result.get("messages", []),
                # 新增流式数据字段
                "need_reinput": result.get("need_reinput", False),
                "reinput_reason": result.get("reinput_reason"),
                "thinking_process": result.get("thinking_process", []),
                "final_results": result.get("final_results", []),
                "stream_data": result.get("stream_data", [])
            }
            
            return jsonify(response_data)
            
        finally:
            loop.close()
            
    except Exception as e:
        print(f"继续对话时发生错误: {str(e)}")
        return jsonify({
            "success": False,
            "error": str(e),
            "step": "error"
        }), 500

@app.route('/api/status', methods=['GET'])
def get_status():
    """获取系统状态"""
    return jsonify({
        "status": "running",
        "agent_initialized": agent is not None,
        "timestamp": json.dumps({"timestamp": "now"}, default=str)
    })

@app.errorhandler(404)
def not_found(error):
    """404错误处理"""
    return jsonify({"error": "API端点不存在"}), 404

@app.errorhandler(500)
def internal_error(error):
    """500错误处理"""
    return jsonify({"error": "服务器内部错误"}), 500

if __name__ == '__main__':
    print("正在启动隐私号平台服务器...")
    
    # 初始化Agent
    try:
        initialize_agent()
    except Exception as e:
        print(f"❌ Agent初始化失败: {str(e)}")
        print("请检查API密钥配置")
        exit(1)
    
    print("🌐 服务器启动成功！")
    print("📱 请访问: http://localhost:5000")
    print("🔧 API端点:")
    print("   - POST /api/process - 处理用户请求")
    print("   - POST /api/continue - 继续对话")
    print("   - GET /api/status - 获取系统状态")
    
    # 启动Flask服务器
    app.run(debug=True, host='0.0.0.0', port=5000)