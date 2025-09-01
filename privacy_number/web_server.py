#!/usr/bin/env python3
"""
隐私号平台Web服务器
提供REST API接口和静态文件服务
"""

import asyncio
import json
from datetime import datetime
from typing import Dict, Any
from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
import logging

from privacy_agent import PrivacyNumberAgent

# 配置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# 创建Flask应用
app = Flask(__name__)
CORS(app)  # 允许跨域请求

# 全局变量存储agent实例
privacy_agent = None

def get_agent():
    """获取或创建PrivacyNumberAgent实例"""
    global privacy_agent
    if privacy_agent is None:
        privacy_agent = PrivacyNumberAgent()
        logger.info("PrivacyNumberAgent实例已创建")
    return privacy_agent


@app.route('/')
def serve_index():
    """提供主页面"""
    try:
        return send_from_directory('.', 'index_optimized.html')
    except Exception as e:
        logger.error(f"提供主页面时发生错误: {e}")
        return f"页面加载错误: {str(e)}", 500


@app.route('/api/process', methods=['POST'])
def process_request():
    """处理用户初始请求"""
    try:
        # 解析请求数据
        data = request.get_json()
        if not data:
            return jsonify({"error": "请求数据为空"}), 400
        
        user_input = data.get('user_input', '').strip()
        thread_id = data.get('thread_id', 'default')
        
        if not user_input:
            return jsonify({"error": "用户输入不能为空"}), 400
        
        logger.info(f"收到处理请求 - Thread ID: {thread_id}, Input: {user_input}")
        
        # 获取agent并处理请求
        agent = get_agent()
        
        # 使用asyncio运行异步方法
        try:
            # 直接使用asyncio.run，这是推荐的方式
            result = asyncio.run(agent.process_request(user_input, thread_id))
        except Exception as e:
            logger.error(f"异步执行错误: {e}")
            # 如果失败，尝试使用nest_asyncio
            try:
                import nest_asyncio
                nest_asyncio.apply()
                result = asyncio.run(agent.process_request(user_input, thread_id))
            except Exception as e2:
                logger.error(f"nest_asyncio执行也失败: {e2}")
                raise e
        
        logger.info(f"处理结果 - Step: {result.get('step')}, 需要确认: {result.get('need_human_confirmation')}")
        
        return jsonify(result)
        
    except Exception as e:
        logger.error(f"处理请求时发生错误: {e}")
        return jsonify({
            "error": f"处理请求时发生错误: {str(e)}",
            "step": "error",
            "need_human_confirmation": False
        }), 500


@app.route('/api/continue', methods=['POST'])
def continue_conversation():
    """处理用户确认响应，恢复对话"""
    try:
        # 解析请求数据
        data = request.get_json()
        if not data:
            return jsonify({"error": "请求数据为空"}), 400
        
        user_response = data.get('user_response', '').strip()
        thread_id = data.get('thread_id', 'default')
        
        if not user_response:
            return jsonify({"error": "用户响应不能为空"}), 400
        
        logger.info(f"收到继续请求 - Thread ID: {thread_id}, Response: {user_response}")
        
        # 获取agent并恢复对话
        agent = get_agent()
        
        # 使用asyncio运行异步方法
        try:
            # 直接使用asyncio.run，这是推荐的方式
            result = asyncio.run(agent.resume_with_response(user_response, thread_id))
        except Exception as e:
            logger.error(f"异步执行错误: {e}")
            # 如果失败，尝试使用nest_asyncio
            try:
                import nest_asyncio
                nest_asyncio.apply()
                result = asyncio.run(agent.resume_with_response(user_response, thread_id))
            except Exception as e2:
                logger.error(f"nest_asyncio执行也失败: {e2}")
                raise e
        
        logger.info(f"恢复结果 - Step: {result.get('step')}, 需要确认: {result.get('need_human_confirmation')}")
        
        return jsonify(result)
        
    except Exception as e:
        logger.error(f"恢复对话时发生错误: {e}")
        return jsonify({
            "error": f"恢复对话时发生错误: {str(e)}",
            "step": "error",
            "need_human_confirmation": False
        }), 500


@app.route('/api/health', methods=['GET'])
def health_check():
    """健康检查接口"""
    return jsonify({
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "service": "隐私号平台Web服务"
    })


@app.errorhandler(404)
def not_found(error):
    """404错误处理"""
    return jsonify({"error": "接口不存在"}), 404


@app.errorhandler(500)
def internal_error(error):
    """500错误处理"""
    logger.error(f"服务器内部错误: {error}")
    return jsonify({"error": "服务器内部错误"}), 500


if __name__ == '__main__':
    logger.info("正在启动隐私号平台Web服务器...")
    
    # 预创建agent实例
    try:
        get_agent()
        logger.info("PrivacyNumberAgent预初始化成功")
    except Exception as e:
        logger.error(f"PrivacyNumberAgent初始化失败: {e}")
        logger.warning("服务器将启动，但功能可能受限")
    
    # 启动Flask服务器
    app.run(
        host='0.0.0.0',
        port=5000,
        debug=True,
        threaded=True
    )
