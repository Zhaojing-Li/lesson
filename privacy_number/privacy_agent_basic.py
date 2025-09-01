#!/usr/bin/env python3
"""
基础版隐私号处理Agent - 不使用中断机制
"""

import re
import json
from typing import Dict, Any, List, Optional, Literal, TypedDict
from datetime import datetime
import config


class BasicPrivacyNumberAgent:
    """基础版隐私号处理Agent - 不依赖LangGraph"""
    
    def __init__(self):
        """初始化Agent"""
        cfg = config.Config()
        print(f"获取到qwen的配置:{cfg.llm_config.api_key}")
        

    def analyze_input(self, user_input: str) -> Dict[str, Any]:
        """分析用户输入"""
        # 提取手机号
        phone_pattern = r'1[3-9]\d{9}'
        phones = re.findall(phone_pattern, user_input)
        
        # 检查隐私号类型
        privacy_type = None
        if "可回拨" in user_input:
            privacy_type = "可回拨"
        elif "不可回拨" in user_input:
            privacy_type = "不可回拨"
            
        return {
            "phones": phones,
            "privacy_type": privacy_type,
            "user_input": user_input
        }


    def generate_privacy_number(self, phone: str, privacy_type: str) -> str:
        """生成隐私号"""
        if privacy_type == "可回拨":
            return f"400{phone[3:]}"
        else:
            return f"300{phone[3:]}"


    async def process_request(self, user_input: str, thread_id: str = "default") -> Dict[str, Any]:
        """处理用户请求"""
        try:
            analysis = self.analyze_input(user_input)
            phones = analysis["phones"]
            privacy_type = analysis["privacy_type"]
            
            if not phones:
                return {
                    "step": "awaiting_confirmation",
                    "need_human_confirmation": True,
                    "confirmation_message": "请提供被调查人员的手机号码",
                    "pending_confirmation_type": "phone_input",
                    "user_input": user_input,
                    "reasoning_process": ["未检测到有效手机号"]
                }
            
            if not privacy_type:
                return {
                    "step": "awaiting_confirmation", 
                    "need_human_confirmation": True,
                    "confirmation_message": "请选择隐私号类型：可回拨或不可回拨",
                    "pending_confirmation_type": "type_confirm",
                    "phone_number": phones[0],
                    "user_input": user_input,
                    "reasoning_process": [f"检测到手机号: {phones[0]}", "等待选择隐私号类型"]
                }
            
            # 信息完整，直接生成
            phone = phones[0]
            privacy_number = self.generate_privacy_number(phone, privacy_type)
            
            return {
                "step": "completed",
                "need_human_confirmation": False,
                "phone_number": phone,
                "privacy_type": privacy_type,
                "privacy_number": privacy_number,
                "user_input": user_input,
                "reasoning_process": [
                    f"检测到手机号: {phone}",
                    f"检测到类型: {privacy_type}",
                    f"生成隐私号: {privacy_number}"
                ]
            }
            
        except Exception as e:
            return {
                "step": "error",
                "error": f"处理过程中发生错误: {str(e)}",
                "need_human_confirmation": False
            }


    async def resume_with_response(self, user_response: str, thread_id: str = "default") -> Dict[str, Any]:
        """处理用户响应（简化版）"""
        try:
            # 这里应该从存储中获取之前的状态，为简化直接处理响应
            if "可回拨" in user_response or "1" in user_response:
                privacy_type = "可回拨"
            elif "不可回拨" in user_response or "2" in user_response:
                privacy_type = "不可回拨"
            else:
                # 尝试从响应中提取手机号
                phone_pattern = r'1[3-9]\d{9}'
                phones = re.findall(phone_pattern, user_response)
                if phones:
                    return {
                        "step": "awaiting_confirmation",
                        "need_human_confirmation": True,
                        "confirmation_message": "请选择隐私号类型：可回拨或不可回拨",
                        "pending_confirmation_type": "type_confirm",
                        "phone_number": phones[0],
                        "reasoning_process": [f"用户提供手机号: {phones[0]}"]
                    }
                else:
                    return {
                        "step": "error",
                        "error": "无法理解您的响应，请重新输入",
                        "need_human_confirmation": False
                    }
            
            # 假设有手机号（实际应该从状态中获取）
            # 这里为演示使用默认值
            phone = "13812345678"  # 实际应该从之前状态获取
            privacy_number = self.generate_privacy_number(phone, privacy_type)
            
            return {
                "step": "completed",
                "need_human_confirmation": False,
                "phone_number": phone,
                "privacy_type": privacy_type,
                "privacy_number": privacy_number,
                "reasoning_process": [
                    f"用户选择类型: {privacy_type}",
                    f"生成隐私号: {privacy_number}"
                ]
            }
            
        except Exception as e:
            return {
                "step": "error",
                "error": f"处理响应时发生错误: {str(e)}",
                "need_human_confirmation": False
            }
