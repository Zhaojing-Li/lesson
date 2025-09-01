#!/usr/bin/env python3
"""
隐私号处理工作流 - 修复版本
正确展示如何接收和处理interrupt的返回值
"""

import os
import sys
import json
import re
import uuid
from typing import Dict, Any, Optional, Literal, TypedDict, List
from datetime import datetime
from pydantic import SecretStr
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from langchain_core.runnables import RunnableConfig
from langchain_community.chat_models.tongyi import ChatTongyi
from langgraph.graph import StateGraph, START, END
from langgraph.types import interrupt, Command
from langgraph.checkpoint.memory import MemorySaver


class PrivacyWorkflowState(TypedDict):
    """隐私号处理工作流状态定义"""
    user_input: str
    phone_number: Optional[str]
    privacy_type: Optional[Literal["可回拨", "不可回拨"]]
    privacy_number: Optional[str]
    current_step: str
    thinking_process: List[Dict[str, Any]]


class PrivacyWorkflow:
    """隐私号处理工作流类"""
    
    def __init__(self, api_key: str):
        self.api_key = api_key
        self.thinking_model = ChatTongyi(
            model="qwen-flash",
            api_key=SecretStr(api_key),
            top_p=0.8,
            streaming=True,
            model_kwargs={
                "temperature": 0.7,
                "enable_thinking": True
            }
        )
        self.memory = MemorySaver()
        self.graph = self._build_workflow_graph()
    
    def _build_workflow_graph(self) -> StateGraph:
        """构建工作流图"""
        workflow = StateGraph(PrivacyWorkflowState)
        
        workflow.add_node("business_check", self._check_business_relevance)
        workflow.add_node("phone_confirm", self._confirm_phone_number)
        workflow.add_node("type_confirm", self._confirm_privacy_type)
        workflow.add_node("generate", self._generate_privacy_number)
        
        workflow.add_edge(START, "business_check")
        workflow.add_edge("business_check", "phone_confirm")
        workflow.add_edge("phone_confirm", "type_confirm")
        workflow.add_edge("type_confirm", "generate")
        workflow.add_edge("generate", END)
        
        return workflow.compile(checkpointer=self.memory)
    
    def _check_business_relevance(self, state: PrivacyWorkflowState, config: RunnableConfig = None) -> PrivacyWorkflowState:
        """节点1: 业务相关性检查"""
        print(f"\n=== 节点1: 业务相关性检查 ===")
        
        # 检查输入是否包含业务关键词
        business_keywords = ["隐私号", "调查", "手机号", "拨打", "回拨"]
        is_relevant = any(keyword in state['user_input'] for keyword in business_keywords)
        
        if not is_relevant:
            print("🛑 业务不相关，触发中断")
            
            # ✅ 正确方式：接收interrupt的返回值
            new_input = interrupt({
                "type": "business",
                "message": "您的输入与隐私号业务无关。请重新输入相关请求。"
            })
            
            # 处理用户的新输入
            print(f"✅ 用户重新输入：{new_input}")
            state["user_input"] = new_input
            
            # 循环重新监察
            return self._check_business_relevance(state, config)
        
        print("✅ 业务相关性检查通过")
        state["current_step"] = "business_checked"
        return state
    
    def _confirm_phone_number(self, state: PrivacyWorkflowState, config: RunnableConfig = None) -> PrivacyWorkflowState:
        """节点2: 手机号确认"""
        print(f"\n=== 节点2: 手机号确认 ===")
        
        # 提取手机号
        phone_pattern = r'1[3-9]\d{9}'
        matches = re.findall(phone_pattern, state['user_input'])
        
        if not matches:
            print("🛑 未找到手机号，触发中断")
            
            # ✅ 正确方式：接收用户输入的手机号
            user_phone = interrupt({
                "type": "phone_input",
                "message": "请提供被调查人的手机号（11位数字）"
            })
            
            # 处理用户输入的手机号
            print(f"✅ 用户提供手机号：{user_phone}")
            
            # 验证手机号格式
            if re.match(phone_pattern, user_phone):
                state["phone_number"] = user_phone
                print(f"✅ 手机号格式验证通过：{user_phone}")
            else:
                print(f"❌ 手机号格式无效：{user_phone}")
                # 重新触发中断
                return self._confirm_phone_number(state, config)
        else:
            phone = matches[0]
            print(f"检测到手机号：{phone}")
            
            # 检查是否需要确认
            if "被调查人" not in state['user_input']:
                print("🛑 需要确认手机号用途，触发中断")
                
                # ✅ 正确方式：接收用户确认
                confirmation = interrupt({
                    "type": "phone_confirmation",
                    "message": f"检测到手机号：{phone}\n请确认这是被调查人员的手机号吗？\n输入 '是' 或 '否'",
                    "phone_number": phone
                })
                
                # 处理用户确认
                print(f"✅ 用户确认结果：{confirmation}")
                
                if confirmation == "是":
                    state["phone_number"] = phone
                    print(f"✅ 手机号确认成功：{phone}")
                else:
                    print(f"❌ 手机号被拒绝：{phone}")
                    # 清除手机号，重新开始
                    state["phone_number"] = None
                    return self._confirm_phone_number(state, config)
            else:
                state["phone_number"] = phone
                print(f"✅ 手机号用途明确：{phone}")
        
        state["current_step"] = "phone_confirmed"
        return state
    
    def _confirm_privacy_type(self, state: PrivacyWorkflowState, config: RunnableConfig = None) -> PrivacyWorkflowState:
        """节点3: 隐私号类型确认"""
        print(f"\n=== 节点3: 隐私号类型确认 ===")
        
        # 检查是否指定了类型
        if "可回拨" in state['user_input']:
            privacy_type = "可回拨"
        elif "不可回拨" in state['user_input']:
            privacy_type = "不可回拨"
        else:
            print("🛑 未指定隐私号类型，触发中断")
            
            # ✅ 正确方式：接收用户选择
            user_choice = interrupt({
                "type": "type_selection",
                "message": """
请选择隐私号类型：

1. 可回拨隐私号
   - 被调查人可以回拨隐私号
   - 回拨时会路由给调查员

2. 不可回拨隐私号  
   - 被调查人无法回拨隐私号
   - 只能调查员单向拨打

请输入 '1' 选择可回拨，或输入 '2' 选择不可回拨：
"""
            })
            
            # 处理用户选择
            print(f"✅ 用户选择：{user_choice}")
            
            if user_choice == "1":
                privacy_type = "可回拨"
            elif user_choice == "2":
                privacy_type = "不可回拨"
            else:
                print(f"❌ 无效选择：{user_choice}")
                # 重新触发中断
                return self._confirm_privacy_type(state, config)
        
        state["privacy_type"] = privacy_type
        print(f"✅ 隐私号类型确认：{privacy_type}")
        state["current_step"] = "type_confirmed"
        return state
    
    def _generate_privacy_number(self, state: PrivacyWorkflowState, config: RunnableConfig = None) -> PrivacyWorkflowState:
        """节点4: 生成隐私号"""
        print(f"\n=== 节点4: 生成隐私号 ===")
        
        phone = state["phone_number"]
        privacy_type = state["privacy_type"]
        
        if not phone or not privacy_type:
            raise ValueError(f"信息不完整：phone={phone}, type={privacy_type}")
        
        # 生成隐私号
        if privacy_type == "可回拨":
            privacy_number = f"400{phone[3:7]}{phone[7:]}"
        else:
            privacy_number = f"300{phone[3:7]}{phone[7:]}"
        
        state["privacy_number"] = privacy_number
        state["current_step"] = "completed"
        
        print(f"🎉 成功生成隐私号：{privacy_number}")
        return state
    
    def run_workflow(self, user_input: str, thread_id: str = None) -> Dict[str, Any]:
        """运行工作流"""
        if thread_id is None:
            thread_id = str(uuid.uuid4())
        
        initial_state = PrivacyWorkflowState(
            user_input=user_input,
            phone_number=None,
            privacy_type=None,
            privacy_number=None,
            current_step="started",
            thinking_process=[]
        )
        
        config = {"configurable": {"thread_id": thread_id}}
        
        print(f"\n{'='*50}")
        print(f"🚀 开始隐私号处理工作流")
        print(f"📝 用户输入：{user_input}")
        print(f"🆔 线程ID：{thread_id}")
        print(f"{'='*50}")
        
        return self._execute_workflow(initial_state, config)
    
    def _execute_workflow(self, state: PrivacyWorkflowState, config: Dict[str, Any]) -> Dict[str, Any]:
        """执行工作流"""
        try:
            result = self.graph.invoke(state, config=config)
            
            if "__interrupt__" in result:
                payload = result["__interrupt__"][0]
                interrupt_data = payload.value if hasattr(payload, 'value') else payload
                
                print(f"\n⏸️  工作流中断，等待用户确认...")
                print(f"📢 {interrupt_data.get('message', '需要用户确认')}")
                
                user_response = input("\n请输入您的回复：").strip()
                return self._resume_workflow(user_response, config["configurable"]["thread_id"])
            
            return self._format_final_result(result)
            
        except Exception as e:
            print(f"❌ 工作流执行失败: {e}")
            return {"error": str(e), "status": "failed"}
    
    def _resume_workflow(self, user_response: str, thread_id: str) -> Dict[str, Any]:
        """恢复工作流"""
        config = {"configurable": {"thread_id": thread_id}}
        
        print(f"▶️  恢复工作流，用户回复：{user_response}")
        
        try:
            result = self.graph.invoke(Command(resume=user_response), config=config)
            
            if "__interrupt__" in result:
                payload = result["__interrupt__"][0]
                interrupt_data = payload.value if hasattr(payload, 'value') else payload
                
                print(f"\n⏸️  工作流再次中断，等待用户确认...")
                print(f"📢 {interrupt_data.get('message', '需要用户确认')}")
                
                next_response = input("\n请输入您的回复：").strip()
                return self._resume_workflow(next_response, thread_id)
            
            return self._format_final_result(result)
            
        except Exception as e:
            print(f"❌ 工作流恢复失败: {e}")
            return {"error": str(e), "status": "failed"}
    
    def _format_final_result(self, final_state: PrivacyWorkflowState) -> Dict[str, Any]:
        """格式化最终结果"""
        result = {
            "status": "completed",
            "result": {
                "phone_number": final_state.get("phone_number"),
                "privacy_type": final_state.get("privacy_type"),
                "privacy_number": final_state.get("privacy_number"),
                "current_step": final_state.get("current_step")
            }
        }
        
        print(f"\n{'='*50}")
        print(f"✅ 隐私号处理完成！")
        print(f"📱 被调查人手机号：{result['result']['phone_number']}")
        print(f"🔧 隐私号类型：{result['result']['privacy_type']}")
        print(f"🎯 生成的隐私号：{result['result']['privacy_number']}")
        print(f"{'='*50}")
        
        return result


if __name__ == "__main__":
    # 测试修复后的工作流
    api_key = "sk-df68b4ca15e0497e83894b6a783ee024"
    workflow = PrivacyWorkflow(api_key)
    
    print("隐私号处理工作流测试（修复版本）")
    print("输入示例：给被调查人13812345678生成可回拨隐私号")
    
    user_input = input("\n请输入您的请求：")
    result = workflow.run_workflow(user_input)
    
    print(f"\n最终结果：")
    print(json.dumps(result, ensure_ascii=False, indent=2))
