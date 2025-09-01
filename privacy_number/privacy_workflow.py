#!/usr/bin/env python3
"""
隐私号处理工作流 - 严格按需求实现的线性流程版本
使用 LangGraph 实现4节点线性工作流，包含思考过程展示和人机交互点
按照LangGraph官方指南正确实现中断机制
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
    # 用户输入
    user_input: str
    
    # 业务核心数据
    phone_number: Optional[str]  # 被调查人手机号
    privacy_type: Optional[Literal["可回拨", "不可回拨"]]  # 隐私号类型
    privacy_number: Optional[str]  # 生成的隐私号
    
    # 流程控制
    current_step: str  # 当前执行步骤
    
    # 思考过程记录
    thinking_process: List[Dict[str, Any]]  # 记录每个节点的思考过程


class PrivacyWorkflow:
    """隐私号处理工作流类"""
    
    def __init__(self, api_key: str):
        """初始化工作流"""
        self.api_key = api_key
        
        # 初始化思考模型
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
        
        # 创建检查点内存
        self.memory = MemorySaver()
        
        # 构建工作流图
        self.graph = self._build_workflow_graph()
    
    def _build_workflow_graph(self) -> StateGraph:
        """构建4节点线性工作流图"""
        workflow = StateGraph(PrivacyWorkflowState)
        
        # 添加4个节点
        workflow.add_node("business_relevance_check", self._check_business_relevance)
        workflow.add_node("phone_confirmation", self._confirm_phone_number) 
        workflow.add_node("type_confirmation", self._confirm_privacy_type)
        workflow.add_node("privacy_number_generation", self._generate_privacy_number)
        
        # 构建线性流程
        workflow.add_edge(START, "business_relevance_check")
        workflow.add_edge("business_relevance_check", "phone_confirmation")
        workflow.add_edge("phone_confirmation", "type_confirmation")
        workflow.add_edge("type_confirmation", "privacy_number_generation")
        workflow.add_edge("privacy_number_generation", END)
        
        # 编译工作流
        compiled_graph = workflow.compile(checkpointer=self.memory)
        return compiled_graph
    

    def _check_business_relevance(self, state: PrivacyWorkflowState, config: RunnableConfig = None) -> PrivacyWorkflowState:
        """节点1: 判断输入是否与隐私号业务相关（思考节点+中断点）"""
        print(f"\n=== 节点1: 业务相关性判断 ===")
        
        # 构建分析提示
        analysis_prompt = f"""
        你是隐私号平台的AI助手，需要判断用户输入是否与隐私号业务场景相关。

        隐私号业务场景说明：
        - 调查员需要保护身份，通过平台为被调查人员的手机号生成隐私号
        - 调查员拨打隐私号，平台路由到被调查人员手机
        - 被调查人员看到的是隐私号，而非调查员真实号码
        - 隐私号分为可回拨和不可回拨两种类型

        用户输入：{state['user_input']}

        请仔细分析这个输入是否与隐私号业务相关，并给出判断理由。
        
        返回JSON格式：
        {{
            "is_relevant": true/false,
            "reason": "判断理由",
            "extracted_info": {{
                "phone_number": "如果能提取到手机号则填入，否则为null",
                "privacy_type": "如果能提取到隐私号类型则填入，否则为null"
            }}
        }}
        """
        
        # 调用思考模型并记录思考过程
        thinking_result = self._call_thinking_model(analysis_prompt, "业务相关性判断")
        state["thinking_process"].append(thinking_result)
        
        # 解析结果
        try:
            result = self._extract_json_from_response(thinking_result["response"])
            is_relevant = result.get("is_relevant", False)
            reason = result.get("reason", "")
            extracted_info = result.get("extracted_info", {})
            
            # 提取可能的手机号和类型信息
            if extracted_info.get("phone_number"):
                state["phone_number"] = extracted_info["phone_number"]
            if extracted_info.get("privacy_type") in ["可回拨", "不可回拨"]:
                state["privacy_type"] = extracted_info["privacy_type"]
            
            if not is_relevant:
                # 不相关，直接中断让用户重新输入
                print(f"🛑 触发业务相关性中断")
                interrupt({
                    "type": "business",
                    "message": f"您的输入似乎与隐私号业务无关。{reason}\n\n隐私号业务说明：\n- 为调查员生成隐私号保护身份\n- 输入被调查人手机号，选择可回拨或不可回拨类型\n- 获得可用于拨打的隐私号\n\n请重新输入相关的业务请求。"
                })
            
        except Exception as e:
            print(f"解析结果失败: {e}")
            # 使用后备判断逻辑
            if not self._fallback_business_check(state["user_input"]):
                print(f"🛑 触发业务相关性中断（后备逻辑）")
                interrupt({
                    "type": "business", 
                    "message": "您的输入似乎与隐私号业务无关。请重新输入相关的业务请求。"
                })
        
        state["current_step"] = "business_relevance_completed"
        return state
    
    def _confirm_phone_number(self, state: PrivacyWorkflowState, config: RunnableConfig = None) -> PrivacyWorkflowState:
        """节点2: 确认手机号是否为被调查人的（思考节点+中断点）"""
        print(f"\n=== 节点2: 手机号确认 ===")
        
        # 构建分析提示
        analysis_prompt = f"""
        用户输入：{state['user_input']}
        
        请分析这个输入中是否明确包含被调查人员的手机号，以及是否明确说明了这是被调查人员的号码。
        
        分析要点：
        1. 是否包含有效的手机号码（1[3-9]xxxxxxxxx格式）
        2. 是否明确说明这是被调查人员/目标人员的手机号
        3. 如果有手机号但没有明确说明用途，需要用户确认
        
        返回JSON格式：
        {{
            "has_phone": true/false,
            "phone_number": "提取的手机号或null",
            "is_target_phone_explicit": true/false,
            "reason": "分析理由",
            "needs_confirmation": true/false
        }}
        """
        
        # 调用思考模型
        thinking_result = self._call_thinking_model(analysis_prompt, "手机号确认")
        state["thinking_process"].append(thinking_result)
        
        try:
            result = self._extract_json_from_response(thinking_result["response"])
            has_phone = result.get("has_phone", False)
            phone_number = result.get("phone_number")
            is_explicit = result.get("is_target_phone_explicit", False)
            needs_confirmation = result.get("needs_confirmation", False)
            reason = result.get("reason", "")
            
            if phone_number:
                state["phone_number"] = phone_number
            
            if not has_phone:
                # 没有手机号，需要用户提供
                print(f"🛑 触发手机号中断")
                interrupt({
                    "type": "phone",
                    "message": f"未能识别到有效的手机号码。{reason}\n请提供被调查人员的手机号码（11位数字，如：13812345678）"
                })
                
            elif not is_explicit or needs_confirmation:
                # 有手机号但需要确认是否为被调查人的
                print(f"🛑 触发手机号确认中断")
                interrupt({
                    "type": "phone",
                    "message": f"检测到手机号：{phone_number}\n请确认这是被调查人员的手机号吗？\n输入 '是' 或 '否'",
                    "phone_number": phone_number
                })
                
        except Exception as e:
            print(f"解析结果失败: {e}")
            # 使用后备逻辑提取手机号
            phone = self._fallback_extract_phone(state["user_input"])
            if phone:
                state["phone_number"] = phone
                print(f"🛑 触发手机号确认中断（后备逻辑）")
                interrupt({
                    "type": "phone",
                    "message": f"检测到手机号：{phone}\n请确认这是被调查人员的手机号吗？\n输入 '是' 或 '否'",
                    "phone_number": phone
                })
            else:
                print(f"🛑 触发手机号中断（后备逻辑）")
                interrupt({
                    "type": "phone",
                    "message": "未能识别到有效的手机号码。请提供被调查人员的手机号码（11位数字，如：13812345678）"
                })
            
        state["current_step"] = "phone_confirmation_completed"
        return state
    
    def _confirm_privacy_type(self, state: PrivacyWorkflowState, config: RunnableConfig = None) -> PrivacyWorkflowState:
        """节点3: 确认隐私号类型（思考节点+中断点）"""
        print(f"\n=== 节点3: 隐私号类型确认 ===")
        
        # 构建分析提示
        analysis_prompt = f"""
        用户输入：{state['user_input']}
        当前已确认手机号：{state.get('phone_number', 'N/A')}
        
        请分析用户是否明确指定了隐私号的类型。
        
        隐私号类型说明：
        - 可回拨：被调查人可以回拨隐私号，平台会路由给调查员
        - 不可回拨：被调查人无法回拨，只能单向拨打
        
        分析用户输入中是否包含以下信息：
        1. 明确提到"可回拨"或"不可回拨"
        2. 有类似意图的表达（如"需要回拨功能"、"单向拨打"等）
        
        返回JSON格式：
        {{
            "type_specified": true/false,
            "privacy_type": "可回拨/不可回拨/null",
            "confidence": "high/medium/low",
            "reason": "分析理由",
            "needs_confirmation": true/false
        }}
        """
        
        # 调用思考模型
        thinking_result = self._call_thinking_model(analysis_prompt, "隐私号类型确认")
        state["thinking_process"].append(thinking_result)
        
        try:
            result = self._extract_json_from_response(thinking_result["response"])
            type_specified = result.get("type_specified", False)
            privacy_type = result.get("privacy_type")
            confidence = result.get("confidence", "low")
            reason = result.get("reason", "")
            needs_confirmation = result.get("needs_confirmation", True)
            
            if privacy_type and privacy_type in ["可回拨", "不可回拨"]:
                state["privacy_type"] = privacy_type
                
            if not type_specified or confidence == "low" or needs_confirmation:
                # 需要用户确认隐私号类型
                print(f"🛑 触发类型确认中断")
                interrupt({
                    "type": "type",
                    "message": f"""
请选择隐私号类型：

1. 可回拨隐私号
   - 被调查人可以回拨隐私号
   - 回拨时会路由给调查员
   - 适合需要双向通信的调查场景

2. 不可回拨隐私号  
   - 被调查人无法回拨隐私号
   - 只能调查员单向拨打
   - 适合单向联系的调查场景

请输入 '1' 选择可回拨，或输入 '2' 选择不可回拨：
"""
                })
                
        except Exception as e:
            print(f"解析结果失败: {e}")
            # 需要用户选择类型
            print(f"触发类型确认中断（后备逻辑）")
            interrupt({
                "type": "type",
                "message": f"""
请选择隐私号类型：

1. 可回拨隐私号
2. 不可回拨隐私号

请输入 '1' 选择可回拨，或输入 '2' 选择不可回拨：
"""
            })
        
        state["current_step"] = "type_confirmation_completed"
        return state
    
    def _generate_privacy_number(self, state: PrivacyWorkflowState, config: RunnableConfig = None) -> PrivacyWorkflowState:
        """节点4: 生成隐私号（工具节点）"""
        print(f"\n=== 节点4: 隐私号生成 ===")
        
        phone = state["phone_number"]
        privacy_type = state["privacy_type"]
        
        if not phone or not privacy_type:
            raise ValueError(f"生成隐私号所需信息不完整：phone={phone}, type={privacy_type}")
        
        # 生成隐私号逻辑
        if privacy_type == "可回拨":
            # 可回拨隐私号使用400开头
            privacy_number = f"400{phone[3:7]}{phone[7:]}"
        else:
            # 不可回拨隐私号使用300开头
            privacy_number = f"300{phone[3:7]}{phone[7:]}"
        
        state["privacy_number"] = privacy_number
        state["current_step"] = "completed"
        
        # 记录生成过程
        generation_info = {
            "timestamp": datetime.now().isoformat(),
            "stage": "隐私号生成",
            "thinking": f"为手机号 {phone} 生成 {privacy_type} 隐私号",
            "response": f"成功生成隐私号：{privacy_number}",
            "type": "tool_execution"
        }
        state["thinking_process"].append(generation_info)
        
        print(f"✓ 成功为手机号 {phone} 生成 {privacy_type} 隐私号：{privacy_number}")
        
        return state
    
    def _call_thinking_model(self, prompt: str, stage: str) -> Dict[str, Any]:
        """调用思考模型并记录思考过程"""
        try:
            print(f" {stage} 思考中...")
            
            # 调用模型
            response = self.thinking_model.invoke([HumanMessage(content=prompt)])
            
            # 提取思考过程（如果有的话）
            thinking_content = ""
            if hasattr(response, 'additional_kwargs') and 'reasoning_content' in response.additional_kwargs:
                thinking_content = response.additional_kwargs['reasoning_content']
            
            result = {
                "timestamp": datetime.now().isoformat(),
                "stage": stage,
                "thinking": thinking_content,
                "response": response.content,
                "type": "thinking_process"
            }
            
            # 显示思考过程
            if thinking_content:
                print(f" AI思考过程：")
                print(f"{thinking_content}")
                print(f" AI分析结果：")
                print(f"{response.content}")
            else:
                print(f" AI分析结果：")
                print(f"{response.content}")
            
            return result
            
        except Exception as e:
            print(f" 模型调用失败: {e}")
            return {
                "timestamp": datetime.now().isoformat(), 
                "stage": stage,
                "thinking": "",
                "response": f"模型调用失败: {e}",
                "type": "error"
            }
    
    def _extract_json_from_response(self, response: str) -> Dict[str, Any]:
        """从响应中提取JSON"""
        try:
            # 查找JSON内容
            json_start = response.find("{")
            json_end = response.rfind("}") + 1
            if json_start != -1 and json_end > json_start:
                json_str = response[json_start:json_end]
                return json.loads(json_str)
            else:
                raise ValueError("未找到有效的JSON格式")
        except (json.JSONDecodeError, ValueError) as e:
            print(f"JSON解析失败: {e}")
            raise
    
    def _fallback_business_check(self, user_input: str) -> bool:
        """后备的业务相关性检查"""
        business_keywords = ["隐私号", "调查", "手机号", "拨打", "回拨", "路由", "匿名", "保护", "身份"]
        return any(keyword in user_input for keyword in business_keywords)
    
    def _fallback_extract_phone(self, text: str) -> Optional[str]:
        """后备的手机号提取"""
        phone_pattern = r'1[3-9]\d{9}'
        matches = re.findall(phone_pattern, text)
        return matches[0] if matches else None
    
    
    def run_workflow(self, user_input: str, thread_id: str = None) -> Dict[str, Any]:
        """运行工作流主入口"""
        if thread_id is None:
            thread_id = str(uuid.uuid4())
            
        # 初始化状态
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
        print(f" 开始隐私号处理工作流")
        print(f" 用户输入：{user_input}")
        print(f" 线程ID：{thread_id}")
        print(f"{'='*50}")
        
        return self._execute_workflow(initial_state, config)
    
    def _execute_workflow(self, state: PrivacyWorkflowState, config: Dict[str, Any]) -> Dict[str, Any]:
        """执行工作流（支持递归处理中断）"""
        try:
            # 运行工作流
            result = self.graph.invoke(state, config=config)
            
            # 检查是否有中断
            if "__interrupt__" in result:
                payload = result["__interrupt__"]
                if isinstance(payload, list) and payload:
                    payload = payload[0]
                
                # 正确处理Interrupt对象
                if hasattr(payload, 'value'):
                    interrupt_data = payload.value
                else:
                    interrupt_data = payload
                
                print(f"\n⏸️  工作流中断，等待用户确认...")
                print(f"📢 {interrupt_data.get('message', '需要用户确认')}")
                
                # 获取用户输入
                user_response = input("\n请输入您的回复：").strip()
                
                # 恢复工作流
                return self._resume_workflow(user_response, config["configurable"]["thread_id"])
            
            # 工作流完成
            return self._format_final_result(result)
            
        except Exception as e:
            print(f" 工作流执行失败: {e}")
            return {"error": str(e), "status": "failed"}
    
    def _resume_workflow(self, user_response: str, thread_id: str) -> Dict[str, Any]:
        """恢复被中断的工作流"""
        config = {"configurable": {"thread_id": thread_id}}
        
        print(f"▶  恢复工作流，用户回复：{user_response}")
        
        # 处理特殊的用户响应
        processed_response = self._process_user_response(user_response)
        print(f" 处理后的响应：{processed_response}")
        
        try:
            # 使用Command恢复工作流
            result = self.graph.invoke(Command(resume=processed_response), config=config)
            
            # 检查是否还有新的中断
            if "__interrupt__" in result:
                payload = result["__interrupt__"]
                if isinstance(payload, list) and payload:
                    payload = payload[0]
                
                # 正确处理Interrupt对象
                if hasattr(payload, 'value'):
                    interrupt_data = payload.value
                else:
                    interrupt_data = payload
                
                print(f"  工作流再次中断，等待用户确认...")
                print(f" {interrupt_data.get('message', '需要用户确认')}")
                
                # 递归处理多次中断
                next_user_response = input("\n请输入您的回复：").strip()
                return self._resume_workflow(next_user_response, thread_id)
            
            # 工作流完成
            return self._format_final_result(result)
            
        except Exception as e:
            print(f" 工作流恢复失败: {e}")
            return {"error": str(e), "status": "failed"}
    
    def _process_user_response(self, response: str) -> str:
        """处理和标准化用户响应"""
        response = response.strip()
        
        # 检查是否是手机号（如果用户提供了新的手机号）
        phone_pattern = r'1[3-9]\d{9}'
        if re.match(phone_pattern, response):
            return response  # 返回手机号本身
        
        # 标准化确认响应
        if response.lower() in ['是', 'yes', 'y', '确认', '对', '正确']:
            return "是"
        elif response.lower() in ['否', 'no', 'n', '不是', '错误']:
            return "否"
        elif response in ['1']:
            return "可回拨"  # 直接返回类型
        elif response in ['2']:
            return "不可回拨"  # 直接返回类型
        
        return response
    
    def _format_final_result(self, final_state: PrivacyWorkflowState) -> Dict[str, Any]:
        """格式化最终结果"""
        result = {
            "status": "completed",
            "result": {
                "phone_number": final_state.get("phone_number"),
                "privacy_type": final_state.get("privacy_type"), 
                "privacy_number": final_state.get("privacy_number"),
                "current_step": final_state.get("current_step")
            },
            "thinking_process": final_state.get("thinking_process", [])
        }
        
        # 显示最终结果
        print(f"\n{'='*50}")
        print(f"✅ 隐私号处理完成！")
        print(f" 被调查人手机号：{result['result']['phone_number']}")
        print(f" 隐私号类型：{result['result']['privacy_type']}")
        print(f" 生成的隐私号：{result['result']['privacy_number']}")
        print(f"{'='*50}")
        
        return result


if __name__ == "__main__":
    # 简单测试
    api_key = "sk-df68b4ca15e0497e83894b6a783ee024"
    workflow = PrivacyWorkflow(api_key)
    
    print("隐私号处理工作流测试")
    print("输入示例：给被调查人13812345678生成可回拨隐私号")
    
    user_input = input("\n请输入您的请求：")
    result = workflow.run_workflow(user_input)
    
    print(f"\n最终结果：")
    print(json.dumps(result, ensure_ascii=False, indent=2))
