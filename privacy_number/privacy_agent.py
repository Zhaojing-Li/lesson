"""
隐私号平台处理系统
基于LangGraph框架实现的隐私号生成与管理Agent
"""

import re
import json
from typing import Dict, Any, List, Optional, Literal, TypedDict, Annotated
from datetime import datetime
from pydantic import BaseModel, SecretStr
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from langchain_community.chat_models.tongyi import ChatTongyi
from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import MemorySaver
from langgraph.prebuilt import create_react_agent
from langgraph.graph.message import add_messages
from typing import Union
import config


class PrivacyNumberState(TypedDict):
    """隐私号处理状态定义"""
    messages: Annotated[List[dict], add_messages]
    user_input: str
    phone_number: Optional[str]
    phone_confirmed: bool
    privacy_type: Optional[Literal["可回拨", "不可回拨"]]
    type_confirmed: bool
    reasoning_process: List[str]
    privacy_number: Optional[str]
    need_human_confirmation: bool
    confirmation_message: Optional[str]
    step: str
    # 新增字段支持流式输出和重新输入
    need_reinput: bool
    reinput_reason: Optional[str]
    thinking_process: List[str]
    final_results: List[str]
    stream_data: List[dict]


class PrivacyNumberAgent:
    """隐私号处理Agent主类"""
    
    def __init__(self):
        """初始化Agent"""
        # 创建配置实例
        cfg = config.Config()
        
        self.chat_model = ChatTongyi(
            model=cfg.llm_config.model,  
            api_key=SecretStr(cfg.llm_config.api_key),
            top_p=0.8,         
            streaming=True,   
            model_kwargs={
                "temperature": 0.5,      
                "enable_thinking": True  
            }
        )
        print(f"获取到qwen的配置:{cfg.llm_config.api_key}")
        
        # 创建内存检查点
        self.memory = MemorySaver()
        # 构建工作流图
        self.graph = self._build_graph()
        
    
    def _build_graph(self):
        """构建LangGraph工作流"""
        workflow = StateGraph(PrivacyNumberState)
        
        # 添加节点
        workflow.add_node("input_analysis", self._analyze_input)
        workflow.add_node("phone_validation", self._validate_phone)
        workflow.add_node("type_inference", self._infer_privacy_type)
        workflow.add_node("human_confirmation", self._handle_human_confirmation)
        workflow.add_node("privacy_generation", self._generate_privacy_number)
        workflow.add_node("result_presentation", self._present_result)
        
        # 定义流程边
        workflow.add_edge(START, "input_analysis")
        
        # 条件边：根据推理结果决定是否需要确认
        workflow.add_conditional_edges(
            "input_analysis",
            self._decide_next_step,
            {
                "phone_validation": "phone_validation",
                "type_inference": "type_inference",
                "human_confirmation": "human_confirmation"
            }
        )
        
        workflow.add_conditional_edges(
            "phone_validation",
            self._decide_after_phone_validation,
            {
                "type_inference": "type_inference",
                "human_confirmation": "human_confirmation"
            }
        )
        
        workflow.add_conditional_edges(
            "type_inference", 
            self._decide_after_type_inference,
            {
                "privacy_generation": "privacy_generation",
                "human_confirmation": "human_confirmation"
            }
        )
        
        workflow.add_conditional_edges(
            "human_confirmation",
            self._decide_after_confirmation,
            {
                "phone_validation": "phone_validation",
                "type_inference": "type_inference", 
                "privacy_generation": "privacy_generation",
                "result_presentation": "result_presentation"
            }
        )
        
        workflow.add_edge("privacy_generation", "result_presentation")
        workflow.add_edge("result_presentation", END)
        
        return workflow.compile(checkpointer=self.memory)
    

    async def _analyze_input(self, state: PrivacyNumberState) -> PrivacyNumberState:
        """分析用户输入，进行语义理解与推理"""
        user_input = state["user_input"]
        
        # 构建分析提示 - 优化后的prompt
        analysis_prompt = f"""
        作为隐私号平台的专业AI助手，请对用户请求进行综合分析：
        
        用户输入：{user_input}
        
        隐私号平台业务说明：
        1. 隐私号是为保护调查员身份而生成的中转号码
        2. 调查员输入被调查人员的手机号，系统生成隐私号
        3. 调查员拨打隐私号，系统路由到被调查人员
        4. 被调查人员看到的是隐私号，而非调查员真实号码
        
        隐私号类型：
        - 可回拨：被调查人员可以回拨隐私号联系调查员
        - 不可回拨：被调查人员无法通过隐私号回拨调查员
        
        请按以下步骤进行分析：
        
        第一步：业务相关性判断
        - 用户的请求是否与隐私号生成相关？
        - 是否属于隐私通信、电话中转、号码隐藏等业务范围？
        - 如果与业务无关（如询问天气、闲聊等），需要提示用户
        
        第二步：信息完整性验证
        - 用户是否提供了必要的信息？
        - 手机号码、隐私号需求等关键信息是否完整？
        - 如果信息不足（如只输入1502、只说“帮我生成”），需要指导用户补充
        
        第三步：意图识别与风险评估
        - 用户是否明确说明手机号属于被调查人员？
        - 是否存在误输入自己手机号的风险？
        - 是否需要确认隐私号类型？
        
        请返回JSON格式的分析结果：
        {{
            "business_related": true/false,
            "information_complete": true/false,
            "extracted_phone": "提取到的手机号或null",
            "privacy_type_specified": "可回拨/不可回拨/未指定",
            "needs_confirmation": true/false,
            "needs_reinput": true/false,
            "reinput_reason": "重新输入的原因",
            "next_action": "continue/confirm/reinput",
            "analysis_summary": "分析总结"
        }}
        """
        
        # 使用流式输出获取模型响应
        thinking_content = []
        final_content = []
        
        async for chunk in self.chat_model.astream([
            SystemMessage(content="你是隐私号平台的专业AI助手，具备精准的语义理解和业务判断能力。"),
            HumanMessage(content=analysis_prompt)
        ]):
            # 解析流式响应，区分thinking和content
            chunk_data = self._parse_stream_chunk(chunk)
            if chunk_data:
                if chunk_data["type"] == "thinking":
                    thinking_content.append(chunk_data["content"])
                    state["thinking_process"].append(chunk_data["content"])
                elif chunk_data["type"] == "content":
                    final_content.append(chunk_data["content"])
                    state["final_results"].append(chunk_data["content"])
                
                # 添加到流式数据中供前端显示
                state["stream_data"].append({
                    "timestamp": datetime.now().isoformat(),
                    "stage": "输入分析",
                    "type": chunk_data["type"],
                    "content": chunk_data["content"]
                })
        
        # 合并最终结果
        full_response = "".join(final_content)
        
        # 记录推理过程
        thinking_summary = "".join(thinking_content)
        state["reasoning_process"].append(f"输入分析阶段思考：{thinking_summary}")
        state["reasoning_process"].append(f"输入分析阶段结果：{full_response}")
        
        # 解析JSON结果
        try:
            # 提取JSON部分
            json_start = full_response.find("{")
            json_end = full_response.rfind("}") + 1
            if json_start != -1 and json_end > json_start:
                json_str = full_response[json_start:json_end]
                analysis_result = json.loads(json_str)
            else:
                raise ValueError("未找到有效的JSON")
        except (json.JSONDecodeError, ValueError) as e:
            # JSON解析失败，使用关键词匹配
            analysis_result = self._fallback_analysis(full_response, user_input)
        
        # 根据分析结果更新状态
        if not analysis_result.get("business_related", True):
            # 业务无关
            state["need_reinput"] = True
            state["reinput_reason"] = "您的请求与隐私号生成业务无关。请输入与隐私号、电话中转或号码隐藏相关的需求。"
            state["need_human_confirmation"] = True
            state["confirmation_message"] = f"{state['reinput_reason']}\n\n您可以这样说：\n- '为138xxxx5678生成一个可回拨的隐私号'\n- '需要一个不可回拨的隐私号给159xxxx1234'"
        elif not analysis_result.get("information_complete", True):
            # 信息不完整
            state["need_reinput"] = True
            state["reinput_reason"] = "您提供的信息不够完整。生成隐私号需要提供被调查人员的完整手机号码。"
            state["need_human_confirmation"] = True
            state["confirmation_message"] = f"{state['reinput_reason']}\n\n请重新输入，包含：\n1. 被调查人员的完整手机号（11位数字）\n2. 隐私号类型（可回拨或不可回拨）"
        else:
            # 信息完整，继续处理
            extracted_phone = analysis_result.get("extracted_phone")
            if extracted_phone and extracted_phone != "null":
                state["phone_number"] = extracted_phone
            
            privacy_type = analysis_result.get("privacy_type_specified", "未指定")
            if privacy_type in ["可回拨", "不可回拨"]:
                state["privacy_type"] = privacy_type
                state["type_confirmed"] = True
            
            state["need_reinput"] = False
        
        state["step"] = "input_analysis_completed"
        state["messages"].append({
            "role": "assistant", 
            "content": f"正在分析您的请求...\n分析结果：{analysis_result.get('analysis_summary', full_response)}"
        })
        
        return state
    
    async def _validate_phone(self, state: PrivacyNumberState) -> PrivacyNumberState:
        """验证手机号码"""
        phone = state.get("phone_number")
        user_input = state["user_input"]
        
        # 使用模型进行语义推理
        validation_prompt = f"""
        用户输入：{user_input}
        提取到的手机号：{phone}
        
        请分析：用户是否明确说明这个手机号是"被调查人员"的号码？
        如果用户只是说"帮我生成隐私号"或类似表述，没有明确说明是被调查人员的号码，
        则存在误输入自己号码的风险，需要确认。
        
        返回：需要确认 或 无需确认，并说明理由。
        """
        
        response = await self.chat_model.ainvoke([
            SystemMessage(content="你是专业的语义分析专家，帮助判断用户意图。"),
            HumanMessage(content=validation_prompt)
        ])
        
        reasoning = response.content
        state["reasoning_process"].append(f"手机号验证阶段：{reasoning}")
        
        # 根据模型推理结果决定是否需要确认
        if "需要确认" in reasoning:
            state["need_human_confirmation"] = True
            state["confirmation_message"] = f"检测到手机号 {phone}，请确认这是被调查人员的手机号，而不是您自己的号码？"
        else:
            state["phone_confirmed"] = True
            
        state["step"] = "phone_validation_completed"
        return state
    

    async def _infer_privacy_type(self, state: PrivacyNumberState) -> PrivacyNumberState:
        """推理隐私号类型需求"""
        user_input = state["user_input"]
        
        type_inference_prompt = f"""
        用户请求：{user_input}
        
        隐私号类型说明：
        - 可回拨：被调查人员可以回拨隐私号联系调查员（用于需要双向沟通的调查）
        - 不可回拨：被调查人员无法回拨（用于单向调查，保护调查员隐私）
        
        请分析用户是否明确指定了隐私号类型？
        如果没有明确指定，是否可以从语境中推断？
        
        返回分析结果和建议。
        """
        
        response = await self.chat_model.ainvoke([
            SystemMessage(content="你是隐私号平台的业务专家，擅长理解用户需求。"),
            HumanMessage(content=type_inference_prompt)
        ])
        
        reasoning = response.content
        state["reasoning_process"].append(f"类型推理阶段：{reasoning}")
        
        # 简单的关键词检测（实际应用中可以更复杂）
        if "可回拨" in user_input:
            state["privacy_type"] = "可回拨"
            state["type_confirmed"] = True
        elif "不可回拨" in user_input or "单向" in user_input:
            state["privacy_type"] = "不可回拨" 
            state["type_confirmed"] = True
        else:
            # 需要用户确认类型
            state["need_human_confirmation"] = True
            state["confirmation_message"] = "请选择隐私号类型：\n1. 可回拨（被调查人员可以回拨联系您）\n2. 不可回拨（被调查人员无法回拨）"
            
        state["step"] = "type_inference_completed"
        return state
    
    
    async def _handle_human_confirmation(self, state: PrivacyNumberState) -> PrivacyNumberState:
        """处理人机交互确认"""
        # 这里在实际实现中会暂停等待用户输入
        # 在demo中，我们模拟用户确认
        
        confirmation_msg = state.get("confirmation_message", "")
        state["reasoning_process"].append(f"人机交互确认：{confirmation_msg}")
        
        # 标记为需要前端处理的确认点
        state["step"] = "awaiting_human_confirmation"
        state["messages"].append({
            "role": "assistant",
            "content": confirmation_msg
        })
        
        return state
    
    async def _generate_privacy_number(self, state: PrivacyNumberState) -> PrivacyNumberState:
        """生成隐私号（模拟实现）"""
        phone = state["phone_number"]
        privacy_type = state["privacy_type"]
        
        # 检查手机号是否为空
        if not phone:
            raise ValueError("手机号不能为空")
            
        # 模拟隐私号生成逻辑
        if privacy_type == "可回拨":
            privacy_number = await self._generate_callable_privacy_number(phone)
        else:
            privacy_number = await self._generate_non_callable_privacy_number(phone)
            
        state["privacy_number"] = privacy_number
        state["step"] = "privacy_number_generated"
        
        reasoning = f"已为手机号 {phone} 生成 {privacy_type} 隐私号：{privacy_number}"
        state["reasoning_process"].append(reasoning)
        
        return state
    
    async def _present_result(self, state: PrivacyNumberState) -> PrivacyNumberState:
        """展示最终结果"""
        phone = state["phone_number"]
        privacy_type = state["privacy_type"] 
        privacy_number = state["privacy_number"]
        
        result_message = f"""
        ✅ 隐私号生成成功！
        
        📞 被调查人员手机号：{phone}
        🔒 生成的隐私号：{privacy_number}
        📋 隐私号类型：{privacy_type}
        
        📌 使用说明：
        - 请拨打隐私号 {privacy_number} 联系被调查人员
        - 对方将看到隐私号显示，无法获取您的真实号码
        {'- 对方可以回拨此隐私号联系您' if privacy_type == '可回拨' else '- 对方无法回拨此隐私号'}
        
        💡 推理过程回顾：
        {chr(10).join(f"{i+1}. {step}" for i, step in enumerate(state["reasoning_process"]))}
        """
        
        state["messages"].append({
            "role": "assistant",
            "content": result_message
        })
        
        state["step"] = "completed"
        return state
    
    # 条件判断函数
    def _decide_next_step(self, state: PrivacyNumberState) -> str:
        """决定分析后的下一步"""
        if state.get("need_reinput"):
            return "human_confirmation"  # 需要重新输入
        if not state.get("phone_number"):
            return "human_confirmation"  # 没有手机号，需要用户提供
        return "phone_validation"
    
    def _decide_after_phone_validation(self, state: PrivacyNumberState) -> str:
        """决定手机号验证后的下一步"""
        if state.get("need_human_confirmation"):
            return "human_confirmation"
        return "type_inference"
    
    def _decide_after_type_inference(self, state: PrivacyNumberState) -> str:
        """决定类型推理后的下一步"""
        if state.get("need_human_confirmation"):
            return "human_confirmation"
        return "privacy_generation"
    
    def _decide_after_confirmation(self, state: PrivacyNumberState) -> str:
        """决定确认后的下一步"""
        if not state.get("phone_confirmed"):
            return "phone_validation"
        elif not state.get("type_confirmed"):
            return "type_inference"
        else:
            return "privacy_generation"
    
    # 隐私号生成工具方法（模拟实现）
    async def _generate_callable_privacy_number(self, phone: str) -> str:
        """生成可回拨隐私号"""
        # 模拟生成逻辑：在原号码基础上变换
        base = phone[3:7] 
        suffix = phone[7:]
        privacy_num = f"400{base}{suffix}"
        return privacy_num
    
    async def _generate_non_callable_privacy_number(self, phone: str) -> str:
        """生成不可回拨隐私号"""
        # 模拟生成逻辑
        base = phone[3:7]
        suffix = phone[7:] 
        privacy_num = f"300{base}{suffix}"
        return privacy_num
    
    async def process_request(self, user_input: str, thread_id: str = "default") -> Dict[str, Any]:
        """处理用户请求的主入口"""
        initial_state = PrivacyNumberState(
            messages=[{"role": "user", "content": user_input}],
            user_input=user_input,
            phone_number=None,
            phone_confirmed=False,
            privacy_type=None,
            type_confirmed=False,
            reasoning_process=[],
            privacy_number=None,
            need_human_confirmation=False,
            confirmation_message=None,
            step="started",
            # 新增字段初始化
            need_reinput=False,
            reinput_reason=None,
            thinking_process=[],
            final_results=[],
            stream_data=[]
        )
        
        config = {"configurable": {"thread_id": thread_id}}
        
        # 执行工作流
        result = await self.graph.ainvoke(initial_state, config=config)
        
        return result
    
    async def continue_conversation(self, user_response: str, thread_id: str = "default") -> Dict[str, Any]:
        """继续对话（处理用户确认回复）"""
        # 获取当前状态
        config = {"configurable": {"thread_id": thread_id}}
        current_state = await self.graph.aget_state(config)
        
        # 根据用户回复更新状态
        state = current_state.values
        state["messages"].append({"role": "user", "content": user_response})
        
        # 处理确认回复
        if "awaiting_human_confirmation" in state.get("step", ""):
            await self._process_confirmation_response(state, user_response)
        
        # 继续执行工作流
        result = await self.graph.ainvoke(state, config=config)
        return result
    
    async def _process_confirmation_response(self, state: PrivacyNumberState, response: str):
        """处理用户确认回复"""
        response_lower = response.lower()
        
        # 处理重新输入确认
        if state.get("need_reinput"):
            if "重新" in response or "确认" in response or "ok" in response_lower or "好" in response:
                # 用户确认重新输入，重置状态
                state["need_reinput"] = False
                state["need_human_confirmation"] = False
                state["confirmation_message"] = None
                # 但不清空原有信息，等待新输入
                return
        
        # 处理手机号确认
        confirmation_msg = state.get("confirmation_message") or ""
        if "请确认这是被调查人员的手机号" in confirmation_msg:
            if "是" in response or "yes" in response_lower or "确认" in response:
                state["phone_confirmed"] = True
                state["need_human_confirmation"] = False
            else:
                # 用户否认，需要重新输入手机号
                state["phone_number"] = None
                state["need_human_confirmation"] = False
        
        # 处理隐私号类型确认
        elif "请选择隐私号类型" in confirmation_msg:
            if "1" in response or "可回拨" in response:
                state["privacy_type"] = "可回拨"
                state["type_confirmed"] = True
                state["need_human_confirmation"] = False
            elif "2" in response or "不可回拨" in response:
                state["privacy_type"] = "不可回拨"
                state["type_confirmed"] = True
                state["need_human_confirmation"] = False
    
    def _parse_stream_chunk(self, chunk) -> Optional[dict]:
        """解析流式响应块，区分thinking和content"""
        try:
            if hasattr(chunk, 'content') and chunk.content:
                content = chunk.content
                
                # 检查是否包含思考标记
                if '<thinking>' in content and '</thinking>' in content:
                    # 提取思考过程
                    thinking_start = content.find('<thinking>') + len('<thinking>')
                    thinking_end = content.find('</thinking>')
                    thinking_content = content[thinking_start:thinking_end].strip()
                    
                    if thinking_content:
                        return {
                            "type": "thinking",
                            "content": thinking_content
                        }
                
                # 如果不包含思考标记，则作为最终内容
                if content.strip() and '<thinking>' not in content:
                    return {
                        "type": "content",
                        "content": content
                    }
                    
        except Exception as e:
            print(f"解析流式响应块错误: {e}")
            
        return None
    
    def _fallback_analysis(self, response_text: str, user_input: str) -> dict:
        """当JSON解析失败时的后备分析方法"""
        # 使用关键词匹配进行基本分析
        business_related = any(keyword in user_input.lower() for keyword in [
            '隐私号', '手机号', '电话', '号码', '中转', '生成', 
            '调查', '联系', '拨打', '回拨', 'privacy', 'phone'
        ])
        
        # 检查手机号
        phone_pattern = r'1[3-9]\d{9}'
        phones = re.findall(phone_pattern, user_input)
        extracted_phone = phones[0] if phones else None
        
        # 检查信息完整性
        information_complete = bool(extracted_phone) and len(user_input.strip()) > 10
        
        # 检查隐私号类型
        privacy_type = "未指定"
        if '可回拨' in user_input:
            privacy_type = "可回拨"
        elif '不可回拨' in user_input or '单向' in user_input:
            privacy_type = "不可回拨"
        
        return {
            "business_related": business_related,
            "information_complete": information_complete,
            "extracted_phone": extracted_phone,
            "privacy_type_specified": privacy_type,
            "needs_confirmation": not information_complete,
            "needs_reinput": not business_related or not information_complete,
            "reinput_reason": "信息不完整或与业务无关" if not business_related or not information_complete else "",
            "next_action": "reinput" if not business_related or not information_complete else "continue",
            "analysis_summary": f"业务相关: {business_related}, 信息完整: {information_complete}"
        }
