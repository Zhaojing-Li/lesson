#!/usr/bin/env python3
"""
隐私号处理Agent - 同步版本
解决 "Called get_config outside of a runnable context" 错误
"""
import os
from pathlib import Path
import sys
import re
import json
from typing import Dict, Any, List, Optional, Literal, TypedDict, Annotated
from datetime import datetime
from pydantic import BaseModel, SecretStr
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from langchain_core.runnables import RunnableConfig
from langchain_community.chat_models.tongyi import ChatTongyi
from langgraph.graph import StateGraph, START, END
from langgraph.types import interrupt, Command
from langgraph.checkpoint.memory import MemorySaver
from langgraph.prebuilt import create_react_agent
from langgraph.graph.message import add_messages
from typing import Union
from privacy_number import config
from langfuse.langchain import CallbackHandler
from langfuse import Langfuse, get_client


class PrivacyNumberState(TypedDict):
    """隐私号处理状态定义 - 同步版"""
    user_input: str
    phone_number: Optional[str]
    privacy_type: Optional[Literal["可回拨", "不可回拨"]]
    privacy_number: Optional[str]
    need_human_confirmation: bool
    confirmation_message: Optional[str]
    step: str
    # 核心业务字段
    need_reinput: bool
    reinput_reason: Optional[str]
    reasoning_process: List[str]
    stream_data: List[dict]
    # 等待确认的类型：phone_confirm | type_confirm | reinput
    pending_confirmation_type: Optional[str]


class SyncPrivacyNumberAgent:
    """隐私号处理Agent - 同步版本"""
    
    def __init__(self):
        """初始化Agent"""
        # 创建配置实例
        cfg = config.Config()
        
        # 初始化 Langfuse（从 .env 文件与 .env 目录加载）
        self._init_langfuse()
        # 回调处理器
        self.handler = CallbackHandler()

        # 同步版本的模型配置
        self.chat_model = ChatTongyi(
            model=cfg.llm_config.model,  
            api_key=SecretStr(cfg.llm_config.api_key),
            top_p=0.8,         
            streaming=False,   # 改为非流式输出
            model_kwargs={
                "temperature": 0.5,      
                "enable_thinking": True,
                "thinking_budget": 500
            }
        )
        
        # 创建内存检查点
        self.memory = MemorySaver()
        # 构建工作流图
        self.graph = self._build_graph()
        
    
    def _build_graph(self):
        """构建同步的LangGraph工作流"""
        workflow = StateGraph(PrivacyNumberState)
        
        # 添加节点 - 使用同步方法
        workflow.add_node("input_analysis", self._analyze_input)
        workflow.add_node("handle_confirmation", self._handle_confirmation)
        workflow.add_node("generate_privacy_number", self._generate_privacy_number)
        workflow.add_node("result_presentation", self._present_result)
        
        # 定义流程边
        workflow.add_edge(START, "input_analysis")
        
        workflow.add_conditional_edges(
            "input_analysis",
            self._decide_next_step,
            {
                "handle_confirmation": "handle_confirmation",
                "generate_privacy_number": "generate_privacy_number"
            }
        )
        
        workflow.add_conditional_edges(
            "handle_confirmation",
            self._decide_after_confirmation,
            {
                "re_analyze": "input_analysis",
                "wait": END
            }
        )
        
        workflow.add_edge("generate_privacy_number", "result_presentation")
        workflow.add_edge("result_presentation", END)
        
        end_graph = workflow.compile(checkpointer=self.memory)

        # self.image_display(end_graph)
        return end_graph
    

    def _analyze_input(self, state: PrivacyNumberState, config: RunnableConfig = None) -> PrivacyNumberState:
        """分析用户输入，进行语义理解与推理 - 同步版本"""
        user_input = state["user_input"]
        
        # 构建分析提示
        analysis_prompt = f"""
        你是隐私号平台的AI助手，具备精准的语义理解和业务判断能力。请分析用户请求并给出明确的下一步动作。

        隐私号业务说明：
        - 调查员输入被调查人员手机号，平台生成中转路由号码
        - 调查员拨打隐私号，平台路由给被调查人员
        - 被调查人员看到隐私号，而非调查员真实号码
        - 隐私号类型：可回拨（可回拨给调查员）、不可回拨（无法回拨）

        对用户输入的判断顺序：
        1. 判断是否是业务无关（如问候、知识问答等）
        2. 判断是否是被调查人的手机号
        3. 判断是否明确了隐私号类型（可回拨，不可回拨）
        4. 判断是否是信息完整，可直接生成隐私号

        提示词：
        用户输入：{user_input}

        请分析并返回JSON格式结果：
        {{
            "action": "reinput|phone_confirm|type_confirm|generate",
            "reason": "执行该动作的原因",
            "extracted_phone": "提取的手机号或null",
            "extracted_type": "提取的类型或null",
            "message": "给用户的提示信息"
        }}

        判断规则：
        1. reinput：业务无关（如问候、知识问答等）
        2. phone_confirm：有手机号但未明确说明是被调查人员号码
        3. type_confirm：有手机号但未明确隐私号类型
        4. generate：信息完整，可直接生成隐私号
        
        特别注意：如果输入中包含"隐私号类型：可回拨"或"隐私号类型：不可回拨"，说明用户已经确认了类型，应该直接进入generate阶段。
        """
        
        # 使用同步调用模型
        try:
            response = self.chat_model.invoke([HumanMessage(content=analysis_prompt)], config={"callbacks": [self.handler]})
            
            # 处理响应内容
            full_response = response.content
            
            # 记录处理过程
            state["reasoning_process"].append(f"输入分析完成：{user_input}")
            
            # 如果有thinking内容，也记录
            if hasattr(response, 'additional_kwargs') and 'reasoning_content' in response.additional_kwargs:
                thinking_content = response.additional_kwargs['reasoning_content']
                if thinking_content:
                    state["stream_data"].append({
                        "timestamp": datetime.now().isoformat(),
                        "stage": "思考分析",
                        "type": "thinking",
                        "content": thinking_content
                    })
            
            # 记录最终结果
            state["stream_data"].append({
                "timestamp": datetime.now().isoformat(),
                "stage": "输出结果",
                "type": "content",
                "content": full_response
            })
            
        except Exception as e:
            print(f"模型调用错误: {e}")
            # 使用简单的正则表达式作为后备方案
            return self._fallback_analysis(state)
        
        # 解析JSON结果
        try:
            json_start = full_response.find("{")
            json_end = full_response.rfind("}") + 1
            if json_start != -1 and json_end > json_start:
                json_str = full_response[json_start:json_end]
                analysis_result = json.loads(json_str)
            else:
                raise ValueError("未找到有效的JSON")
        except (json.JSONDecodeError, ValueError) as e:
            print(f"JSON解析失败: {e}")
            return self._fallback_analysis(state)

        # 根据分析结果更新状态
        action = analysis_result.get("action", "reinput")
        reason = analysis_result.get("reason", "")
        message = analysis_result.get("message", "")
        
        # 提取手机号和类型
        extracted_phone = analysis_result.get("extracted_phone")
        if extracted_phone and extracted_phone != "null":
            state["phone_number"] = extracted_phone
        
        extracted_type = analysis_result.get("extracted_type")
        if extracted_type and extracted_type in ["可回拨", "不可回拨"]:
            state["privacy_type"] = extracted_type
        
        # 设置下一步动作
        if action == "reinput":
            state["need_reinput"] = True
            state["reinput_reason"] = reason
            state["need_human_confirmation"] = True
            state["confirmation_message"] = message
        elif action == "phone_confirm":
            state["need_human_confirmation"] = True
            state["confirmation_message"] = message
            state["pending_confirmation_type"] = "phone_confirm"
        elif action == "type_confirm":
            state["need_human_confirmation"] = True
            state["confirmation_message"] = message
            state["pending_confirmation_type"] = "type_confirm"
        elif action == "generate":
            # 信息完整，可以直接生成
            state["need_human_confirmation"] = False
            state["pending_confirmation_type"] = None
        
        state["step"] = "input_analysis_completed"
        return state
    

    def _fallback_analysis(self, state: PrivacyNumberState) -> PrivacyNumberState:
        """后备分析方案 - 使用正则表达式"""
        user_input = state["user_input"]
        
        # 提取手机号
        phone_pattern = r'1[3-9]\d{9}'
        phones = re.findall(phone_pattern, user_input)
        
        # 检查隐私号类型
        privacy_type = None
        if "隐私号类型：可回拨" in user_input or "可回拨" in user_input:
            privacy_type = "可回拨"
        elif "隐私号类型：不可回拨" in user_input or "不可回拨" in user_input:
            privacy_type = "不可回拨"
        
        if not phones:
            state["need_reinput"] = True
            state["reinput_reason"] = "未检测到有效手机号"
            state["need_human_confirmation"] = True
            state["confirmation_message"] = "请提供被调查人员的手机号码"
            state["pending_confirmation_type"] = "reinput"
        elif not privacy_type:
            state["phone_number"] = phones[0]
            state["need_human_confirmation"] = True
            state["confirmation_message"] = "请选择隐私号类型：可回拨或不可回拨"
            state["pending_confirmation_type"] = "type_confirm"
        else:
            state["phone_number"] = phones[0]
            state["privacy_type"] = privacy_type
            state["need_human_confirmation"] = False
            state["pending_confirmation_type"] = None
        
        state["reasoning_process"].append(f"后备分析完成：手机号={phones}, 类型={privacy_type}")
        state["step"] = "input_analysis_completed"
        return state


    def _handle_confirmation(self, state: PrivacyNumberState, config: RunnableConfig = None) -> PrivacyNumberState:
        """统一处理各种确认场景 - 同步版本"""
        # 组织中断负载，告知前端需要何种确认
        confirm_type = state.get("pending_confirmation_type")
        # 尝试推断
        if not confirm_type:
            if not state.get("phone_number"):
                confirm_type = "phone_confirm"
            elif not state.get("privacy_type"):
                confirm_type = "type_confirm"
            else:
                confirm_type = "need_reinput"

        message = state.get("confirmation_message") or (
            "请选择隐私号类型：可回拨或不可回拨" if confirm_type == "type_confirm" else "请确认这是被调查人员的手机号码？"
        )

        payload = {"type": confirm_type, "message": message}

        # 触发中断，等待前端恢复（这里会暂停执行，等待 resume）
        user_response = interrupt(payload)

        # 当从 resume 恢复时，user_response 包含用户的回复
        # 处理用户响应
        self._process_confirmation_response(state, user_response)

        # 标记
        state["step"] = "confirmation_processed"
        state["reasoning_process"].append(f"确认处理：{message}")

        return state
    
    
    def _generate_privacy_number(self, state: PrivacyNumberState, config: RunnableConfig = None) -> PrivacyNumberState:
        """生成隐私号 - 同步版本"""
        phone = state["phone_number"]
        privacy_type = state["privacy_type"]
        
        # 检查手机号是否为空
        if not phone:
            raise ValueError("手机号不能为空")
            
        # 生成隐私号逻辑
        if privacy_type == "可回拨":
            privacy_number = self._generate_callable_privacy_number(phone)
        else:
            privacy_number = self._generate_non_callable_privacy_number(phone)
            
        state["privacy_number"] = privacy_number
        state["step"] = "privacy_number_generated"
        
        reasoning = f"已为手机号 {phone} 生成 {privacy_type} 隐私号：{privacy_number}"
        state["reasoning_process"].append(reasoning)
        
        return state
    

    def _present_result(self, state: PrivacyNumberState, config: RunnableConfig = None) -> PrivacyNumberState:
        """展示最终结果 - 同步版本"""
        state["step"] = "completed"
        return state
    

    # 条件判断函数
    def _decide_next_step(self, state: PrivacyNumberState) -> str:
        """决定分析后的下一步"""
        if state.get("need_reinput") or state.get("need_human_confirmation"):
            return "handle_confirmation"
        return "generate_privacy_number"
    

    def _decide_after_confirmation(self, state: PrivacyNumberState) -> str:
        """确认后的路由"""
        if state.get("need_reinput"):
            return "wait"
        
        # 如果用户刚刚确认了隐私号类型，需要重新分析以规划下一步
        if state.get("pending_confirmation_type") == "type_confirm" and state.get("privacy_type"):
            # 重新分析以确定下一步动作
            return "re_analyze"
        
        # 检查是否所有信息都已完整（手机号和隐私号类型都有了）
        if state.get("phone_number") and state.get("privacy_type"):
            # 信息完整，直接生成隐私号
            return "generate_privacy_number"
        
        # 其他情况，重新分析
        return "re_analyze"
    

    # 隐私号生成工具方法
    def _generate_callable_privacy_number(self, phone: str) -> str:
        """生成可回拨隐私号"""
        base = phone[3:7] 
        suffix = phone[7:]
        privacy_num = f"400{base}{suffix}"
        return privacy_num
    

    def _generate_non_callable_privacy_number(self, phone: str) -> str:
        """生成不可回拨隐私号"""
        base = phone[3:7]
        suffix = phone[7:] 
        privacy_num = f"300{base}{suffix}"
        return privacy_num
    

    def process_request(self, user_input: str, thread_id: str = "default") -> Dict[str, Any]:
        """处理用户请求的主入口 - 同步版本"""
        initial_state = PrivacyNumberState(
            user_input=user_input,
            phone_number=None,
            privacy_type=None,
            privacy_number=None,
            need_human_confirmation=False,
            confirmation_message=None,
            step="started",
            need_reinput=False,
            reinput_reason=None,
            reasoning_process=[],
            stream_data=[],
            pending_confirmation_type=None
        )
        
        graph_config = {"configurable": {"thread_id": thread_id}}
        
        # 使用同步的 stream 方法
        for event in self.graph.stream(initial_state, config=graph_config, stream_mode="values"):
            # 捕获中断，返回给前端
            if "__interrupt__" in event:
                payload = event["__interrupt__"]
                if isinstance(payload, list) and payload:
                    payload = payload[0]
                return {
                    **initial_state,
                    "need_human_confirmation": True,
                    "confirmation_message": payload.get("message"),
                    "pending_confirmation_type": payload.get("type"),
                    "step": "awaiting_confirmation"
                }
            last = event
        return last
    

    def resume_with_response(self, user_response: str, thread_id: str = "default") -> Dict[str, Any]:
        """恢复被中断的对话 - 同步版本"""
        from langgraph.types import Command
        
        graph_config = {"configurable": {"thread_id": thread_id}}
        
        # 使用同步的 stream 方法
        for event in self.graph.stream(Command(resume=user_response), config=graph_config, stream_mode="values"):
            # 捕获新的中断
            if "__interrupt__" in event:
                payload = event["__interrupt__"]
                if isinstance(payload, list) and payload:
                    payload = payload[0]
                return {
                    "need_human_confirmation": True,
                    "confirmation_message": payload.get("message"),
                    "pending_confirmation_type": payload.get("type"),
                    "step": "awaiting_confirmation"
                }
            last = event
        return last
    

    def _process_confirmation_response(self, state: PrivacyNumberState, response: str):
        """处理用户确认回复 - 同步版本"""
        response_lower = response.lower()
        pending_type = state.get("pending_confirmation_type") or ""

        # 处理重新输入确认
        if state.get("need_reinput") or pending_type == "reinput":
            if "重新" in response or "确认" in response or "ok" in response_lower or "好" in response:
                # 用户确认重新输入，重置状态
                state["need_reinput"] = False
                state["reinput_reason"] = None
                state["need_human_confirmation"] = False
                state["confirmation_message"] = None
                state["pending_confirmation_type"] = None
                return
        
        # 处理手机号确认
        if pending_type == "phone_confirm":
            original_input = state.get("user_input", "")
            
            if "是" in response or "yes" in response_lower or "确认" in response:
                state["need_human_confirmation"] = False
                state["pending_confirmation_type"] = None
                enhanced_input = f"{original_input}， 已确认是被调查人员的手机号；"

            else:
                # 用户否认，需要重新输入手机号
                state["phone_number"] = None
                state["need_human_confirmation"] = True
                state["need_reinput"] = True
                state["reinput_reason"] = "用户否认手机号，需重新输入"
                state["confirmation_message"] = "需要重新输入手机号"
                state["pending_confirmation_type"] = "reinput"
        
        # 处理隐私号类型确认 - 按照ReAct模式，将确认信息作为补充重新进入分析
        elif pending_type == "type_confirm":
            # 将用户的类型选择作为补充信息，重新构建输入进行分析
            original_input = state.get("user_input", "")
            phone_number = state.get("phone_number", "")
            
            # 根据用户响应确定类型
            if "1" in response or "可回拨" in response or "选择可回拨类型" in response:
                privacy_type = "可回拨"
            elif "2" in response or "不可回拨" in response or "选择不可回拨类型" in response:
                privacy_type = "不可回拨"
            else:
                # 如果用户输入不明确，保持当前状态等待进一步确认
                return
            
            # 构建包含类型信息的完整输入
            enhanced_input = f"{original_input} 隐私号类型：{privacy_type}"
            
            # 更新状态中的输入，准备重新分析
            state["user_input"] = enhanced_input
            state["phone_number"] = phone_number  # 保持已确认的手机号
            state["privacy_type"] = privacy_type  # 设置用户选择的类型
            state["need_human_confirmation"] = False
            # 保持pending_confirmation_type为"type_confirm"，让_decide_after_confirmation知道需要重新分析
            state["pending_confirmation_type"] = "type_confirm"
            
            # 记录处理过程
            state["reasoning_process"].append(f"用户确认类型：{privacy_type}")
            state["stream_data"].append({
                "timestamp": datetime.now().isoformat(),
                "stage": "类型确认",
                "type": "content",
                "content": f"用户选择隐私号类型：{privacy_type}"
            })

    def _init_langfuse(self) -> None:
        """初始化 Langfuse 客户端，兼容 .env 目录与环境变量。"""
        try:
            
            Langfuse(
                public_key=os.getenv("LANGFUSE_PUBLIC_KEY"),
                secret_key=os.getenv("LANGFUSE_SECRET_KEY"),
                host=os.getenv("LANGFUSE_HOST") or os.getenv("LANGFUSE_BASE_URL")
            )
        except Exception as e:
            print(f"[Langfuse] 初始化失败: {e}")
    
    

    def image_display(self, graph):
        # 终端环境下：将流程图渲染为 PNG 文件并尝试打开
        try:
            png_bytes = graph.get_graph().draw_mermaid_png()
            out_path = os.path.join(os.path.dirname(__file__), "graph.png")
            with open(out_path, "wb") as f:
                f.write(png_bytes)
            # macOS 自动打开
            if sys.platform == "darwin":
                os.system(f'open "{out_path}"')
            print(f"流程图已保存：{out_path}")
        except Exception as e:
            print(f"渲染流程图失败：{e}")
            print("你可以改用 graph.get_graph().draw_mermaid() 打印 Mermaid 文本在终端查看。")