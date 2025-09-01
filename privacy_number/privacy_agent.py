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
import config


class PrivacyNumberState(TypedDict):
    """隐私号处理状态定义 - 优化版"""
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
                "enable_thinking": True,
                "thinking_budget": 500
            }
        )
        print(f"获取到qwen的配置:{cfg.llm_config.api_key}")
        
        # 创建内存检查点
        self.memory = MemorySaver()
        # 构建工作流图
        self.graph = self._build_graph()
        
    
    def _build_graph(self):
        """构建简化的LangGraph工作流"""
        workflow = StateGraph(PrivacyNumberState)
        
        # 添加节点
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
        
        return workflow.compile(checkpointer=self.memory)
    

    async def _analyze_input(self, state: PrivacyNumberState, config: RunnableConfig = None) -> PrivacyNumberState:
        """分析用户输入，进行语义理解与推理"""
        user_input = state["user_input"]
        
        # 构建简化的分析提示
        analysis_prompt = f"""
        你是隐私号平台的AI助手.具备精准的语义理解和业务判断能力。请分析用户请求并给出明确的下一步动作。

        隐私号业务说明：
        - 调查员输入被调查人员手机号，平台生成中转路由号码
        - 调查员拨打隐私号，平台路由给被调查人员
        - 被调查人员看到隐私号，而非调查员真实号码
        - 隐私号类型：可回拨（可回拨给调查员）、不可回拨（无法回拨）

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
        """
        
        # 使用流式输出获取模型响应
        thinking_content = []
        final_content = []
        
        async for chunk in self.chat_model.astream([
            HumanMessage(content=analysis_prompt)
        ]):
            # 解析流式响应，区分thinking和content
            chunk_data = self._parse_stream_chunk(chunk)
            if chunk_data:
                # 处理thinking内容
                if chunk_data.get("thinking"):
                    thinking_content.append(chunk_data['thinking'])
                
                # 处理content内容
                if chunk_data.get("content"):
                    final_content.append(chunk_data["content"])
                    
        
        # 合并最终结果
        full_response = "".join(final_content)
        
        # 记录推理过程
        # if thinking_content:
        #     thinking_summary = "".join(thinking_content)
        #     state["reasoning_process"].append(f"输入分析阶段思考过程：{thinking_summary}")
        
        # if final_content:
        #     state["reasoning_process"].append(f"输入分析阶段最终结果：{full_response}")
        
        # 将流式内容进行汇总后仅推送少量记录，避免前端出现大量一行一条
        # 思考过程（合并）
        if thinking_content:
            state["stream_data"].append({
                "timestamp": datetime.now().isoformat(),
                "stage": "思考分析",
                "type": "thinking",
                "content": "".join(thinking_content)
            })
        # 最终结果（合并）
        if full_response:
            state["stream_data"].append({
                "timestamp": datetime.now().isoformat(),
                "stage": "输出结果",
                "type": "content",
                "content": full_response
            })
        
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
            # JSON解析失败，使用后备分析
            raise ValueError("未找到有效的JSON")
        

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
    

    async def _handle_confirmation(self, state: PrivacyNumberState, config: RunnableConfig = None) -> PrivacyNumberState:
        """统一处理各种确认场景"""
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
        await self._process_confirmation_response(state, user_response)

        # 标记
        state["step"] = "confirmation_processed"
        state["reasoning_process"].append(f"确认处理：{message}")

        return state
    
    
    async def _generate_privacy_number(self, state: PrivacyNumberState, config: RunnableConfig = None) -> PrivacyNumberState:
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
    
    async def _present_result(self, state: PrivacyNumberState, config: RunnableConfig = None) -> PrivacyNumberState:
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
        
        state["step"] = "completed"
        return state
    
    # 条件判断函数
    def _decide_next_step(self, state: PrivacyNumberState) -> str:
        """决定分析后的下一步"""
        
        if state.get("need_reinput") or state.get("need_human_confirmation"):
            return "handle_confirmation"
        # 信息完整，直接生成
        return "generate_privacy_number"
    
    def _decide_after_confirmation(self, state: PrivacyNumberState) -> str:
        """确认后的路由：若需要重新输入则停止；否则回到再分析（ReAct循环）。"""
        if state.get("need_reinput"):
            return "wait"
        # 确认结束后回到输入分析，让模型基于新信息再次判断下一步
        return "re_analyze"
    
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
        
        # 执行工作流（支持中断捕获）
        async for event in self.graph.astream(initial_state, config=graph_config, stream_mode="values"):
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
    
    async def resume_with_response(self, user_response: str, thread_id: str = "default") -> Dict[str, Any]:
        """恢复被中断的对话，传入用户响应"""
        from langgraph.types import Command
        
        graph_config = {"configurable": {"thread_id": thread_id}}
        
        # 使用Command.resume恢复执行工作流
        async for event in self.graph.astream(Command(resume=user_response), config=graph_config, stream_mode="values"):
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
    
    async def _process_confirmation_response(self, state: PrivacyNumberState, response: str):
        """处理用户确认回复"""
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
                # 但不清空原有信息，等待新输入
                return
        
        # 处理手机号确认（基于类型而非提示文案）
        if pending_type == "phone_confirm":
            if "是" in response or "yes" in response_lower or "确认" in response:
                state["need_human_confirmation"] = False
                state["pending_confirmation_type"] = None
            else:
                # 用户否认，需要重新输入手机号
                state["phone_number"] = None
                state["need_human_confirmation"] = True
                state["need_reinput"] = True
                state["reinput_reason"] = "用户否认手机号，需重新输入"
                state["confirmation_message"] = "需要重新输入手机号"
                state["pending_confirmation_type"] = "reinput"
        
        # 处理隐私号类型确认
        elif pending_type == "type_confirm":
            if "1" in response or "可回拨" in response:
                state["privacy_type"] = "可回拨"
                state["need_human_confirmation"] = False
                state["pending_confirmation_type"] = None
            elif "2" in response or "不可回拨" in response:
                state["privacy_type"] = "不可回拨"
                state["need_human_confirmation"] = False
                state["pending_confirmation_type"] = None
    


    def _parse_stream_chunk(self, chunk) -> dict:
        """解析流式响应块,区分thinking和content"""
        try:
            result = {}
            reasoning_content = chunk.additional_kwargs.get('reasoning_content', '')
            content = chunk.content 
            
            if reasoning_content: 
                result["thinking"] = reasoning_content
            if content:
                result["content"] = content
            return result
                    
        except Exception as e:
            print(f"解析流式响应块错误: {e}")
            return None
    

   
