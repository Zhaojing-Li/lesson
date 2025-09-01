#!/usr/bin/env python3
"""
LangGraph中断机制详细演示
展示interrupt()和Command(resume=...)的完整用法
"""

import uuid
from typing import TypedDict, List, Dict, Any
from langgraph.graph import StateGraph, START, END
from langgraph.types import interrupt, Command
from langgraph.checkpoint.memory import MemorySaver


class DemoState(TypedDict):
    """演示状态"""
    user_input: str
    current_step: str
    confirmed_data: Dict[str, Any]
    history: List[str]


def step1_initial_check(state: DemoState) -> DemoState:
    """步骤1：初始检查"""
    print(f"\n=== 步骤1：初始检查 ===")
    print(f"用户输入：{state['user_input']}")
    
    # 检查输入是否包含必要信息
    if "手机号" not in state['user_input']:
        print("🛑 触发中断：缺少手机号信息")
        interrupt({
            "type": "missing_phone",
            "message": "您的输入中缺少手机号信息",
            "required_info": "请提供被调查人的手机号（11位数字）",
            "example": "例如：给被调查人13812345678生成隐私号"
        })
    
    # 如果包含手机号，继续执行
    state["current_step"] = "step1_completed"
    state["history"].append("步骤1完成：输入检查通过")
    return state


def step2_phone_confirmation(state: DemoState) -> DemoState:
    """步骤2：手机号确认"""
    print(f"\n=== 步骤2：手机号确认 ===")
    
    # 提取手机号（这里简化处理）
    import re
    phone_pattern = r'1[3-9]\d{9}'
    matches = re.findall(phone_pattern, state['user_input'])
    
    if matches:
        phone = matches[0]
        print(f"检测到手机号：{phone}")
        
        # 检查是否明确说明是被调查人的号码
        if "被调查人" in state['user_input']:
            print("✅ 手机号用途明确，继续执行")
            state["confirmed_data"]["phone_number"] = phone
        else:
            print("🛑 触发中断：需要确认手机号用途")
            interrupt({
                "type": "phone_confirmation",
                "message": f"检测到手机号：{phone}",
                "question": "请确认这是被调查人员的手机号吗？",
                "phone_number": phone,
                "options": ["是", "否"],
                "instructions": "输入 '是' 确认，输入 '否' 重新输入"
            })
    else:
        print("🛑 触发中断：未找到有效手机号")
        interrupt({
            "type": "invalid_phone",
            "message": "未在输入中找到有效的手机号",
            "format": "手机号格式：1[3-9]xxxxxxxxx（11位数字）",
            "example": "例如：13812345678"
        })
    
    state["current_step"] = "step2_completed"
    state["history"].append("步骤2完成：手机号确认")
    return state


def step3_type_selection(state: DemoState) -> DemoState:
    """步骤3：类型选择"""
    print(f"\n=== 步骤3：类型选择 ===")
    
    # 检查是否指定了隐私号类型
    if "可回拨" in state['user_input']:
        privacy_type = "可回拨"
        print(f"✅ 用户指定了类型：{privacy_type}")
    elif "不可回拨" in state['user_input']:
        privacy_type = "不可回拨"
        print(f"✅ 用户指定了类型：{privacy_type}")
    else:
        print("🛑 触发中断：需要选择隐私号类型")
        interrupt({
            "type": "type_selection",
            "message": "请选择隐私号类型：",
            "choices": [
                {
                    "id": "1",
                    "name": "可回拨",
                    "description": "被调查人可以回拨隐私号，平台会路由给调查员",
                    "scenario": "适合需要双向通信的调查场景"
                },
                {
                    "id": "2", 
                    "name": "不可回拨",
                    "description": "被调查人无法回拨隐私号，只能单向拨打",
                    "scenario": "适合单向联系的调查场景"
                }
            ],
            "instructions": "请输入 '1' 选择可回拨，或输入 '2' 选择不可回拨"
        })
        return state
    
    state["confirmed_data"]["privacy_type"] = privacy_type
    state["current_step"] = "step3_completed"
    state["history"].append(f"步骤3完成：选择类型 {privacy_type}")
    return state


def step4_generation(state: DemoState) -> DemoState:
    """步骤4：生成隐私号"""
    print(f"\n=== 步骤4：生成隐私号 ===")
    
    phone = state["confirmed_data"]["phone_number"]
    privacy_type = state["confirmed_data"]["privacy_type"]
    
    # 生成隐私号
    if privacy_type == "可回拨":
        privacy_number = f"400{phone[3:7]}{phone[7:]}"
    else:
        privacy_number = f"300{phone[3:7]}{phone[7:]}"
    
    state["confirmed_data"]["privacy_number"] = privacy_number
    state["current_step"] = "completed"
    state["history"].append(f"步骤4完成：生成隐私号 {privacy_number}")
    
    print(f"✅ 成功生成隐私号：{privacy_number}")
    return state


def build_demo_graph():
    """构建演示工作流图"""
    workflow = StateGraph(DemoState)
    
    # 添加节点
    workflow.add_node("step1", step1_initial_check)
    workflow.add_node("step2", step2_phone_confirmation)
    workflow.add_node("step3", step3_type_selection)
    workflow.add_node("step4", step4_generation)
    
    # 构建线性流程
    workflow.add_edge(START, "step1")
    workflow.add_edge("step1", "step2")
    workflow.add_edge("step2", "step3")
    workflow.add_edge("step3", "step4")
    workflow.add_edge("step4", END)
    
    # 编译工作流
    checkpointer = MemorySaver()
    return workflow.compile(checkpointer=checkpointer)


def run_demo_with_interrupts():
    """运行演示，展示中断和恢复的完整流程"""
    print("🚀 LangGraph中断机制演示")
    print("=" * 60)
    
    # 构建工作流
    graph = build_demo_graph()
    
    # 测试用例
    test_cases = [
        {
            "name": "缺少手机号",
            "input": "我需要生成隐私号",
            "expected_interrupts": 1
        },
        {
            "name": "缺少类型选择", 
            "input": "给被调查人13812345678生成隐私号",
            "expected_interrupts": 1
        },
        {
            "name": "完整输入",
            "input": "给被调查人13812345678生成可回拨隐私号",
            "expected_interrupts": 0
        }
    ]
    
    for i, test_case in enumerate(test_cases, 1):
        print(f"\n{'='*60}")
        print(f"测试用例 {i}: {test_case['name']}")
        print(f"输入: {test_case['input']}")
        print(f"{'='*60}")
        
        # 初始化状态
        initial_state = DemoState(
            user_input=test_case['input'],
            current_step="started",
            confirmed_data={},
            history=[]
        )
        
        # 生成唯一线程ID
        thread_id = str(uuid.uuid4())
        config = {"configurable": {"thread_id": thread_id}}
        
        try:
            # 执行工作流
            result = execute_workflow_with_resume(graph, initial_state, config)
            print(f"✅ 测试用例 {i} 完成")
            print(f"最终结果: {result}")
            
        except Exception as e:
            print(f"❌ 测试用例 {i} 失败: {e}")
        
        print(f"\n{'='*60}")


def execute_workflow_with_resume(graph, initial_state, config):
    """执行工作流并处理中断恢复"""
    try:
        # 运行工作流
        result = graph.invoke(initial_state, config=config)
        
        # 检查是否有中断
        if "__interrupt__" in result:
            return handle_interrupt(graph, result, config)
        
        # 工作流完成
        return format_result(result)
        
    except Exception as e:
        print(f"❌ 工作流执行失败: {e}")
        return {"error": str(e), "status": "failed"}


def handle_interrupt(graph, result, config):
    """处理中断"""
    payload = result["__interrupt__"]
    if isinstance(payload, list) and payload:
        payload = payload[0]
    
    # 正确处理Interrupt对象
    if hasattr(payload, 'value'):
        interrupt_data = payload.value
    else:
        interrupt_data = payload
    
    print(f"\n⏸️  工作流中断")
    print(f"中断类型: {interrupt_data.get('type', 'unknown')}")
    print(f"消息: {interrupt_data.get('message', '需要用户确认')}")
    
    # 根据中断类型提供不同的处理
    user_response = handle_interrupt_by_type(interrupt_data)
    
    print(f"用户回复: {user_response}")
    
    # 恢复工作流
    return resume_workflow(graph, user_response, config)


def handle_interrupt_by_type(interrupt_data):
    """根据中断类型处理用户输入"""
    interrupt_type = interrupt_data.get('type', 'unknown')
    
    if interrupt_type == "missing_phone":
        print("📝 提示: 请提供手机号信息")
        return input("请输入包含手机号的完整请求: ").strip()
    
    elif interrupt_type == "phone_confirmation":
        phone = interrupt_data.get('phone_number', '')
        print(f"📝 提示: 确认手机号 {phone} 是否为被调查人的")
        return input("请输入 '是' 或 '否': ").strip()
    
    elif interrupt_type == "invalid_phone":
        print("📝 提示: 请提供有效的手机号")
        return input("请输入有效的手机号: ").strip()
    
    elif interrupt_type == "type_selection":
        print("📝 提示: 请选择隐私号类型")
        choices = interrupt_data.get('choices', [])
        for choice in choices:
            print(f"  {choice['id']}. {choice['name']} - {choice['description']}")
        return input("请输入选择 (1 或 2): ").strip()
    
    else:
        print("📝 提示: 请提供必要信息")
        return input("请输入您的回复: ").strip()


def resume_workflow(graph, user_response, config):
    """恢复工作流"""
    print(f"▶️  恢复工作流")
    
    try:
        # 使用Command恢复工作流
        result = graph.invoke(Command(resume=user_response), config=config)
        
        # 检查是否还有新的中断
        if "__interrupt__" in result:
            print("🔄 检测到新的中断，递归处理...")
            return handle_interrupt(graph, result, config)
        
        # 工作流完成
        return format_result(result)
        
    except Exception as e:
        print(f"❌ 工作流恢复失败: {e}")
        return {"error": str(e), "status": "failed"}


def format_result(result):
    """格式化最终结果"""
    return {
        "status": "completed",
        "current_step": result.get("current_step"),
        "confirmed_data": result.get("confirmed_data", {}),
        "history": result.get("history", [])
    }


if __name__ == "__main__":
    run_demo_with_interrupts()
