#!/usr/bin/env python3
"""
LangGraph Resume机制详细演示
展示用户输入后如何从中断点继续执行
"""

import uuid
from typing import TypedDict, Dict, Any
from langgraph.graph import StateGraph, START, END
from langgraph.types import interrupt, Command
from langgraph.checkpoint.memory import MemorySaver


class ResumeDemoState(TypedDict):
    """演示状态"""
    user_input: str
    step: str
    execution_count: int
    user_responses: list
    current_data: Dict[str, Any]


def step_with_interrupt(state: ResumeDemoState) -> ResumeDemoState:
    """包含中断的步骤，演示resume机制"""
    print(f"\n=== 步骤执行 (第{state['execution_count']}次) ===")
    print(f"当前状态: {state}")
    
    # 增加执行计数
    state['execution_count'] += 1
    
    # 检查是否已经获取了用户输入
    if 'user_responses' not in state:
        state['user_responses'] = []
    
    if 'current_data' not in state:
        state['current_data'] = {}
    
    # 模拟需要用户确认的场景
    if state['step'] == "started":
        print("🛑 第一次中断：需要用户提供手机号")
        state['step'] = "waiting_phone"
        
        # 触发中断
        user_phone = interrupt({
            "type": "phone_input",
            "message": "请提供被调查人的手机号",
            "format": "11位数字，如：13812345678"
        })
        
        # 注意：当resume后，代码会从这里继续执行
        # user_phone 就是用户输入的值
        print(f"✅ 获取到用户输入: {user_phone}")
        
        # 处理用户输入
        state['current_data']['phone_number'] = user_phone
        state['user_responses'].append(f"手机号: {user_phone}")
        state['step'] = "phone_received"
        
        print(f"📝 状态更新: {state}")
    
    elif state['step'] == "phone_received":
        print("🛑 第二次中断：需要用户确认手机号")
        state['step'] = "waiting_confirmation"
        
        phone = state['current_data']['phone_number']
        confirmation = interrupt({
            "type": "phone_confirmation",
            "message": f"请确认手机号 {phone} 是否为被调查人的",
            "phone_number": phone,
            "options": ["是", "否"]
        })
        
        # resume后从这里继续执行
        print(f"✅ 获取到用户确认: {confirmation}")
        
        if confirmation == "是":
            state['current_data']['phone_confirmed'] = True
            state['user_responses'].append("手机号确认: 是")
            state['step'] = "phone_confirmed"
            print("✅ 手机号确认成功")
        else:
            state['current_data']['phone_confirmed'] = False
            state['user_responses'].append("手机号确认: 否")
            state['step'] = "phone_rejected"
            print("❌ 手机号被拒绝")
    
    elif state['step'] == "phone_confirmed":
        print("🛑 第三次中断：需要用户选择隐私号类型")
        state['step'] = "waiting_type"
        
        type_choice = interrupt({
            "type": "type_selection",
            "message": "请选择隐私号类型",
            "choices": [
                {"id": "1", "name": "可回拨", "description": "被调查人可以回拨"},
                {"id": "2", "name": "不可回拨", "description": "只能单向拨打"}
            ]
        })
        
        # resume后从这里继续执行
        print(f"✅ 获取到用户选择: {type_choice}")
        
        if type_choice == "1":
            privacy_type = "可回拨"
        elif type_choice == "2":
            privacy_type = "不可回拨"
        else:
            privacy_type = "未知"
        
        state['current_data']['privacy_type'] = privacy_type
        state['user_responses'].append(f"类型选择: {privacy_type}")
        state['step'] = "type_selected"
        
        print(f"✅ 隐私号类型选择完成: {privacy_type}")
    
    elif state['step'] == "type_selected":
        print("🎯 所有信息收集完成，生成隐私号")
        
        phone = state['current_data']['phone_number']
        privacy_type = state['current_data']['privacy_type']
        
        # 生成隐私号
        if privacy_type == "可回拨":
            privacy_number = f"400{phone[3:7]}{phone[7:]}"
        else:
            privacy_number = f"300{phone[3:7]}{phone[7:]}"
        
        state['current_data']['privacy_number'] = privacy_number
        state['step'] = "completed"
        
        print(f"🎉 成功生成隐私号: {privacy_number}")
    
    return state


def build_resume_demo_graph():
    """构建演示工作流图"""
    workflow = StateGraph(ResumeDemoState)
    workflow.add_node("step", step_with_interrupt)
    workflow.add_edge(START, "step")
    workflow.add_edge("step", END)
    
    checkpointer = MemorySaver()
    return workflow.compile(checkpointer=checkpointer)


def demonstrate_resume_mechanism():
    """演示resume机制的工作原理"""
    print("🔄 LangGraph Resume机制演示")
    print("=" * 80)
    
    graph = build_resume_demo_graph()
    
    # 初始化状态
    initial_state = ResumeDemoState(
        user_input="开始隐私号生成流程",
        step="started",
        execution_count=0,
        user_responses=[],
        current_data={}
    )
    
    thread_id = str(uuid.uuid4())
    config = {"configurable": {"thread_id": thread_id}}
    
    print(f"🚀 开始执行工作流")
    print(f"初始状态: {initial_state}")
    
    try:
        # 第一次执行 - 会触发第一次中断
        print(f"\n{'='*50}")
        print("第1次执行：触发手机号输入中断")
        print(f"{'='*50}")
        
        result = graph.invoke(initial_state, config=config)
        
        if "__interrupt__" in result:
            print("⏸️  工作流中断，等待用户输入手机号")
            payload = result["__interrupt__"][0]
            interrupt_data = payload.value if hasattr(payload, 'value') else payload
            print(f"中断信息: {interrupt_data.get('message')}")
            
            # 模拟用户输入手机号
            user_phone = "13812345678"
            print(f"用户输入: {user_phone}")
            
            # 第一次resume
            print(f"\n{'='*50}")
            print("第1次resume：恢复工作流，继续执行")
            print(f"{'='*50}")
            
            result = graph.invoke(Command(resume=user_phone), config=config)
            
            if "__interrupt__" in result:
                print("⏸️  工作流再次中断，等待用户确认手机号")
                payload = result["__interrupt__"][0]
                interrupt_data = payload.value if hasattr(payload, 'value') else payload
                print(f"中断信息: {interrupt_data.get('message')}")
                
                # 模拟用户确认手机号
                user_confirmation = "是"
                print(f"用户确认: {user_confirmation}")
                
                # 第二次resume
                print(f"\n{'='*50}")
                print("第2次resume：恢复工作流，继续执行")
                print(f"{'='*50}")
                
                result = graph.invoke(Command(resume=user_confirmation), config=config)
                
                if "__interrupt__" in result:
                    print("⏸️  工作流第三次中断，等待用户选择类型")
                    payload = result["__interrupt__"][0]
                    interrupt_data = payload.value if hasattr(payload, 'value') else payload
                    print(f"中断信息: {interrupt_data.get('message')}")
                    
                    # 模拟用户选择类型
                    user_type_choice = "1"
                    print(f"用户选择: {user_type_choice}")
                    
                    # 第三次resume
                    print(f"\n{'='*50}")
                    print("第3次resume：恢复工作流，继续执行")
                    print(f"{'='*50}")
                    
                    result = graph.invoke(Command(resume=user_type_choice), config=config)
                    
                    if "__interrupt__" in result:
                        print("❌ 意外中断")
                    else:
                        print("✅ 工作流完成")
                        print(f"最终结果: {result}")
                else:
                    print("✅ 工作流完成")
                    print(f"最终结果: {result}")
            else:
                print("✅ 工作流完成")
                print(f"最终结果: {result}")
        else:
            print("✅ 工作流完成，无中断")
            print(f"结果: {result}")
            
    except Exception as e:
        print(f"❌ 执行失败: {e}")
        import traceback
        traceback.print_exc()


def explain_resume_mechanism():
    """解释resume机制的关键点"""
    print(f"\n📚 Resume机制关键点解释:")
    print(f"=" * 60)
    
    print(f"""
1. **执行流程控制**
   - interrupt() 调用时，工作流暂停并保存状态
   - 状态包括：当前执行位置、变量值、调用栈等
   - 不是简单的"重新执行节点"

2. **状态恢复机制**
   - Command(resume=...) 恢复之前保存的完整状态
   - 从interrupt()调用的确切位置继续执行
   - 所有变量值保持不变

3. **数据传递方式**
   - interrupt() 的参数会传递给用户界面
   - Command(resume=...) 的参数会作为interrupt()的返回值
   - 数据在中断点无缝传递

4. **执行次数控制**
   - 每个节点只执行一次
   - 中断和恢复不会重复执行已完成的代码
   - 只在需要用户输入的地方暂停

5. **实际应用场景**
   - 表单填写：分步骤收集信息
   - 用户确认：关键决策点等待确认
   - 错误处理：用户输入错误时重新输入
   - 分支选择：根据用户选择走不同路径
""")


if __name__ == "__main__":
    # 解释resume机制
    explain_resume_mechanism()
    
    # 演示resume机制
    demonstrate_resume_mechanism()
