#!/usr/bin/env python3
"""
interrupt函数数据传递和解析机制演示
展示如何设置和解析interrupt中的JSON数据
"""

import uuid
from typing import TypedDict, Dict, Any
from langgraph.graph import StateGraph, START, END
from langgraph.types import interrupt, Command
from langgraph.checkpoint.memory import MemorySaver


class DemoState(TypedDict):
    """演示状态"""
    user_input: str
    step: str


def demo_node_with_interrupt(state: DemoState) -> DemoState:
    """演示节点：展示不同的interrupt数据格式"""
    print(f"\n=== 演示节点执行 ===")
    print(f"用户输入: {state['user_input']}")
    
    # 根据输入内容决定触发哪种中断
    if "手机号" in state['user_input']:
        # 示例1：标准格式的中断数据
        print("🛑 触发手机号确认中断")
        interrupt({
            "type": "phone_confirmation",
            "message": "请确认手机号是否为被调查人的",
            "phone_number": "13812345678",
            "options": ["是", "否"],
            "instructions": "输入 '是' 或 '否' 进行确认"
        })
    
    elif "类型" in state['user_input']:
        # 示例2：复杂格式的中断数据
        print("🛑 触发类型选择中断")
        interrupt({
            "type": "type_selection",
            "message": "请选择隐私号类型",
            "choices": [
                {
                    "id": "1",
                    "name": "可回拨",
                    "description": "被调查人可以回拨",
                    "features": ["双向通信", "回拨路由"]
                },
                {
                    "id": "2", 
                    "name": "不可回拨",
                    "description": "只能单向拨打",
                    "features": ["单向通信", "隐私保护"]
                }
            ],
            "default_choice": "1",
            "help_text": "可回拨适合需要双向通信的场景，不可回拨适合单向联系"
        })
    
    elif "自定义" in state['user_input']:
        # 示例3：完全自定义格式的中断数据
        print("🛑 触发自定义格式中断")
        interrupt({
            "custom_type": "user_defined_interrupt",
            "user_message": "这是一个完全自定义的中断消息",
            "data_structure": {
                "nested": {
                    "deep": {
                        "value": "深层嵌套的数据"
                    }
                }
            },
            "array_data": ["item1", "item2", "item3"],
            "boolean_flag": True,
            "numeric_value": 42,
            "null_value": None,
            "任意字段名": "支持中文字段名",
            "special_chars": "!@#$%^&*()",
            "emoji": "🚀📱💻"
        })
    
    else:
        # 示例4：简单格式的中断数据
        print("🛑 触发简单中断")
        interrupt({
            "message": "请输入更多信息",
            "required": True
        })
    
    return state


def build_demo_graph():
    """构建演示工作流图"""
    workflow = StateGraph(DemoState)
    workflow.add_node("demo", demo_node_with_interrupt)
    workflow.add_edge(START, "demo")
    workflow.add_edge("demo", END)
    
    checkpointer = MemorySaver()
    return workflow.compile(checkpointer=checkpointer)


def demonstrate_interrupt_parsing():
    """演示interrupt数据的解析过程"""
    print("🔍 interrupt函数数据传递和解析机制演示")
    print("=" * 80)
    
    graph = build_demo_graph()
    
    # 测试不同的中断数据格式
    test_cases = [
        {
            "name": "手机号确认中断",
            "input": "确认手机号13812345678",
            "description": "标准格式的中断数据"
        },
        {
            "name": "类型选择中断", 
            "input": "选择隐私号类型",
            "description": "复杂格式的中断数据"
        },
        {
            "name": "自定义格式中断",
            "input": "测试自定义格式",
            "description": "完全自定义的中断数据"
        },
        {
            "name": "简单中断",
            "input": "简单测试",
            "description": "简单格式的中断数据"
        }
    ]
    
    for i, test_case in enumerate(test_cases, 1):
        print(f"\n{'='*80}")
        print(f"测试 {i}: {test_case['name']}")
        print(f"描述: {test_case['description']}")
        print(f"输入: {test_case['input']}")
        print(f"{'='*80}")
        
        # 执行工作流
        initial_state = DemoState(
            user_input=test_case['input'],
            step="started"
        )
        
        thread_id = str(uuid.uuid4())
        config = {"configurable": {"thread_id": thread_id}}
        
        try:
            # 执行工作流
            result = graph.invoke(initial_state, config=config)
            
            # 解析中断数据
            if "__interrupt__" in result:
                parse_interrupt_data(result["__interrupt__"][0])
            else:
                print("✅ 工作流完成，无中断")
                
        except Exception as e:
            print(f"❌ 执行失败: {e}")


def parse_interrupt_data(interrupt_obj):
    """详细解析中断数据"""
    print(f"\n📊 中断数据解析过程:")
    print(f"中断对象类型: {type(interrupt_obj)}")
    print(f"中断对象ID: {interrupt_obj.id}")
    
    # 获取中断数据
    if hasattr(interrupt_obj, 'value'):
        interrupt_data = interrupt_obj.value
        print(f"中断数据类型: {type(interrupt_data)}")
        print(f"中断数据内容: {interrupt_data}")
        
        # 详细解析各个字段
        print(f"\n🔍 字段解析:")
        for key, value in interrupt_data.items():
            print(f"  字段名: '{key}'")
            print(f"  字段值: {value}")
            print(f"  值类型: {type(value)}")
            print(f"  是否可访问: {key in interrupt_data}")
            print(f"  使用get()方法: {interrupt_data.get(key, '默认值')}")
            print()
        
        # 演示不同的访问方式
        print(f"📝 数据访问演示:")
        print(f"  直接访问: {interrupt_data.get('message', '无message字段')}")
        print(f"  嵌套访问: {interrupt_data.get('data_structure', {}).get('nested', {}).get('deep', {}).get('value', '无嵌套数据')}")
        print(f"  数组访问: {interrupt_data.get('array_data', [])}")
        print(f"  中文字段: {interrupt_data.get('任意字段名', '无中文字段')}")
        
        # 演示如何根据字段类型进行不同处理
        print(f"\n🎯 根据字段类型处理:")
        handle_interrupt_by_type(interrupt_data)
        
    else:
        print("❌ 无法获取中断数据")


def handle_interrupt_by_type(interrupt_data):
    """根据中断数据类型进行不同处理"""
    print(f"  处理中断类型: {interrupt_data.get('type', 'unknown')}")
    
    # 根据type字段决定处理方式
    interrupt_type = interrupt_data.get('type', 'unknown')
    
    if interrupt_type == "phone_confirmation":
        print(f"  📱 手机号确认处理:")
        print(f"    手机号: {interrupt_data.get('phone_number')}")
        print(f"    选项: {interrupt_data.get('options')}")
        print(f"    说明: {interrupt_data.get('instructions')}")
        
    elif interrupt_type == "type_selection":
        print(f"  🔧 类型选择处理:")
        choices = interrupt_data.get('choices', [])
        for choice in choices:
            print(f"    选项{choice.get('id')}: {choice.get('name')} - {choice.get('description')}")
        print(f"    默认选择: {interrupt_data.get('default_choice')}")
        print(f"    帮助信息: {interrupt_data.get('help_text')}")
        
    elif interrupt_data.get('custom_type') == "user_defined_interrupt":
        print(f"  🎨 自定义格式处理:")
        print(f"    自定义消息: {interrupt_data.get('user_message')}")
        print(f"    嵌套数据: {interrupt_data.get('data_structure')}")
        print(f"    数组数据: {interrupt_data.get('array_data')}")
        print(f"    中文字段: {interrupt_data.get('任意字段名')}")
        print(f"    特殊字符: {interrupt_data.get('special_chars')}")
        print(f"    Emoji: {interrupt_data.get('emoji')}")
        
    else:
        print(f"  📝 简单消息处理:")
        print(f"    消息: {interrupt_data.get('message')}")
        print(f"    是否必需: {interrupt_data.get('required')}")


def demonstrate_field_access():
    """演示字段访问的各种方式"""
    print(f"\n🔧 字段访问方式演示:")
    print(f"=" * 50)
    
    # 模拟一个中断数据
    sample_data = {
        "type": "demo",
        "message": "演示消息",
        "nested": {
            "level1": {
                "level2": "深层数据"
            }
        },
        "array": ["item1", "item2", "item3"],
        "中文字段": "支持中文",
        "number": 42,
        "boolean": True,
        "null_value": None
    }
    
    print(f"原始数据: {sample_data}")
    print()
    
    # 1. 直接访问
    print(f"1. 直接访问:")
    print(f"   sample_data['type'] = {sample_data['type']}")
    print(f"   sample_data['message'] = {sample_data['message']}")
    
    # 2. get()方法访问
    print(f"\n2. get()方法访问:")
    print(f"   sample_data.get('type') = {sample_data.get('type')}")
    print(f"   sample_data.get('nonexistent') = {sample_data.get('nonexistent')}")
    print(f"   sample_data.get('nonexistent', '默认值') = {sample_data.get('nonexistent', '默认值')}")
    
    # 3. 嵌套访问
    print(f"\n3. 嵌套访问:")
    print(f"   sample_data['nested']['level1']['level2'] = {sample_data['nested']['level1']['level2']}")
    print(f"   sample_data.get('nested', {}).get('level1', {}).get('level2') = {sample_data.get('nested', {}).get('level1', {}).get('level2')}")
    
    # 4. 数组访问
    print(f"\n4. 数组访问:")
    print(f"   sample_data['array'][0] = {sample_data['array'][0]}")
    print(f"   sample_data.get('array', [])[1] = {sample_data.get('array', [])[1]}")
    
    # 5. 中文字段访问
    print(f"\n5. 中文字段访问:")
    print(f"   sample_data['中文字段'] = {sample_data['中文字段']}")
    print(f"   sample_data.get('中文字段') = {sample_data.get('中文字段')}")


if __name__ == "__main__":
    # 演示字段访问方式
    demonstrate_field_access()
    
    # 演示中断数据解析
    demonstrate_interrupt_parsing()
