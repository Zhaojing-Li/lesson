#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
演示 qwen 模型的 enable_thinking 参数使用
展示深度思考模式与非深度思考模式的区别
"""

from langgraph_base import (
    create_custom_chat_model,
    get_chat_model,
    chat_model_reasoning,
    chat_model_standard
)
from langchain_core.messages import HumanMessage

def demo_enable_thinking():
    """演示 enable_thinking 参数的效果"""
    print("🧠 演示 qwen 模型的深度思考模式控制\n")
    print("=" * 60)
    
    # 测试问题
    test_question = "请解决这个数学问题：一个数的3倍加上5等于20，求这个数。"
    
    print(f"📝 测试问题: {test_question}\n")
    
    # 1. 深度思考模式 (enable_thinking=True)
    print("🔍 模式1: 深度思考模式 (enable_thinking=True)")
    print("-" * 40)
    
    thinking_model = create_custom_chat_model(
        model_name="qwen-plus",
        enable_thinking=True,
        temperature=0.3,
        top_p=0.8,
        streaming=False
    )
    
    print("模型配置:")
    print(f"  - model: qwen-plus")
    print(f"  - enable_thinking: True")
    print(f"  - temperature: 0.3")
    print(f"  - top_p: 0.8")
    print()
    
    # 注意：实际调用需要有效的API密钥
    print("💭 模拟输出 (启用深度思考):")
    print("思考过程: 这是一个简单的一元一次方程...")
    print("最终答案: 设这个数为x，则3x + 5 = 20，解得x = 5")
    print()
    
    # 2. 快速响应模式 (enable_thinking=False)
    print("⚡ 模式2: 快速响应模式 (enable_thinking=False)")
    print("-" * 40)
    
    fast_model = create_custom_chat_model(
        model_name="qwen-flash",
        enable_thinking=False,
        temperature=0.3,
        top_p=0.8,
        streaming=False
    )
    
    print("模型配置:")
    print(f"  - model: qwen-flash")
    print(f"  - enable_thinking: False")
    print(f"  - temperature: 0.3")
    print(f"  - top_p: 0.8")
    print()
    
    print("💫 模拟输出 (禁用深度思考):")
    print("答案: 这个数是5。")
    print()

def demo_different_scenarios():
    """演示不同场景下的模式选择"""
    print("=" * 60)
    print("🎯 不同场景的模式选择建议\n")
    
    scenarios = [
        {
            "scenario": "复杂数学问题求解",
            "question": "求解二次方程 x² - 5x + 6 = 0",
            "recommended": "enable_thinking=True",
            "reason": "需要逐步推理和验证"
        },
        {
            "scenario": "日常闲聊对话",
            "question": "今天天气真不错呢",
            "recommended": "enable_thinking=False",
            "reason": "简单对话，无需深度思考"
        },
        {
            "scenario": "编程调试分析",
            "question": "为什么我的Python代码出现IndexError?",
            "recommended": "enable_thinking=True",
            "reason": "需要分析代码逻辑和错误原因"
        },
        {
            "scenario": "创意文案写作",
            "question": "写一句广告语推销新款手机",
            "recommended": "enable_thinking=False",
            "reason": "创意写作，直接输出更自然"
        },
        {
            "scenario": "逻辑推理题",
            "question": "三个人中有一个说谎者，如何推理出谁在说谎？",
            "recommended": "enable_thinking=True",
            "reason": "需要逐步分析逻辑关系"
        }
    ]
    
    for i, scenario in enumerate(scenarios, 1):
        print(f"📋 场景 {i}: {scenario['scenario']}")
        print(f"   问题: {scenario['question']}")
        print(f"   推荐: {scenario['recommended']}")
        print(f"   原因: {scenario['reason']}")
        print()

def demo_model_comparison():
    """演示预设模型的配置对比"""
    print("=" * 60)
    print("⚖️  预设模型配置对比\n")
    
    models_config = [
        {
            "name": "深度思考模式 (reasoning)",
            "model": "qwen-plus",
            "enable_thinking": True,
            "temperature": 0.7,
            "top_p": 0.8,
            "use_case": "复杂推理、分析问题"
        },
        {
            "name": "标准模式 (standard)",
            "model": "qwen-flash",
            "enable_thinking": False,
            "temperature": 0.5,
            "top_p": 0.9,
            "use_case": "日常对话、平衡性能"
        },
        {
            "name": "创造性模式 (creative)",
            "model": "qwen-max",
            "enable_thinking": False,
            "temperature": 0.9,
            "top_p": 0.95,
            "use_case": "创意写作、头脑风暴"
        },
        {
            "name": "精确模式 (precise)",
            "model": "qwen-plus",
            "enable_thinking": True,
            "temperature": 0.1,
            "top_p": 0.5,
            "use_case": "数学计算、精确答案"
        }
    ]
    
    for config in models_config:
        print(f"🤖 {config['name']}")
        print(f"   模型: {config['model']}")
        print(f"   深度思考: {'启用' if config['enable_thinking'] else '禁用'}")
        print(f"   温度: {config['temperature']}")
        print(f"   top_p: {config['top_p']}")
        print(f"   适用: {config['use_case']}")
        print()

def demo_api_cost_consideration():
    """演示API成本考虑"""
    print("=" * 60)
    print("💰 API 成本考虑\n")
    
    print("🔍 深度思考模式 (enable_thinking=True)")
    print("   ✅ 优势: 答案质量高，推理过程清晰")
    print("   ❌ 劣势: 响应时间长，Token消耗多（思考内容计费）")
    print("   💡 建议: 用于重要的复杂问题")
    print()
    
    print("⚡ 快速响应模式 (enable_thinking=False)")
    print("   ✅ 优势: 响应快速，Token消耗少")
    print("   ❌ 劣势: 可能缺少深度分析")
    print("   💡 建议: 用于日常对话和简单问题")
    print()
    
    print("🎯 最佳实践:")
    print("   1. 根据问题复杂度选择模式")
    print("   2. 开发时使用快速模式测试")
    print("   3. 生产环境根据需求精确控制")
    print("   4. 监控Token使用量和成本")

if __name__ == "__main__":
    try:
        demo_enable_thinking()
        demo_different_scenarios()
        demo_model_comparison()
        demo_api_cost_consideration()
        
        print("=" * 60)
        print("✅ 演示完成！")
        print("\n💡 关键要点:")
        print("   • enable_thinking=True: 深度思考，质量高，成本高")
        print("   • enable_thinking=False: 快速响应，速度快，成本低")
        print("   • 根据具体场景选择合适的模式")
        print("   • 可以动态创建自定义配置的模型")
        
    except Exception as e:
        print(f"❌ 演示过程中出现错误: {e}")
        print("\n💡 提示: 如需实际测试API调用，请确保:")
        print("   1. API密钥有效")
        print("   2. 网络连接正常")
        print("   3. 账户余额充足")