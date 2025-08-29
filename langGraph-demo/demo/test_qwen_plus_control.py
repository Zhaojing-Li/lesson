#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
测试 qwen-plus 模型的深度思考模式控制功能
"""

from langgraph_base import (
    create_qwen_plus_model,
    get_qwen_plus_for_scenario,
    chat_model_reasoning,
    chat_model_plus_fast
)

def test_qwen_plus_configurations():
    """测试不同配置的 qwen-plus 模型"""
    print("🧠 qwen-plus 模型深度思考控制测试\n")
    print("=" * 60)
    
    # 测试自定义配置
    print("📋 1. 自定义配置测试")
    print("-" * 30)
    
    models = [
        {
            "name": "深度思考模式",
            "model": create_qwen_plus_model(enable_thinking=True, temperature=0.3),
            "config": "enable_thinking=True, temperature=0.3"
        },
        {
            "name": "快速响应模式", 
            "model": create_qwen_plus_model(enable_thinking=False, temperature=0.7),
            "config": "enable_thinking=False, temperature=0.7"
        }
    ]
    
    for model_info in models:
        print(f"🤖 {model_info['name']}")
        print(f"   配置: {model_info['config']}")
        print(f"   模型: {model_info['model'].model_name}")
        print(f"   thinking: {model_info['model'].model_kwargs.get('enable_thinking', 'N/A')}")
        print(f"   temperature: {model_info['model'].model_kwargs.get('temperature', 'N/A')}")
        print()

def test_scenario_models():
    """测试针对不同场景优化的模型"""
    print("📋 2. 场景优化模型测试")
    print("-" * 30)
    
    scenarios = [
        ("reasoning", "复杂推理场景"),
        ("math", "数学计算场景"),
        ("chat", "日常对话场景"),
        ("creative", "创意写作场景"),
        ("analysis", "文本分析场景")
    ]
    
    for scenario, description in scenarios:
        model = get_qwen_plus_for_scenario(scenario)
        thinking = model.model_kwargs.get('enable_thinking', False)
        temp = model.model_kwargs.get('temperature', 0.5)
        top_p = model.top_p
        streaming = model.streaming
        
        print(f"🎯 {description} ({scenario})")
        print(f"   深度思考: {'启用' if thinking else '禁用'}")
        print(f"   温度: {temp}")
        print(f"   top_p: {top_p}")
        print(f"   流式输出: {'是' if streaming else '否'}")
        print()

def test_usage_examples():
    """展示使用示例"""
    print("📋 3. 实际使用示例")
    print("-" * 30)
    
    examples = [
        {
            "scenario": "数学问题求解",
            "question": "求解方程组: 2x + 3y = 7, x - y = 1",
            "recommended_model": "math",
            "reason": "需要精确计算和逐步推理"
        },
        {
            "scenario": "日常对话",
            "question": "今天天气怎么样？",
            "recommended_model": "chat", 
            "reason": "简单问答，快速响应即可"
        },
        {
            "scenario": "创意写作",
            "question": "写一首关于春天的诗",
            "recommended_model": "creative",
            "reason": "创意任务，高温度增加创造性"
        },
        {
            "scenario": "逻辑推理",
            "question": "如果所有A都是B，所有B都是C，那么所有A都是C吗？",
            "recommended_model": "reasoning",
            "reason": "需要逻辑推理和深度思考"
        }
    ]
    
    for example in examples:
        model = get_qwen_plus_for_scenario(example["recommended_model"])
        thinking = model.model_kwargs.get('enable_thinking', False)
        
        print(f"💡 {example['scenario']}")
        print(f"   问题: {example['question']}")
        print(f"   推荐模型: {example['recommended_model']}")
        print(f"   深度思考: {'启用' if thinking else '禁用'}")
        print(f"   原因: {example['reason']}")
        print()

def test_performance_comparison():
    """性能对比说明"""
    print("📋 4. 性能对比分析")
    print("-" * 30)
    
    comparison_data = [
        {
            "模式": "深度思考 (enable_thinking=True)",
            "响应时间": "较慢 (2-5秒)",
            "Token消耗": "高 (包含思考过程)",
            "答案质量": "高 (逐步推理)",
            "适用场景": "复杂问题、数学计算、逻辑推理"
        },
        {
            "模式": "快速响应 (enable_thinking=False)", 
            "响应时间": "快速 (1-2秒)",
            "Token消耗": "低 (仅最终答案)",
            "答案质量": "中等 (直接回答)",
            "适用场景": "日常对话、创意写作、简单问答"
        }
    ]
    
    for data in comparison_data:
        print(f"⚖️  {data['模式']}")
        print(f"   响应时间: {data['响应时间']}")
        print(f"   Token消耗: {data['Token消耗']}")
        print(f"   答案质量: {data['答案质量']}")
        print(f"   适用场景: {data['适用场景']}")
        print()

def test_best_practices():
    """最佳实践建议"""
    print("📋 5. 最佳实践建议")
    print("-" * 30)
    
    practices = [
        "🎯 根据问题复杂度选择模式：复杂问题启用深度思考，简单问题使用快速模式",
        "💰 成本控制：开发测试时使用快速模式，生产环境根据需求精确选择",
        "⚡ 性能优化：高频调用场景优先使用快速模式",
        "🔄 动态切换：根据用户输入自动判断是否需要深度思考",
        "📊 监控分析：跟踪不同模式的使用情况和效果",
        "🎨 场景专用：为特定场景预设优化配置",
        "🧪 A/B测试：对比不同配置在实际应用中的表现"
    ]
    
    for practice in practices:
        print(f"   {practice}")
    print()

if __name__ == "__main__":
    try:
        test_qwen_plus_configurations()
        test_scenario_models()
        test_usage_examples()
        test_performance_comparison()
        test_best_practices()
        
        print("=" * 60)
        print("✅ qwen-plus 深度思考控制功能测试完成！")
        print("\n🚀 核心功能:")
        print("   • create_qwen_plus_model(): 灵活创建自定义配置模型")
        print("   • get_qwen_plus_for_scenario(): 获取场景优化模型")
        print("   • enable_thinking参数: 精确控制深度思考模式")
        print("   • 多种预设配置: 满足不同使用需求")
        
    except Exception as e:
        print(f"❌ 测试过程中出现错误: {e}")
        print("\n💡 提示: 这是配置测试，实际API调用需要有效密钥")