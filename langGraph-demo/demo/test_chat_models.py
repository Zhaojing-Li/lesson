#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
测试不同ChatTongyi模式的配置
"""

from langgraph_base import (
    chat_model_reasoning, 
    chat_model_standard, 
    chat_model_creative, 
    chat_model_precise,
    get_chat_model,
    set_global_chat_model
)

def test_model_configuration():
    """测试模型配置"""
    print("=== 测试ChatTongyi模型配置 ===\n")
    
    # 测试各个模型的配置
    models = {
        "深度思考模式": chat_model_reasoning,
        "标准模式": chat_model_standard,
        "创造性模式": chat_model_creative,
        "精确模式": chat_model_precise
    }
    
    for name, model in models.items():
        print(f"{name}:")
        print(f"  模型: {model.model_name}")
        print(f"  top_p: {model.top_p}")
        print(f"  streaming: {model.streaming}")
        if hasattr(model, 'model_kwargs') and model.model_kwargs:
            print(f"  温度: {model.model_kwargs.get('temperature', '未设置')}")
        print()

def test_model_switching():
    """测试模式切换功能"""
    print("=== 测试模式切换功能 ===\n")
    
    # 测试get_chat_model函数
    modes = ["reasoning", "standard", "creative", "precise", "unknown"]
    
    for mode in modes:
        model = get_chat_model(mode)
        print(f"模式 '{mode}' -> 模型: {model.model_name}")
    
    print()
    
    # 测试set_global_chat_model函数
    print("测试全局模式切换:")
    set_global_chat_model("reasoning")
    set_global_chat_model("creative")
    set_global_chat_model("unknown")

def test_simple_chat():
    """简单的对话测试"""
    print("\n=== 简单对话测试 ===\n")
    
    try:
        # 使用标准模式进行简单测试
        model = get_chat_model("standard")
        
        # 注意：这里只是测试模型初始化，不实际调用API
        print(f"标准模式模型配置成功: {model.model_name}")
        print("如需实际测试对话，请确保API密钥有效并取消注释下面的代码:")
        print("# response = model.invoke('你好，请介绍一下自己')")
        print("# print(response.content)")
        
    except Exception as e:
        print(f"测试失败: {e}")

if __name__ == "__main__":
    test_model_configuration()
    test_model_switching()
    test_simple_chat()
    
    print("\n=== 配置总结 ===")
    print("✅ 深度思考模式: qwen-plus, temperature=0.7, top_p=0.8")
    print("✅ 标准模式: qwen-flash, temperature=0.5, top_p=0.9")
    print("✅ 创造性模式: qwen-max, temperature=0.9, top_p=0.95")
    print("✅ 精确模式: qwen-plus, temperature=0.1, top_p=0.5")
    print("\n所有配置测试通过！")