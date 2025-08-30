from typing import Literal
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.tools import tool
from langgraph import graph
from langgraph.graph import StateGraph, MessagesState, START, END
from langgraph.checkpoint.memory import MemorySaver
from langgraph.prebuilt import ToolNode
from langchain_community.chat_models.tongyi import ChatTongyi
from pydantic import SecretStr
from IPython.display import Image, display
import os
import sys


# 深度思考模式配置（qwen-plus with reasoning）
chat_model_reasoning = ChatTongyi(
    model="qwen-flash",  
    api_key=SecretStr("sk-df68b4ca15e0497e83894b6a783ee024"),
    top_p=0.8,         
    streaming=True,   
    model_kwargs={
        "temperature": 0.7,      
        "enable_thinking": True  
    }
)

# 非深度思考模式配置（qwen-plus without reasoning）
chat_model_no_reasoning = ChatTongyi(
    model="qwen-flash",  
    api_key=SecretStr("sk-df68b4ca15e0497e83894b6a783ee024"),
    top_p=0.9,        
    streaming=True,   
    model_kwargs={
        "temperature": 0.5,       
    }
)


@tool
def query_weather(query: str) -> str:
    """Query weather info for a place and return a short summary."""
    if "上海" in query or "天气" in query:
        print("进入tool调用，询问上海天气")
        return "明天上午有雨，下午阴天"
    return "明天天晴"



tools = [query_weather]
# 构建工具节点
tool_node = ToolNode(tools)
# 模型绑定工具（注意：返回一个新的 Runnable，需要用该对象来调用）
llm_with_tools = chat_model_no_reasoning.bind_tools(tools)


# 系统提示：指导模型在天气/城市相关问题时必须调用 query_weather工具
SYSTEM_PROMPT = (
    "你是一个善用工具的助手。遇到与天气、城市、上海等相关的查询时，必须优先调用名为 `query_weather` 的工具，"
    "不要直接编造答案。只有在明确不需要工具时，才可以直接回答。"
)


def should_continue(state: MessagesState) -> Literal["tools",END]:
    """
    决定是否继续使用工具节点
    """
    last_message =state['messages'][-1]
    if last_message.tool_calls:
        return "tools"
    return END



def call_model(state: MessagesState):
    """
    调用模型, 传输的是全部的消息列表， state["messages"] 获取的是全部的消息列表。
    首轮若没有系统提示，则自动注入一条系统提示，强制模型优先调用工具。
    """
    messages = state["messages"]
    if not messages or not isinstance(messages[0], SystemMessage):
        messages = [SystemMessage(content=SYSTEM_PROMPT)] + messages
    return {"messages": [llm_with_tools.invoke(messages)]}


#创建 Graph，构建流程图，
# ! 将消息状态处理器传递,可以传递多种类型参数（dict,list,tuple）
graph_builder = StateGraph(MessagesState)


# 添加节点  
graph_builder.add_node("agent", call_model)
graph_builder.add_node("tools", tool_node)

# 添加开始节点
#graph_builder.set_entry_point("agent")


# 添加边
graph_builder.add_edge(START, "agent")  #普通边
graph_builder.add_conditional_edges("agent", should_continue,{True: "tools", False: END}) #条件边
graph_builder.add_edge("tools", "agent") 
# 如果没有工具调用，就结束
graph_builder.add_edge("agent", END)


# 添加检查点  初始化内存， 可以记住以前的消息！ 可以扩展存redis，MongoDB
check_point = MemorySaver()
# ! 编译图表， 进行检查后可以执行  添加检查点
graph = graph_builder.compile(checkpointer=check_point)


# 执行图（如需执行可取消注释）
final_state = graph.invoke(
    {"messages": [HumanMessage(content="今天上海天气怎么样？")]}, 
    config={"configurable": {"thread_id": "1"}}) # 配置会话ID
print(final_state['messages'][-1].content)




def image_display():
    # 终端环境下：将流程图渲染为 PNG 文件并尝试打开
    try:
        png_bytes = graph.get_graph().draw_mermaid_png()
        out_path = os.path.join(os.path.dirname(__file__), "graph.png")
        with open(out_path, "wb") as f:
            f.write(png_bytes)
        # macOS 自动打开
        if sys.platform == "darwin":
            os.system(f'open "{out_path}"')
        elif sys.platform.startswith("linux"):
            os.system(f'xdg-open "{out_path}" >/dev/null 2>&1 || echo "Saved to {out_path}"')
        elif sys.platform.startswith("win"):
            os.startfile(out_path)  # type: ignore
        print(f"流程图已保存：{out_path}")
    except Exception as e:
        print(f"渲染流程图失败：{e}")
        print("你可以改用 graph.get_graph().draw_mermaid() 打印 Mermaid 文本在终端查看。")


