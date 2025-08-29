from langchain_core.tools import tool
from langgraph import graph
from langgraph.graph import StateGraph, MessagesState, START, END
from langgraph.checkpoint.memory import MemorySaver
from langgraph.prebuilt import ToolNode
from langchain_community.chat_models.tongyi import ChatTongyi
from pydantic import SecretStr


#创建 Graph，
graph_builder = StateGraph(MessagesState)


# 深度思考模式配置（qwen-plus with reasoning）
chat_model_reasoning = ChatTongyi(
    model="qwen-plus",  
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
    model="qwen-plus",  
    api_key=SecretStr("sk-df68b4ca15e0497e83894b6a783ee024"),
    top_p=0.9,        
    streaming=True,   
    model_kwargs={
        "temperature": 0.5,       
    }
)



@tool
def search(query: str) -> str:
    """Search the web for information and return a summary."""
    if "上海" in query:
        return "明天上午有雨，下午阴天"
    return "明天天晴"


tools = [search]

# 构建工具节点
tool_node = ToolNode(tools)




    






