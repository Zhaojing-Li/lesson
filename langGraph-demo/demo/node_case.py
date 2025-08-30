from langchain_core.runnables import RunnableConfig
from langgraph.graph import StateGraph,START,END


graph = StateGraph(dict)

def base_node(state:dict,config:RunnableConfig):
    print("In node:",config["configurable"]["user_id"])
    return {"result": f"Hello ,{state['inout']}!"}


def other_node(state: dict):
    return state


# 添加节点
graph.add_node("base_node", base_node)
graph.add_node("other_node", other_node)


graph.add_edge(START, "base_node")
graph.add_node("start_node", "other_node")
graph.add_node("END", "other_node")



