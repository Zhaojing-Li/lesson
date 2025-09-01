from typing import Literal
from langgraph.types import interrupt, Command
from langgraph.graph import StateGraph, MessagesState, START, END
from langgraph.checkpoint.memory import MemorySaver

def human_approval(state: MessagesState) -> Command[Literal["some_node", "another_node"]]:
    is_approved = interrupt(
        {
            "question": "这是正确的吗？",
            # 展示应由人类审查和批准的输出。
            "llm_output": state["llm_output"]
        }
    )

    if is_approved:
        return Command(goto="some_node")
    else:
        return Command(goto="another_node")

# 将节点添加到图形中的适当位置并连接到相关节点。
graph_builder = StateGraph(dict)
graph_builder.add_node("human_approval", human_approval)

graph_builder.add_edge(START, "human_approval")
graph_builder.add_edge("human_approval", END)
checkpointer = MemorySaver()
graph = graph_builder.compile(checkpointer=checkpointer)

# 在运行图形并触发中断后，图形将暂停。
# 用批准或拒绝恢复。
thread_config = {"configurable": {"thread_id": "some_id"}}
graph.invoke(Command(resume=True), config=thread_config)