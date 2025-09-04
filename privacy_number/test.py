from langchain_community.chat_models.tongyi import ChatTongyi
import config
from pydantic import SecretStr
from langchain_community.chat_models import ChatOpenAI
from langfuse import Langfuse, get_client
from langfuse.langchain import CallbackHandler


cfg = config.Config()
chat_model = ChatTongyi(
            model=cfg.llm_config.model,  
            api_key=SecretStr(cfg.llm_config.api_key),
            top_p=0.8,         
            streaming=True,   
            model_kwargs={
                "temperature": 0.5,      
                "enable_thinking": True,
            }
        )

Langfuse(
    secret_key="sk-lf-0c8ca4b7-e188-4803-b2d7-d710b6ba804a",
    public_key="pk-lf-33bc5786-7976-4339-9808-ea05e03f11f6",
    host="http://localhost:3000"
    )
langfuse = get_client()
langfuse_handler = CallbackHandler()





# chat_model = ChatOpenAI(
#     base_url="http://auto-scan.jd.com/v1",  
#     api_key= None,  
#     model="qwen3-14b",  
#     streaming=True,
#     model_kwargs={
#         "top_p": 0.8,
#         "temperature": 0.5,      
#         "enable_thinking": True,
#     }
# )

result = chat_model.invoke("你是谁",config={"callbacks": [langfuse_handler],  "langfuse_session_id": "random-session"})
print(result.additional_kwargs.get('reasoning_content',str))