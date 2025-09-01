from langchain_community.chat_models.tongyi import ChatTongyi
import config
from pydantic import SecretStr

cfg = config.Config()
chat_model = ChatTongyi(
            model=cfg.llm_config.model,  
            api_key=SecretStr(cfg.llm_config.api_key),
            top_p=0.8,         
            streaming=True,   
            model_kwargs={
                "temperature": 0.5,      
                "enable_thinking": True  
            }
        )

result = chat_model.invoke("你是谁")
print(result)