"""
隐私号平台配置文件
"""

from dataclasses import dataclass
import os
from typing import Dict, Any


@dataclass
class LLMConfig:
     # API配置
    api_key:str  = "sk-df68b4ca15e0497e83894b6a783ee024"
    model:str = "qwen-flash"


class Config:
    """系统配置类"""
    def __init__(self):
        self.llm_config = LLMConfig()
   
    
