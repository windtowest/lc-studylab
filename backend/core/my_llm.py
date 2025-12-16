import os
import dotenv

from langchain_openai import ChatOpenAI
from zai import ZhipuAiClient

dotenv.load_dotenv()
# 使用ChatOpenAI，但指向DeepSeek的API端点
my_deepseek = ChatOpenAI(
    model="deepseek-chat",  # DeepSeek的模型名称
    api_key=os.getenv("DEEPSEEK_API_KEY"),
    base_url=os.getenv("DEEPSEEK_BASE_URL"),  # 关键：修改API基础地址
    streaming = True,
    max_tokens = 512,
    temperature = 1.5
)


# 本地vllm部署的qwen3

qwen3_local = ChatOpenAI(
    model="qwen3-8b",
    api_key='xx',
    base_url=os.getenv("LOCAL_BASE_URL"),
    temperature = 1.5,
    extra_body={'chat_template_kwargs': {'enable_thing': True}}
)
zhipuClient = ZhipuAiClient(api_key="ea2e536468d84b76b9f623edaed8b5bd.5eiuzpjlHSmF1CJj")


