import torch
from langchain_core.embeddings import Embeddings
from sentence_transformers import SentenceTransformer

from config import settings


class MyCustomEmbeddings(Embeddings):

    def __init__(self, model_path: str):
        self.qwen3_embedding = SentenceTransformer(
            model_path,
            model_kwargs={"attn_implementation": "flash_attention_2", "device_map": "cuda:0", "torch_dtype": torch.float16},
            tokenizer_kwargs={"padding_side": "left"},
        )

    def check_device(self):
        """检查模型实际运行在哪个设备上"""
        if hasattr(self.qwen3_embedding, 'device'):
            print(f"编码器设备: {self.qwen3_embedding.device}")
            return self.qwen3_embedding.device

    def embed_query(self, text: str) -> list[float]:
        return self.embed_documents([text])[0]

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return self.qwen3_embedding.encode(texts)

if __name__ == "__main__":
    model = MyCustomEmbeddings(settings.local_model_path + settings.embedding_model)
    model.check_device()