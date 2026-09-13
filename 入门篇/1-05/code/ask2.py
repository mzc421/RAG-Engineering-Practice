# -*- coding: UTF-8 -*-
"""
@Project : 
@File    : ask2.py
@Author  : 牧锦程
@Date    : 2026/7/24 13:08
"""

import os
from llama_index.core import VectorStoreIndex, SimpleDirectoryReader, Settings
from llama_index.llms.openai_like import OpenAILike
from llama_index.embeddings.huggingface import HuggingFaceEmbedding

Settings.llm = OpenAILike(
    model="deepseek-v4-flash",
    api_key=os.getenv('DEEPSEEK_API_KEY'),
    api_base="https://api.deepseek.com",
    is_chat_model=True,
)
Settings.embed_model = HuggingFaceEmbedding("BAAI/bge-small-zh-v1.5")

documents = SimpleDirectoryReader(input_files=["markdown.md"]).load_data()

index = VectorStoreIndex.from_documents(documents)
query_engine = index.as_query_engine()

print(query_engine.query("文中举了哪些例子?"))
