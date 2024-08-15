import streamlit as st
import json
import torch
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from langchain.chains import RetrievalQA
from langchain.llms import HuggingFacePipeline
from langchain.document_loaders import TextLoader
from transformers import pipeline as hf_pipeline
 
# 设置标题和描述
st.title("💬 Yuan2.0 AIReader")
st.write("一个结合了 RAG（检索增强生成）的名著阅读助手。")
 
# 加载名著清单 JSON 文件
@st.cache_data
def load_books():
    # 从JSON文件中加载名著清单
    try:
        with open('./code/books.json', 'r', encoding='utf-8') as f:
            return json.load(f)["books"]
    except Exception as e:
        st.error(f"加载名著清单时出错: {e}")
        return []
 
books = load_books()
 
# 名著选择框
book_names = [book["name"] for book in books]
book_selection = st.selectbox("请选择你想提问的名著：", book_names)
 
# 根据选择的名著获取对应的 document_path
def get_document_path(selected_book_name):
    # 根据名著名称查找对应的文档路径
    for book in books:
        if book["name"] == selected_book_name:
            return book["document_path"]
    st.error("未找到对应的文档路径")
    return None
 
document_path = get_document_path(book_selection)
 
# 定义 HuggingFacePipeline 以适配模型
@st.cache_resource
def load_pipeline(model_path: str, torch_dtype):
    # 加载HuggingFace模型管道
    try:
        st.write("正在加载模型，请稍候...")
        hf_model_pipeline = hf_pipeline("text-generation", model=model_path, tokenizer=model_path, torch_dtype=torch_dtype, device=0)
        st.write("模型加载完成。")
        return HuggingFacePipeline(pipeline=hf_model_pipeline)
    except Exception as e:
        st.error(f"加载模型时出错: {e}")
        return None
 
# 加载 HuggingFacePipeline
model_path = './IEITYuan/Yuan2-2B-Mars-hf'
torch_dtype = torch.bfloat16  # A10
pipeline = load_pipeline(model_path, torch_dtype)
 
# 定义向量库索引
@st.cache_resource
def load_vectorstore(document_path: str, embed_model_path: str):
    # 加载文本文件并创建向量库
    loader = TextLoader(document_path)
    embeddings = HuggingFaceEmbeddings(embed_model_path)
    return FAISS.from_documents(loader.load(), embeddings)
 
# 加载向量库
embed_model_path = './AI-ModelScope/bge-small-zh-v1___5'
vectorstore = load_vectorstore(document_path, embed_model_path) if document_path else None
 
# 定义 RAG Chain
qa_chain = RetrievalQA.from_chain_type(
    llm=pipeline,
    chain_type="stuff",
    retriever=vectorstore.as_retriever() if vectorstore else None
) if vectorstore else None
 
# 初始化 session_state
if "messages" not in st.session_state:
    st.session_state["messages"] = []
if 'previous_book' not in st.session_state:
    st.session_state.previous_book = ""
 
# 清空对话历史当名著发生变化时
if st.session_state.previous_book != book_selection:
    st.session_state["messages"] = []
    st.session_state.previous_book = book_selection
 
# 显示对话历史
for msg in st.session_state.messages:
    st.chat_message(msg["role"]).write(msg["content"])
 
# 处理用户输入
if prompt := st.chat_input("请输入你的问题..."):
    try:
        st.session_state.messages.append({"role": "user", "content": prompt})
        st.chat_message("user").write(prompt)
 
        # 使用 RAG Chain 进行问答
        if qa_chain:
            response = qa_chain.run({"query": prompt})
            st.session_state.messages.append({"role": "assistant", "content": response})
            st.chat_message("assistant").write(response)
        else:
            st.error("无法进行问答，请检查向量库是否正确加载。")
    except Exception as e:
        st.error(f"处理用户输入时出错: {e}")