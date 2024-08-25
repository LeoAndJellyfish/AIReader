import torch
import streamlit as st
from transformers import AutoTokenizer, AutoModelForCausalLM
from langchain.prompts import PromptTemplate
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.chains import LLMChain
from langchain.chains.question_answering import load_qa_chain
from langchain.llms.base import LLM
from langchain.callbacks.manager import CallbackManagerForLLMRun
from langchain.text_splitter import RecursiveCharacterTextSplitter

from typing import Any, List, Optional

# 向量模型下载
from modelscope import snapshot_download
model_dir = snapshot_download('AI-ModelScope/bge-small-zh-v1.5', cache_dir='./')

# 源大模型下载
model_dir = snapshot_download('IEITYuan/Yuan2-2B-Mars-hf', cache_dir='./')

# 定义模型路径
model_path = './IEITYuan/Yuan2-2B-Mars-hf'

# 定义向量模型路径
embedding_model_path = './AI-ModelScope/bge-small-zh-v1___5'

# 定义模型数据类型
torch_dtype = torch.float32  # CPU

# 定义源大模型类
class Yuan2_LLM(LLM):
    """
    class for Yuan2_LLM
    """
    tokenizer: AutoTokenizer = None
    model: AutoModelForCausalLM = None

    def __init__(self, mode_path: str):
        super().__init__()

        # 加载预训练的分词器和模型
        print("Creat tokenizer...")
        self.tokenizer = AutoTokenizer.from_pretrained(mode_path, add_eos_token=False, add_bos_token=False, eos_token='<eod>')
        self.tokenizer.add_tokens(['<sep>', '<pad>', '<mask>', '<predict>', '<FIM_SUFFIX>', '<FIM_PREFIX>', '<FIM_MIDDLE>', '<commit_before>', '<commit_msg>', '<commit_after>', '<jupyter_start>', '<jupyter_text>', '<jupyter_code>', '<jupyter_output>', '<empty_output>'], special_tokens=True)

        print("Creat model...")
        self.model = AutoModelForCausalLM.from_pretrained(mode_path, torch_dtype=torch_dtype, trust_remote_code=True)

    def _call(
        self,
        prompt: str,
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> str:
        prompt = prompt.strip()
        prompt += "<sep>"
        inputs = self.tokenizer(prompt, return_tensors="pt")["input_ids"]
        outputs = self.model.generate(inputs, do_sample=False, max_new_tokens=2048)
        output = self.tokenizer.decode(outputs[0])
        response = output.split("<sep>")[-1].split("<eod>")[0]

        return response

    @property
    def _llm_type(self) -> str:
        return "Yuan2_LLM"

# 定义一个函数，用于获取llm和embeddings
@st.cache_resource
def get_models():
    llm = Yuan2_LLM(model_path)

    model_kwargs = {'device': 'cpu'}
    encode_kwargs = {'normalize_embeddings': True}  # set True to compute cosine similarity
    embeddings = HuggingFaceEmbeddings(
        model_name=embedding_model_path,
        model_kwargs=model_kwargs,
        encode_kwargs=encode_kwargs,
    )
    return llm, embeddings

chatbot_template = '''
假设你是一个文章阅读助手，请基于背景，简要回答问题。

背景：
{context}

问题：
{question}
'''.strip()

# 定义ChatBot类
class ChatBot:
    """
    class for ChatBot.
    """

    def __init__(self, llm, embeddings):
        self.prompt = PromptTemplate(
            input_variables=["text"],
            template=chatbot_template
        )
        self.chain = load_qa_chain(llm=llm, chain_type="stuff", prompt=self.prompt)
        self.embeddings = embeddings

        # 加载 text_splitter
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=450,
            chunk_overlap=10,
            length_function=len
        )

    def run(self, docs, query):
        # 读取所有内容
        text = ''.join([doc.page_content for doc in docs])

        # 切分成chunks
        all_chunks = self.text_splitter.split_text(text=text)

        # 转成向量并存储
        VectorStore = FAISS.from_texts(all_chunks, embedding=self.embeddings)

        # 检索相似的chunks
        chunks = VectorStore.similarity_search(query=query, k=1)

        # 生成回复
        response = self.chain.run(input_documents=chunks, question=query)

        return chunks, response

def main():
    # 创建一个标题
    st.title('💬 Yuan2.0 AIReader')

    # 获取llm和embeddings
    llm, embeddings = get_models()

    # 初始化ChatBot
    chatbot = ChatBot(llm, embeddings)

    # 上传pdf
    uploaded_file = st.file_uploader("Upload your file", type=['pdf', 'txt'])

    if uploaded_file:
        # 加载上传PDF的内容
        file_content = uploaded_file.read()

        # 获取文件扩展名
        file_extension = uploaded_file.name.split('.')[-1]

        # 写入临时文件
        temp_file_path = f"temp.{file_extension}"
        with open(temp_file_path, "wb") as temp_file:
            temp_file.write(file_content)

        # 加载临时文件中的内容
        if file_extension == 'pdf':
            loader = PyPDFLoader(temp_file_path)
        elif file_extension == 'txt':
            loader = TextLoader(temp_file_path)
        docs = loader.load()

        # 显示文档内容在主页面
        st.subheader("Document Content:")
        for doc in docs:
            st.write(doc.page_content)

        # 使用侧边栏显示对话历史和用户输入
        st.sidebar.header("Chat History")
        messages = st.sidebar.empty()
        query = st.sidebar.text_input("Ask questions about your file", key="query")

        if query:
            # 检索 + 生成回复
            chunks, response = chatbot.run(docs, query)

            # 更新侧边栏的对话历史
            with messages:
                # 清空之前的对话历史
                messages.empty()

                # 显示用户的提问
                st.write(f"You: {query}")

                # 显示模型的输出
                st.write(f"Assistant: {response}")

if __name__ == '__main__':
    main()
