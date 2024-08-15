# Yuan2.0 AIReader（在建）
AIReader 是一个基于Yuan2.0的名著阅读助手，它可以帮助用户快速阅读名著，并回答相关问题。
---
## 环境准备
```Shell
# 查看已安装依赖
pip list
# 安装环境
pip install --upgrade pip
pip install faiss-gpu streamlit langchain langchain_community langchain_huggingface
```
---

## 启动服务
首先运行code\Dmodle.py下载向量模型与源大模型  
然后运行
```Shell
streamlit run code/main.py --server.address 127.0.0.1 --server.port 6006
```
## TODO
- [ ] 微调模型
- [√] 用langchain重构代码