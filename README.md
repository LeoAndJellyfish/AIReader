# Yuan2.0 AIReader（在建）
AIReader 是一个基于Yuan2.0的名著阅读助手，它可以帮助用户快速阅读名著，并回答相关问题。
---
## 环境准备  
需要modelscope，torch，transformers，streamlit，sentencepiece，protobuf，einops等依赖
```Shell
# 查看已安装依赖,包括modelscope，torch，transformers等
pip list
# 安装 streamlit
pip install streamlit==1.24.0
```
---

## 启动服务
### 方法一：
首先运行code\Dmodle.py下载向量模型与源大模型  
然后运行
```Shell
streamlit run code/main.py --server.address 127.0.0.1 --server.port 6006
```

### 方法二：（自动化一键启动）
双击打开code\start.bat文件，即可启动服务

## TODO
- [ ] 微调模型
- [√] 增加自动化一键启动功能