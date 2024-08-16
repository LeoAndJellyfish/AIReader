# Yuan2.0 AIReader（在建）
AIReader 是一个基于Yuan2.0的名著阅读助手，它可以帮助用户快速阅读名著，并回答相关问题。

# 部署流程

## 获取代码
```Shell
git clone https://github.com/LeoAndJellyfish/AIReader.git
```

## 环境准备
```Shell
# 查看已安装依赖
pip list
# 安装环境
pip install --upgrade pip
pip install faiss-gpu streamlit==1.24.0 langchain langchain_community langchain_huggingface pypdf
```

## 启动服务
运行
```Shell
streamlit run app.py --server.address 127.0.0.1 --server.port 6006
```

# TODO
- [ ] 微调模型
- [x] 用langchain重构代码
- [ ] 改进概括算法实现
- [ ] 改进用户页面
- [ ] 关闭后删除临时文件

# 备注
这里的代码是gpu运行版本，创空间发布的是cpu版本。

# 链接
体验demo：https://www.modelscope.cn/studios/leo12QWER/AIReader

开发日志：https://zhuanlan.zhihu.com/p/713670859