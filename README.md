# AIExamGrader（在建）
AIExamGrader 是一个基于Yuan2.0的自动评分系统，旨在通过人工智能技术对考试卷和论文进行智能化评估。该系统能够分析和评分学生提交的作业，提供准确、公平的评分结果。支持多种类型的考试题目和评分标准，并且可以根据实际需求进行自定义配置。
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
streamlit run code\main.py
```

### 方法二：（自动化一键启动）（待实现）


## TODO
- [ ] 微调模型
- [ ] 增加自动化一键启动功能