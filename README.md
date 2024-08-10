# AIExamGrader
ModelGrader 是一个基于大模型的自动评分系统，旨在通过人工智能技术对考试卷和论文进行智能化评估。该系统能够分析和评分学生提交的作业，提供准确、公平的评分结果。支持多种类型的考试题目和评分标准，并且可以根据实际需求进行自定义配置。
---
环境准备
需要modelscope，torch，transformers，streamlit，sentencepiece，protobuf，einops等依赖
```Shell
# 查看已安装依赖,包括modelscope，torch，transformers等
pip list
# 安装 streamlit
pip install streamlit==1.24.0
```
---
# 向量模型和源大模型下载
```Python
# 向量模型下载
from modelscope import snapshot_download
model_dir = snapshot_download("AI-ModelScope/bge-small-zh-v1.5", cache_dir='.')
# 源大模型下载
from modelscope import snapshot_download
model_dir = snapshot_download('IEITYuan/Yuan2-2B-Mars-hf', cache_dir='.')
```
---
<!-- # 启动服务 -->
