# 向量模型下载
from modelscope import snapshot_download
model_dir = snapshot_download("AI-ModelScope/bge-small-zh-v1.5", cache_dir='.')
# 源大模型下载
model_dir = snapshot_download('IEITYuan/Yuan2-2B-Mars-hf', cache_dir='.')
