import json
import random

def generate_data():
    # 示例文章和问题
    articles = [
        {
            "text": "张伟在清华大学获得了电子工程学位，并且现在在华为公司工作。",
            "questions": [
                {"question": "张伟获得了什么学位？", "answer": "电子工程"},
                {"question": "张伟现在在哪家公司工作？", "answer": "华为公司"},
                {"question": "张伟在哪里获得了学位？", "answer": "清华大学"}
            ]
        },
        {
            "text": "王芳在复旦大学获得了法律学位，现在在一家律师事务所工作。",
            "questions": [
                {"question": "王芳获得了什么学位？", "answer": "法律"},
                {"question": "王芳现在在哪里工作？", "answer": "律师事务所"},
                {"question": "王芳在哪里获得了学位？", "answer": "复旦大学"}
            ]
        },
        {
            "text": "赵敏在北京读书，学习生物医学科学。",
            "questions": [
                {"question": "赵敏在学习什么？", "answer": "生物医学科学"},
                {"question": "赵敏在哪里读书？", "answer": "北京"}
            ]
        },
        {
            "text": "刘明在上海交通大学获得了化学学位，并且现在在某制药公司工作。",
            "questions": [
                {"question": "刘明获得了什么学位？", "answer": "化学"},
                {"question": "刘明现在在哪里工作？", "answer": "某制药公司"},
                {"question": "刘明在哪里获得了学位？", "answer": "上海交通大学"}
            ]
        },
        {
            "text": "陈娜在南开大学获得了历史学位，现任教于某大学。",
            "questions": [
                {"question": "陈娜获得了什么学位？", "answer": "历史"},
                {"question": "陈娜现在在哪个大学任教？", "answer": "某大学"},
                {"question": "陈娜在哪里获得了学位？", "answer": "南开大学"}
            ]
        },
        {
            "text": "李明在北京大学获得了数学学位，现在在一家科技公司担任数据分析师。",
            "questions": [
                {"question": "李明获得了什么学位？", "answer": "数学"},
                {"question": "李明现在在做什么工作？", "answer": "数据分析师"},
                {"question": "李明在哪里获得了学位？", "answer": "北京大学"}
            ]
        },
        {
            "text": "周静在西安交通大学获得了医学博士学位，目前在一家医院工作。",
            "questions": [
                {"question": "周静获得了什么学位？", "answer": "医学博士"},
                {"question": "周静现在在哪里工作？", "answer": "医院"},
                {"question": "周静在哪里获得了学位？", "answer": "西安交通大学"}
            ]
        },
        {
            "text": "王军在中山大学获得了环境工程学位，目前在环保公司担任项目经理。",
            "questions": [
                {"question": "王军获得了什么学位？", "answer": "环境工程"},
                {"question": "王军现在在做什么工作？", "answer": "项目经理"},
                {"question": "王军在哪里获得了学位？", "answer": "中山大学"}
            ]
        },
        {
            "text": "刘晓在浙江大学获得了生物科技学位，现在在制药公司工作。",
            "questions": [
                {"question": "刘晓获得了什么学位？", "answer": "生物科技"},
                {"question": "刘晓现在在哪里工作？", "answer": "制药公司"},
                {"question": "刘晓在哪里获得了学位？", "answer": "浙江大学"}
            ]
        },
        {
            "text": "孙莉在东北大学获得了材料科学学位，现在在科技公司从事研发工作。",
            "questions": [
                {"question": "孙莉获得了什么学位？", "answer": "材料科学"},
                {"question": "孙莉现在从事什么工作？", "answer": "研发工作"},
                {"question": "孙莉在哪里获得了学位？", "answer": "东北大学"}
            ]
        },
        {
            "text": "吴敏在南昌大学获得了法律学位，现任职于一家大型法律事务所。",
            "questions": [
                {"question": "吴敏获得了什么学位？", "answer": "法律"},
                {"question": "吴敏现在在哪家公司工作？", "answer": "法律事务所"},
                {"question": "吴敏在哪里获得了学位？", "answer": "南昌大学"}
            ]
        }
    ]

    data = []
    
    for _ in range(200):
        article_entry = random.choice(articles)
        article_text = article_entry["text"]
        question_entry = random.choice(article_entry["questions"])
        question = question_entry["question"]
        reference_answer = question_entry["answer"]
        user_answer = random.choice([reference_answer, "错误答案"])  # 50%概率正确
        
        output = "正确" if user_answer == reference_answer else "错误"
        
        entry = {
            "input": f"# 任务描述\n假设你是一个答案评估助手，你的任务是根据文章、问题、参考答案和用户答案来判断用户的答案是否正确。\n\n# 任务要求\n请根据文章、问题和参考答案判断用户的答案是否正确，并以json格式返回结果。\n\n# 样例\n输入：\n文章：\"李华在北京大学学习计算机科学，毕业于2022年。\"\n问题：\"李华在什么大学学习？\"\n参考答案：\"北京大学\"\n用户答案：\"北京大学\"\n输出：\"正确\"\n\n# 当前数据\n文章：\"{article_text}\"\n问题：\"{question}\"\n参考答案：\"{reference_answer}\"\n用户答案：\"{user_answer}\"\n\n# 任务重述\n请参考样例，根据文章、问题和参考答案判断用户的答案是否正确，并以json格式返回结果。",
            "output": f"\"{output}\""
        }
        data.append(entry)
    
    return data

def save_to_file(data, filename='data.json'):
    with open(filename, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=4)

# 生成数据并保存到文件
data = generate_data()
save_to_file(data)
