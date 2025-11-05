# -*- coding: utf-8 -*-

import json
import os
import random
import requests
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
import torch
from zai import ZhipuAiClient
from dotenv import load_dotenv

# 加载环境变量
load_dotenv()

def select_random_passages(positive_passages, negative_passages, random_seed=2025):
    """
    从positive_passages的第二个片段和negative_passages的前两个片段中随机选择2个，
    加上positive_passages的第一个片段，总共3个片段。
    
    :param positive_passages: 正向片段列表
    :param negative_passages: 负向片段列表
    :param random_seed: 随机种子
    :return: 选择的3个片段列表
    """
    random.seed(random_seed)
    
    # 获取positive_passages的第一个片段（必选）
    first_positive = positive_passages[0]
    
    # 候选片段：positive_passages的第二个 + negative_passages的前两个
    candidates = []
    if len(positive_passages) > 1:
        candidates.append(positive_passages[1])
    if len(negative_passages) >= 2:
        candidates.extend(negative_passages[:2])
    
    # 从候选片段中随机选择2个
    selected_candidates = random.sample(candidates, min(2, len(candidates)))
    
    # 组合结果：第一个positive + 随机选择的2个
    result = [first_positive] + selected_candidates
    
    return result

def generate_llama_answer(query, passages, pipe):
    """
    使用Llama模型生成回答
    
    :param query: 问题
    :param passages: 检索到的片段列表
    :param pipe: transformers pipeline对象
    :return: 生成的回答
    """
    # 构建参考文档
    reference_text = "\n\n".join([f"Document {i+1}: {passage['text']}" for i, passage in enumerate(passages)])
    
    # 构建提示词
    system_prompt = ("Answer the question based on the given document. "
                    "Provide only the most direct and concise answer. Do not include explanations, full sentences, or additional context. "
                    "Just give the key information that directly answers the question.\n\n"
                    "Example:\n"
                    "Question: Where do the Great Lakes meet the ocean?\n"
                    "Answer: the Saint Lawrence River\n\n"
                    f"The following are given documents.\n\n{reference_text}")
    
    user_prompt = f"Question: {query}\nAnswer:"
    
    # 构建 Llama 3.1 的消息列表
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ]
    
    # Generation arguments
    generation_args = {
        "max_new_tokens": 256,       # 限制答案长度
        "do_sample": False,          # 使用 Greedy decoding (非采样)
        "pad_token_id": pipe.tokenizer.eos_token_id  # 消除关于 pad_token_id 的警告
    }
    
    # 调用 pipeline
    outputs = pipe(messages, **generation_args)
    
    # 解析结果
    # 当 pipeline 接收 list (messages) 时，它会返回一个 list
    # `outputs[0]["generated_text"]` 是一个包含所有对话历史的新列表
    # 我们需要的是最后一条，即 "assistant" 的 "content"
    assistant_reply_content = outputs[0]["generated_text"][-1]["content"]
    answer = assistant_reply_content.strip()
    
    return answer

def check_answer_correctness_with_glm(llama_answer, correct_answers, api_key=None):
    """
    使用GLM API判断Llama回答是否正确
    
    :param llama_answer: Llama生成的回答
    :param correct_answers: 正确答案列表
    :param api_key: GLM API密钥
    :return: 判断结果 (True/False)
    """
    try:
        # 从环境变量获取API密钥
        if api_key is None:
            api_key = os.getenv('GLM_API_KEY')
            if api_key is None:
                raise ValueError("GLM API密钥未设置，请在.env文件中设置GLM_API_KEY或传入api_key参数")
        
        # 创建智谱AI客户端
        client = ZhipuAiClient(api_key=api_key)
        
        # 构建判断提示词
        prompt = f"""
Please determine whether the following answer matches the standard answers.

Standard answers: {correct_answers}
Answer to be judged: {llama_answer}

Judgment criteria:
1. If the answer to be judged contains any of the standard answers, it is considered correct
2. Ignore case differences
3. Ignore punctuation differences
4. Consider synonyms and abbreviations

Please return your response in the following JSON format:
{{"answer": "true"}} or {{"answer": "false"}}

Only return the JSON object, no additional text.
"""
        
        # 调用GLM API
        response = client.chat.completions.create(
            model="glm-4.5-flash",
            messages=[
                {"role": "user", "content": prompt}
            ],
            thinking={
                "type": "disabled",    # 禁用深度思考模式
            },
            stream=False,             # 不使用流式输出，便于解析结果
            max_tokens=128,          # 最大输出tokens
            temperature=0.1,           # 低温度确保输出稳定
            response_format={
                "type": "json_object"
            }
        )
        
        # 获取回复内容
        result_content = json.loads(response.choices[0].message.content)
        
        # 解析JSON格式的结果
        if isinstance(result_content, dict) and "answer" in result_content:
            answer_value = result_content["answer"].lower()
            if answer_value == "true":
                return True
            elif answer_value == "false":
                return False
        else:
            # 如果返回结果不是预期的true/false，使用简单的字符串匹配作为备选方案
            print(f"GLM返回结果格式异常，使用备选方案: {result_content}")
            llama_lower = llama_answer.lower()
            for answer in correct_answers:
                if answer.lower() in llama_lower:
                    return True
            return False
            
    except Exception as e:
        print(f"GLM API调用错误: {e}")
        # 发生错误时使用简单的字符串匹配作为备选方案
        try:
            llama_lower = llama_answer.lower()
            for answer in correct_answers:
                if answer.lower() in llama_lower:
                    return True
            return False
        except:
            return False

def generate_new_dataset(input_file, output_file, num_samples=10, random_seed=2025):
    """
    生成新的数据集
    
    :param input_file: 输入文件路径
    :param output_file: 输出文件路径
    :param num_samples: 处理的样本数量
    :param random_seed: 随机种子
    """
    print("正在加载Llama模型...")
    
    # 使用pipeline加载模型
    model_path = "/root/autodl-tmp/RAG-llm/models/Llama-3.1-8B-Instruct"
    
    # 自动选择设备 (GPU 优先)
    device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
    # 使用 bfloat16 提高性能并减少显存占用
    torch_dtype = torch.bfloat16
    
    try:
        # 初始化 text-generation pipeline
        pipe = pipeline(
            "text-generation",
            model=model_path,
            model_kwargs={"dtype": torch_dtype},  # 加载模型时使用 bfloat16
            device=device,
        )
        print("Llama模型加载成功！")
    except Exception as e:
        print(f"模型加载失败: {e}")
        return
    
    new_dataset = []
    new_id = 0
    
    print(f"开始处理 {num_samples} 条数据...")
    
    try:
        with open(input_file, 'r', encoding='utf-8') as f:
            for i, line in enumerate(f):
                if i >= num_samples:
                    break
                
                print(f"处理第 {i+1} 条数据...")
                
                # 解析JSON数据
                data = json.loads(line.strip())
                query = data['query']
                query_id_old = data['query_id']
                answers = data['answers']
                positive_passages = data['positive_passages']
                negative_passages = data['negative_passages']
                
                # 随机选择片段
                selected_passages = select_random_passages(
                    positive_passages, negative_passages, random_seed
                )
                
                # 使用Llama生成回答
                llama_answer = generate_llama_answer(query, selected_passages, pipe)
                
                # 预处理llama_answer：去掉英语句号
                processed_answer = llama_answer.rstrip('.')
                
                # 使用GLM判断回答正确性
                is_correct = check_answer_correctness_with_glm(processed_answer, answers)
                
                # 只保留错误的回答
                if not is_correct:
                    new_entry = {
                        "query_id_new": new_id,
                        "query_id_old": query_id_old,
                        "query": query,
                        "answers": answers,
                        "wrong_answer": llama_answer,
                        "retrieve_snippets": selected_passages
                    }
                    new_dataset.append(new_entry)
                    new_id += 1
                    print(f"  -> 发现错误回答，已添加到新数据集 (当前总数: {len(new_dataset)})")
                    
                    # 检查是否达到600条数据限制
                    if len(new_dataset) >= 600:
                        print(f"已达到600条数据限制，停止处理")
                        break
                else:
                    print(f"  -> 回答正确，跳过")
    
    except Exception as e:
        print(f"处理数据时发生错误: {e}")
        return
    
    # 保存新数据集
    try:
        with open(output_file, 'w', encoding='utf-8') as f:
            for entry in new_dataset:
                f.write(json.dumps(entry, ensure_ascii=False) + '\n')
        
        print(f"新数据集生成完成！")
        print(f"原始数据: {num_samples} 条")
        print(f"错误回答: {len(new_dataset)} 条")
        print(f"输出文件: {output_file}")
        
    except Exception as e:
        print(f"保存文件时发生错误: {e}")

# --- 主程序 ---
if __name__ == "__main__":
    # 新功能：生成新数据集
    input_file = '/root/autodl-tmp/RAG-llm/RAG-ACT/data/train.jsonl'
    output_file = '/root/autodl-tmp/RAG-llm/RAG-ACT/data/new_dataset.jsonl'
    
    # 先用10条数据进行测试
    generate_new_dataset(input_file, output_file, num_samples=3000, random_seed=2025)