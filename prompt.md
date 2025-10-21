data_process.py
我需要在代码上续写一段功能，目标是生成一个新的数据集。
1. 因为数据文件比较大，我拿出了10条作为数据参考，位置在RAG-ACT/data/train-example.jsonl
2. 我希望在positive_passages的第二个片段和negative_passages的前两个片段，这三个片段中随机选两条，加上positive_passages的第一条，总共三条片段来模拟RAG检索到的片段，随机种子为2025
3. 我希望调用models/Llama-3.1-8B-Instruct模型采用贪心解码去生成回答，然后调用GLM大模型API去判断该回答在不在数据集中给出的answer中，只取llama回答错误的条目作为新数据集的条目
4. 新的数据集标签为query_id_new（id为新数据集的id），query_id_old，query，answers，wrong_answer，retrieve_snippets
5. llama的提示词参照prompt.py这段代码，GLM的提示词有你自己设计，务必格式化输出，便于构造数据集
6. GLM的API key用占位符代替，我将自己进行修改
7. 可以先选择10条进行测试，由我检查过之后，没问题的话，可以大批量生成