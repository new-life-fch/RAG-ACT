import torch
from transformers import pipeline
import warnings

# 0. 抑制一些不必要的警告
warnings.filterwarnings("ignore")

# --- 1. 初始化模型和 Pipeline ---

# Llama 3.1-8B 指令微调版
model_id = "meta-llama/Meta-Llama-3.1-8B-Instruct"

# 自动选择设备 (GPU 优先)
device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
# 使用 bfloat16 提高性能并减少显存占用
torch_dtype = torch.bfloat16

print(f"正在加载模型 {model_id} 到 {device}...")

# 初始化 text-generation pipeline
# 这是使用 transformers 调用 LLM 最高效简洁的方式
pipe = pipeline(
    "text-generation",
    model=model_id,
    model_kwargs={"torch_dtype": torch_dtype}, # 加载模型时使用 bfloat16
    device=device,
)

print("模型加载完毕。")

# --- 2. 模拟 RAG 检索到的数据 ---
# (这里我们硬编码一些示例数据，实际应用中你将从向量数据库中检索)

passages = [
    {"text": "文档1指出：天空是蓝色的主要原因是瑞利散射（Rayleigh scattering）。当阳光穿过大气层时，空气中的氮分子和氧分子会将波长较短的蓝光向各个方向散射。"},
    {"text": "文档2补充：虽然所有颜色的光都会被散射，但蓝光和紫光比红光和黄光散射得更强烈。我们的眼睛对蓝光更敏感，所以我们看到天空是蓝色的。"},
    {"text": "文档3描述：日落时天空呈红色，是因为此时太阳光需要穿过更厚的大气层，大部分蓝光已被散射掉，只剩下波长较长的红光和橙光到达我们的眼睛。"}
]

# 模拟用户的提问
query = "天空为什么是蓝色的？"

# --- 3. 你的提示词构建逻辑 ---
# (完全复用你提供的代码)

# 构建参考文档
reference_text = "\n\n".join([f"Document {i+1}: {passage['text']}" for i, passage in enumerate(passages)])

# 构建提示词（基于FlashRAG模板）
# 注意：为了匹配中文文档，我将你的英文指令翻译成了中文，
# 如果你的文档是英文，请使用你原来的英文 System Prompt
system_prompt = (
    "请根据给定的文档回答问题。"
    "只提供一个答案，不要输出任何其他词语。"
    f"\n以下是给定的文档：\n\n{reference_text}"
)

# 你的 User Prompt
user_prompt = f"Question: {query}\nAnswer:"

# --- 4. 构建 Llama 3.1 的消息列表 ---

# transformers 的 pipeline 推荐使用 messages 列表
# 它会自动为你应用 Llama 3.1 特定的聊天模板 (如 <|begin_of_text|>, <|eot_id|> 等)
messages = [
    {"role": "system", "content": system_prompt},
    {"role": "user", "content": user_prompt},
]

# (可选) 打印查看最终的提示词结构
# print("--- System Prompt ---")
# print(system_prompt)
# print("\n--- User Prompt ---")
# print(user_prompt)

print("\n正在调用模型生成答案...")

# --- 5. 调用模型并生成答案 ---

# Generation arugments
generation_args = {
    "max_new_tokens": 256,       # 限制答案长度
    "do_sample": False,          # 使用 Greedy decoding (非采样)
                                 # 这对于 RAG 和你“只给一个答案”的指令非常重要
    "pad_token_id": pipe.tokenizer.eos_token_id  # 消除关于 pad_token_id 的警告
}

# 调用 pipeline
# 当输入是 messages 列表时，pipeline 会自动处理聊天模板
outputs = pipe(messages, **generation_args)

# --- 6. 解析并打印结果 ---

# 当 pipeline 接收 list (messages) 时，它会返回一个 list
# `outputs[0]["generated_text"]` 是一个包含所有对话历史的新列表
# 我们需要的是最后一条，即 "assistant" 的 "content"
assistant_reply_content = outputs[0]["generated_text"][-1]["content"]
answer = assistant_reply_content.strip()

print("\n--- 模型的最终回答 ---")
print(answer)

# --- 示例：测试另一个问题 ---

print("\n--- 测试第二个问题 (RAG) ---")
query_2 = "什么是日落时天空呈红色的原因？"
user_prompt_2 = f"Question: {query_2}\nAnswer:"

messages_2 = [
    {"role": "system", "content": system_prompt}, # 复用同一个 system_prompt (包含上下文)
    {"role": "user", "content": user_prompt_2},
]

outputs_2 = pipe(messages_2, **generation_args)
answer_2 = outputs_2[0]["generated_text"][-1]["content"].strip()

print("问题:", query_2)
print("回答:", answer_2)