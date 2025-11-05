### 项目理解

- 现有流程围绕 LLaMA 收集层和注意力头的激活，并基于每个注意力头训练逻辑回归探针，挑选验证集表现最佳的前 top-k 头进行干预，从而提升模型对“正确/错误答案”的区分能力。
- 干预通过在 self_attn.head_out 上加一个方向向量（例如探针的 coef_ ），方向强度按该方向上的激活标准差进行调制，使模型的输出更贴近“正确”的方向。
- 评估侧重 TruthfulQA（MC、judge/info、CE/KL），而你希望把这一套迁移到 RAG/NQ，强化“证据遵循”，并在 NQ 的 EM/F1 指标上体现收益。

### 数据集格式（NQ）

- RAG/new_dataset.jsonl 每行包含： query ， answers （正确答案列表）， wrong_answer （一个错误答案）， retrieve_snippets （包含若干片段对象，每个至少有 text 字段）。
- 我已查看了首条样本，字段符合预期，例如 answers 是列表， retrieve_snippets 有若干 text 段落。

### 改动与新增

- 在 RAG/llama_utils.py 新增：
  - tokenized_nq_with_docs_dual(...) 按照 collect_activations.py 的思路构造系统+用户+助手的聊天输入，针对每题生成两条样本：正确答案（label=1）与错误答案（label=0），并可选使用 tokenizer.apply_chat_template 。
  - get_separated_activations_nq(...) 将样本按“每题两条（正/负）”进行分组拆分，供探针训练使用。
  - normalize_answer(...) 、 compute_em_f1(...) 、 evaluate_nq_em_f1(...) ：NQ 常用的 EM/F1 评估（考虑整个正确答案列表），带有初学者友好的注释。
- 修改 RAG/llama_get_activations.py ：
  - 支持 --dataset_name nq ，从 RAG/new_dataset.jsonl 读取数据。
  - 参照 collect_activations.py 的提示词风格构造系统+用户+助手输入，保证教师强制（teacher forcing）和“只输出直接答案”的证据遵循提示。
  - 统一保存位置到 ../features/{model_name}_nq_* ，同时保存 categories/tokens 便于后续分析。
- 新增顶层脚本 nq_train_save_probes.py ：
  - 从 ../features/{model_name}_nq_head_wise.npy 和 ../features/{model_name}_nq_labels.npy 载入特征。
  - 自动按 NQ“每题两样本”规则分组，训练每个注意力头的 LR 探针，按验证集准确率选前 top_k 。
  - 保存探针列表、 (layer, head) 的 top-k 头、验证准确率数组到 ./results_dump/probes/ ，便于后续复现实验与干预。

### 如何收集 NQ 激活

- 先准备模型（需要本地可用的 HF 权重），示例以 LLaMA-2-7B-Chat：
- 在 RAG 目录运行：
  - python llama_get_activations.py --model_name llama2_chat_7B --dataset_name nq --nq_jsonl ./new_dataset.jsonl --use_chat_template
- 输出文件：
  - ../features/llama2_chat_7B_nq_labels.npy
  - ../features/llama2_chat_7B_nq_head_wise.npy （形状约为 B × L × (H*D) ，其中 D=128 ）
  - ../features/llama2_chat_7B_nq_categories.pkl （全为 NQ ）
  - ../features/llama2_chat_7B_nq_tokens.pkl （用于辅助分析）

### 如何训练并保存前 top-k 探针

- 在项目根目录运行：
  - python nq_train_save_probes.py --model_name llama2_chat_7B --top_k 48 --seed 42 --val_ratio 0.2
- 输出文件位于 ./results_dump/probes/ ：
  - llama2_chat_7B_nq_seed_42_top_48_probes.pkl （全部逻辑回归探针）
  - llama2_chat_7B_nq_seed_42_top_48_top_heads.pkl （top-k 的 (layer, head) 列表）
  - llama2_chat_7B_nq_seed_42_top_48_val_accs.npy （验证准确率数组）
EM/F1 评估用法

- 你在生成阶段得到 predictions （字符串列表），同时从 new_dataset.jsonl 收集每条样本的 answers 列表作为 gold_answers_list ：
  - 调用 evaluate_nq_em_f1(predictions, gold_answers_list) 返回整体平均 EM 和 F1。
- 指标细节：
  - EM：规范化后字符串完全匹配（对某个正确答案）则 1，否则 0。
  - F1：基于 token 的最大 F1，考虑整个正确答案列表。

### 提示词与证据遵循（关键设计）

- 系统提示词明确要求“只输出直接答案，不要解释”，并在系统消息中附带检索片段 Document i: ... ，提高模型关注检索证据的概率。
- 用户提示词固定结构 Question: ...\nAnswer: ，通过教师强制将候选答案作为助手消息；收集激活时取最后一个 token 的激活，直接对齐“最终答案”的表示。

### 下一步建议（将干预应用到生成阶段）

- 已保存的 top-k 探针可与现有 llama_validate_2fold.py 里的干预函数（ lt_modulated_vector_add ）组合使用：对 NQ 生成脚本，在 TraceDict 钩住 model.layers.{layer}.self_attn.head_out ，按 (layer, head) 和探针方向 coef_ 添加方向向量，强度用 proj_val_std * alpha 调制。
- 你可以基于当前的 NQ提示构造，写一个 nq_generate_with_interventions.py ：
  - 载入 top-k 探针与头列表；
  - 从 ../features/{model_name}_nq_head_wise.npy 里抽取一部分样本的激活做 proj_val_std （或复用 get_interventions_dict 的逻辑）；
  - 生成答案，最后用 evaluate_nq_em_f1 报告指标，与不干预的生成结果对比
  
### 注意事项

- 模型与分词器： tokenizer.apply_chat_template 仅在 Instruct/Chat 模型可用，代码已做回退；建议优先用 Chat 版 LLaMA（如 LLaMA-2-7B-Chat、LLaMA-3.1-8B-Instruct）。
- 资源与路径：大模型下载与推理成本较高， --nq_max_samples 与 --nq_max_docs 可用于加快开发调试。
- 维度一致性： head_dim 默认 128，如你的模型配置不同需在 nq_train_save_probes.py 指定 --num_heads 或调整 --head_dim 保证 (H*D) 匹配。

### 代码内注释

- 新增/修改的函数与脚本都已添加“简单易懂”的中文注释，重点解释了每步的作用、输入输出与设计理由，便于上手。

---
1. nq_train_save_probes.py 是否需要二折交叉验证？

- 我已在 RAG/nq_train_save_probes.py 增加了可选的 K 折交叉验证参数 --num_fold （默认 2）。当 --num_fold > 1 时，脚本会：
  - 将问题索引分成 K 折，轮流用 K-1 折训练、1 折验证；
  - 记录每折的验证准确率，对所有注意力头的准确率做“折均值”，以折均后的分数来筛选 top_k 头；
  - 探针对象列表在第一次训练时记录一次（用于保存和后续干预方向），因为我们主要用验证准确率来选头，探针训练不跨折合并。
- 当 --num_fold 1 或不指定时，脚本只做一次随机划分（按 --val_ratio ），这相当于“无交叉验证”的简单基线。
- 使用示例：
  - 单次划分： python RAG/nq_train_save_probes.py --model_name llama2_chat_7B --top_k 48 --seed 42
  - 二折交叉： python RAG/nq_train_save_probes.py --model_name llama2_chat_7B --top_k 48 --seed 42 --num_fold 2
- 输出保持一致：
  - ./results_dump/probes/{model}_nq_seed_{seed}_top_{k}[_folds_{K}]_probes.pkl （探针列表）
  - ./results_dump/probes/{model}_nq_seed_{seed}_top_{k}[_folds_{K}]_top_heads.pkl （top-k 头列表；元素是 (layer, head) ）
  - ./results_dump/probes/{model}_nq_seed_{seed}_top_{k}[_folds_{K}]_val_accs.npy （验证准确率矩阵，形状 (L, H) ；在 K 折时为折均值）
2. 生成阶段干预时，强度需要在原基础上添加“探针分数因子”

- 我已在 RAG/llama_utils.py 的干预构造函数中加入探针分数因子（probe_factor），并修改了 RAG/llama_validate_2fold.py 的生成干预函数以乘上该因子：
  - get_top_heads(...) 现在返回 (top_heads, probes, all_head_accs_np) ，其中 all_head_accs_np 是 (num_layers, num_heads) 的验证准确率矩阵。
  - get_interventions_dict(...) 新增参数 probe_score_map ，若提供将把对应 (layer, head) 的验证准确率作为 probe_factor 注入到干预元组中，元组从 (head, direction, proj_val_std) 扩展为 (head, direction, proj_val_std, probe_factor) 。
  - 在生成阶段的 lt_modulated_vector_add(...) 里，强度从 alpha * proj_val_std * direction 调整为 alpha * proj_val_std * probe_factor * direction ，使更可靠的探针（高验证准确率）贡献更大的干预强度。
- 这满足你“在原本强度基础上添加探针分数因子”的要求，同时保持可回退（若未传 probe_score_map 则 probe_factor=1.0 ）。

### 运行顺序建议

- 收集 NQ 激活：
  - python RAG/llama_get_activations.py --model_name llama2_chat_7B --dataset_name nq --nq_jsonl ./RAG/new_dataset.jsonl --use_chat_template
- 训练并保存 top-k 探针（可选 K 折）：
  - python RAG/nq_train_save_probes.py --model_name llama2_chat_7B --top_k 48 --seed 42 --num_fold 2
- 进行生成阶段干预与评估（如基于 llama_validate_2fold.py 思路或你后续的 NQ 生成脚本）：
  - 载入 top_heads 与 probes ；
  - 利用 get_interventions_dict(..., probe_score_map=all_head_accs_np) 构造干预字典；
  - 在生成时使用更新后的 lt_modulated_vector_add 将强度调制为 alpha * proj_val_std * probe_factor 。

### 设计理由与注意

- 交叉验证提升选头的稳健性，缓解不同划分导致的波动，符合你希望减少“检索片段包含答案但LLM生成错误”的现象，避免对噪声片段过拟合。
- 探针分数因子让高质量探针对模型的影响更强，鼓励模型沿“可信方向”调整，有助于提高证据遵循。
- 如果未来你希望进一步控制强度，可改为 probe_factor^γ 或归一化到 [min,max] 区间；当前实现直接使用验证准确率（0~1）是简单有效的初始策略。
如果你需要我将生成阶段的 NQ 推理脚本也补全（例如 nq_generate_with_interventions.py ，支持EM/F1评估），我可以继续添加，并给出一条命令即可运行的范例。

---
新增脚本

- 已添加 RAG/nq_generate_with_interventions.py ，实现加载探针与 top-k 头、构造干预并在生成时应用，最后计算 NQ 的 EM/F1。
脚本功能

- 加载模型与分词器： llama2_chat_7B 或 llama3_8B （需本地权重或能从 HF 拉取）。
- 构造生成输入：系统 + 用户（不含助手答案），系统提示强调“仅输出直接答案”，系统消息中包含检索片段 Document i: ... ，提升证据遵循。
- 构造干预：基于已保存的探针 probes.pkl 、top-k 头 top_heads.pkl 、验证准确率矩阵 val_accs.npy ，使用验证准确率作为“探针分数因子”来调制干预强度。
- 生成与后处理：拦截每层的 self_attn.head_out ，在最后一个 token 的头输出加方向向量，生成答案后做简单截断/抽取。
- 评估：计算 NQ 的 EM/F1（考虑整个正确答案列表），保存逐条预测和汇总指标。
干预强度

- 强度公式为： alpha * proj_val_std * probe_factor * direction
  - alpha ：手动设定的系数（脚本参数）。
  - proj_val_std ：沿干预方向的激活标准差（从调优激活数据计算）。
  - probe_factor ：探针分数因子，取验证准确率（越高说明该头更可靠）。
- 已在生成阶段钩子中使用该因子，满足“在原强度基础上添加探针分数因子”的要求。
使用示例

- 先收集 NQ 激活（已在 RAG/llama_get_activations.py 增加对 NQ 的支持）：
  - python RAG/llama_get_activations.py --model_name llama2_chat_7B --dataset_name nq --nq_jsonl RAG/new_dataset.jsonl --use_chat_template
- 训练并保存 top-k 探针（可选 K 折，推荐 2 折以提升稳健性）：
  - python RAG/nq_train_save_probes.py --model_name llama2_chat_7B --top_k 48 --seed 42 --num_fold 2
- 生成并评估（带干预）：
  - python RAG/nq_generate_with_interventions.py --model_name llama2_chat_7B --dataset_path RAG/new_dataset.jsonl --use_chat_template --probes_path ./results_dump/probes/llama2_chat_7B_nq_seed_42_top_48_folds_2_probes.pkl --top_heads_path ./results_dump/probes/llama2_chat_7B_nq_seed_42_top_48_folds_2_top_heads.pkl --val_accs_path ./results_dump/probes/llama2_chat_7B_nq_seed_42_top_48_folds_2_val_accs.npy --tuning_headwise_path ../features/llama2_chat_7B_nq_head_wise.npy --tuning_labels_path ../features/llama2_chat_7B_nq_labels.npy --alpha 15 --save_answers_path ./results_dump/answer_dump/nq_gen_answers.jsonl --save_summary_path ./results_dump/summary_dump/nq_gen_summary.json
参数说明

- --dataset_path ：NQ jsonl 路径（默认 RAG/new_dataset.jsonl ）。
- --use_chat_template ：使用 tokenizer.apply_chat_template 构造 system+user 输入。
- --max_samples 、 --max_docs ：限制样本数和每条样本的检索片段数，便于快速调试。
- --alpha ：干预强度系数（默认 15.0 ）。
- --probes_path 、 --top_heads_path 、 --val_accs_path ：从训练脚本保存的路径； val_accs_path 为 (L, H) 的验证准确率矩阵，作为干预的探针分数因子。
- --tuning_headwise_path 、 --tuning_labels_path ：用于计算 proj_val_std 的激活（默认使用 NQ 收集到的 head-wise 激活），脚本会将 (B, L, H*D) reshape 为 (B, L, H, D) 。
- --save_answers_path 、 --save_summary_path ：输出位置，分别保存逐条预测和汇总指标。
环境与依赖

- 需安装 transformers 、 baukit 、 datasets 、 tqdm 、 einops 、 scikit-learn 。
- baukit 的 TraceDict 已在脚本中导入用于拦截头输出。
- 分词器填充符已设置为 eos_token ，避免生成报错。
与“降低噪声影响”的关系

- 系统提示词强约束“只输出直接答案”，减少模型啰嗦和偏离检索证据。
- 探针驱动的头干预通过强化“正确答案方向”来减少“检索片段含答案但模型生成错误”的情况。
- 探针分数因子提升可靠头的影响力，进一步抑制不可靠头带来的干扰。
代码注释

- 新增脚本与修改处均包含“对代码小白友好”的中文注释，解释每步的目的、输入输出和设计理由，便于快速上手与扩展。
如需我加一版“无干预对照生成”脚本，或在该脚本中增加“无干预开关+对比输出”，我可以继续补充，方便你直接对比干预前后的 EM/F1 提升幅度。