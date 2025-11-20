# RAG-llm 项目说明

本项目聚焦于缓解基于检索的生成（RAG/RALM）在检索到无关片段时的噪声干扰问题。核心思路是：在大模型推理过程中对注意力头的输出进行方向性“微调”，让生成更偏向与正确答案相关的方向，从而提高 EM/F1 等指标。

## 研究目标

- 在包含噪声文档的检索场景中，提升答案生成的精确度（EM）和重叠度（F1）。
- 分析并干预模型内部的注意力头激活，使其更可靠地区分正负样本。
- 系统评估不同干预方向、强度和选择策略对性能的影响。

## 方法概述

- 激活采集：在真实的 NQ（Natural Questions）检索输入下，记录每层每个注意力头的输出激活。
- 探针训练：为每个 `(layer, head)` 训练二分类逻辑回归探针，衡量该头对区分正/负样本的能力（验证准确率）。
- 干预方向：默认采用“质心均值偏移”（Center-of-Mass, CoM），即正样本均值减负样本均值的方向；也支持随机方向或探针系数方向。
- 动态强度：在生成时根据当前激活的探针“动态分数”进行抑制，使用 `1 - sigmoid(w·x + b)` 作为强度因子；并乘以 `alpha × proj_val_std × (reliability^γ)`，其中 `reliability` 来自该头的验证准确率。

## 代码结构与主脚本

- `RAG/llama_get_activations.py`：采集激活，保存至 `RAG/features/`。
- `RAG/train_save_probes.py`：训练所有头的探针，保存探针、准确率矩阵，以及可选的 Top-k 头列表至 `RAG/results_dump/probes/`。
- `RAG/nq_generate_with_interventions.py`：主实验脚本。读取“所有头”的探针与准确率，可按需选择 Top-k；默认使用 CoM 方向并引入 `(1-动态分数)`。
- `RAG/nq_hparam_search.py`：超参数搜索。支持多种头选择策略、强度网格与分层强度映射；默认使用 CoM 方向与 `(1-动态分数)`。
- `RAG/nq_generate_random_and_fixed.py`：消融实验。包含随机方向与固定强度两路；已加入 `(1-动态分数)` 并支持 CoM 方向的固定强度设置。

## 数据与准备

1. 准备模型权重（示例路径）：
   - Llama-2-7b-chat: `/root/shared-nvme/RAG-llm/models/Llama-2-7b-chat-hf`
   - Llama-3.1-8B-Instruct: `/root/shared-nvme/RAG-llm/models/Llama-3.1-8B-Instruct`

2. 采集 NQ 激活与标签：
   - 将 `RAG/data/llama_2_chat_7b_train.jsonl` 或自定义 NQ 格式数据放置到合适位置。
   - 运行示例（使用 chat 模板）：
     - `python RAG/llama_get_activations.py --model_name llama2_chat_7B --dataset_name nq --use_chat_template`
   - 产物：`RAG/features/{model}_nq_head_wise.npy`、`RAG/features/{model}_nq_labels.npy`、可选 `tokens.pkl`。

## 训练探针

- 基本命令：
  - `python RAG/train_save_probes.py --model_name llama2_chat_7B --num_heads 32 --head_dim 128 --top_k 48`
- 输出目录：`RAG/results_dump/probes/`
  - `*_probes.pkl`：所有头的探针（长度 = L×H）。
  - `*_val_accs.npy`：验证准确率矩阵，形状 `(L, H)`。
  - `*_top_heads.pkl`：Top-k 头列表（可选）。

## 主实验（干预生成）

- 读取“所有头”探针与准确率，按需选择 Top-k（或全部）：
  - `python RAG/nq_generate_with_interventions.py \
     --model_name llama2_chat_7B \
     --dataset_path RAG/data/test.jsonl \
     --use_chat_template \
     --probes_path RAG/results_dump/probes/llama2_chat_7B_nq_seed_42_top_1024_folds_2_probes.pkl \
     --val_accs_path RAG/results_dump/probes/llama2_chat_7B_nq_seed_42_top_1024_folds_2_val_accs.npy \
     --tuning_headwise_path RAG/features/llama2_chat_7B_nq_head_wise.npy \
     --tuning_labels_path RAG/features/llama2_chat_7B_nq_labels.npy \
     --select_top_k 1024 \
     --alpha 5 --pf_gamma 1.0`

- 输出：
  - `results_dump/main/.../answer_dump/*.jsonl`
  - `results_dump/main/.../summary_dump/*.json`

## 超参数搜索

- 示例命令：
  - `python RAG/nq_hparam_search.py \
     --model_name llama2_chat_7B \
     --dataset_path RAG/data/test.jsonl \
     --use_chat_template \
     --probes_path RAG/results_dump/probes/llama2_chat_7B_nq_seed_42_top_1024_folds_2_probes.pkl \
     --val_accs_path RAG/results_dump/probes/llama2_chat_7B_nq_seed_42_top_1024_folds_2_val_accs.npy \
     --tuning_headwise_path RAG/features/llama2_chat_7B_nq_head_wise.npy \
     --tuning_labels_path RAG/features/llama2_chat_7B_nq_labels.npy \
     --scores_csv RAG/results_dump/probes/accs_csv.csv \
     --alphas range:1:19:2 --probe_factor_modes true --max_new_tokens 256`

- 产物：`results_dump/llama-2-7b-instruct-unified/` 下的逐次 summary 与最终汇总 CSV。

## 消融实验（随机/固定）

- 示例命令：
  - `python RAG/nq_generate_random_and_fixed.py \
     --model_name llama2_chat_7B \
     --dataset_path RAG/data/test.jsonl \
     --use_chat_template \
     --top_heads_path RAG/results_dump/probes/<your>_top_heads.pkl \
     --probes_path RAG/results_dump/probes/<your>_probes.pkl \
     --val_accs_path RAG/results_dump/probes/<your>_val_accs.npy \
     --tuning_headwise_path RAG/features/llama2_chat_7B_nq_head_wise.npy \
     --tuning_labels_path RAG/features/llama2_chat_7B_nq_labels.npy \
     --alpha 15 --pf_gamma 1.0`

- 两路设置：
  - 随机方向：`use_random_dir=True`，仍使用 `(1-动态分数)` 抑制与 `reliability^γ`。
  - 固定强度：改为 CoM 方向；使用 `(1-动态分数)` 与 `reliability^γ`。

## 评估指标

- `evaluate_nq_em_f1`：按全部 gold 答案进行 EM/F1 计算，脚本在生成结束后自动保存指标文件。

## 环境要求

- Ubuntu 20.04，Python 3.10+ 建议。
- 安装依赖：Transformers、baukit、datasets、numpy、scikit-learn、einops、tqdm、torch（支持 bfloat16/float16）。
- GPU 环境（device_map='auto'）。

## 注意事项

- 路径命名中 `{model}` 必须与实际采集/训练时一致，确保 `head_wise.npy` 与 `labels.npy` 对应同一模型与数据集。
- `--num_heads` 与 `--head_dim` 必须与模型配置一致（如 Llama2-7B chat 为 32×128）。
- 干预强度的三个乘子：`alpha`、`proj_val_std`、`reliability^γ`；以及抑制因子 `(1-动态分数)`。
- 若出现空类样本导致 CoM 方向为零向量，代码会回退为零向量，不影响运行但建议检查数据覆盖。