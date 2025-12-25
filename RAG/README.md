# RAG-llm 项目说明

本项目聚焦于缓解基于检索的生成（RAG/RALM）在检索到无关片段时的噪声干扰问题。核心思路是：在大模型推理过程中对注意力头的输出进行方向性“微调”，让生成更偏向与正确答案相关的方向，从而提高 EM/F1 等指标。

## 研究目标

- 在包含噪声文档的检索场景中，提升答案生成的精确度（EM）和重叠度（F1）。
- 分析并干预模型内部的注意力头激活，使其更可靠地区分正负样本（证据样本和噪声样本）。
- 系统评估不同干预方向、强度和干预头的数量对性能的影响。

## 方法概述

- 激活采集：在真实的 PopQA 检索输入下，记录每层每个注意力头的输出激活。
- 探针训练：为每个 `(layer, head)` 训练二分类逻辑回归探针，衡量该头对区分正/负样本的能力（验证准确率）。
- 干预方向：默认采用“质心均值偏移”（Center-of-Mass, CoM），即正样本均值减负样本均值的方向；也支持随机方向或探针系数方向。

## 代码结构与主脚本

- `llama_get_activations.py`：采集激活，保存至 `RAG/features/`。
- `train_save_probes.py`：训练所有头的探针，保存探针、准确率矩阵，以及可选的 Top-k 头列表至 `RAG/probes/`。
- `generate.py`：支持选择不同数量的头，更换不同的强度进行干预；默认使用 CoM 方向。
- `rag_hparam_search_topk_alpha.py`:超参数搜索。给出干预头的网格列表和干预强度的网格列表进行超参数实验，得到最优参数。

## 数据与准备

1. 训练集构建：
   - `python RAG/utils/2_generate_dataset.py`

2. 测试集构建：
   - `python RAG/utils/3_extract_remaining_testset.py --input RAG/data/PopQA/PopQA_processed.jsonl --val-output RAG/data/PopQA/val_noise_test.jsonl --test-output RAG/data/PopQA/test_noise_test.jsonl --train-num 0 --val-num 0 --test-num 1190 --method random --seed 2025 --noise-levels 0,1,2,3,4,5`

3. 准备模型权重（示例路径）：
   - Llama-2-7b-chat: `RAG-llm/models/Llama-2-7b-chat-hf`
   - Llama-3-8B-Instruct: `RAG-llm/models/Llama-3-8B-Instruct`

4. 采集 PopQA 激活与标签：
   - 将 `RAG/data/PopQA/train_user.jsonl` 放置到合适位置。
   - 运行示例（使用 chat 模板）：
     - `python RAG/llama_get_activations.py --model_name llama2_chat_7B --dataset_name popqa --use_chat_template --nq_jsonl RAG/data/PopQA/train_user.jsonl --output_dir RAG/features/llama2_chat_7B_popqa_user_noise --use_noise_contrastive --extraction_point prompt_end`
   - 产物：`RAG/features/{model}_{dataset_name}_head_wise.npy`、`RAG/features/{model}_{dataset_name}_labels.npy`、可选 `tokens.pkl`。

## 训练探针

- 基本命令：
  - `python RAG/train_save_probes.py --model_name llama2_chat_7B --num_heads 32 --head_dim 128 --feat_dir RAG/features/llama2_chat_7B_popqa_user_noise --save_dir RAG/probes/llama2_chat_7B_popqa_user_noise --cv_final_train full`
- 输出目录：`RAG/results/probes/`
  - `*_probes.pkl`：所有头的探针（长度 = L×H）。
  - `*_val_accs.npy`：验证准确率矩阵，形状 `(L, H)`。
  - `*_top_heads.pkl`：Top-k 头列表（可选）。

## 获取探针的验证集分数

- 基本命令：
  - `python RAG/utils/inspect_probes.py RAG/probes/llama2_chat_7B_popqa_user_noise/llama2_chat_7B_popqa_seed_2025_top_1024_folds_3_val_accs.npy --out-csv RAG/probes/llama2_chat_7B_popqa_user_noise/accs_csv.csv`

## 验证集分数转化为热力图

- 基本命令：
  - `python RAG/utils/plot_probe_heatmap.py RAG/probes/llama2_chat_7B_popqa_user_noise/accs_csv.csv --output RAG/probes/llama2_chat_7B_popqa_user_noise/accs_heatmap.png --cmap turbo --bins 12 --discrete`

## 实验

- 示例命令：

NQ:

```bash
python RAG/generate.py \
--model_name llama2_chat_7B \
--dataset_path RAG/data/NQ/test_noise_test_noise4.jsonl \
--use_chat_template \
--probes_path RAG/probes/llama2_chat_7B_popqa_user_noise/llama2_chat_7B_popqa_seed_2025_top_1024_folds_3_probes.pkl \
--val_accs_path RAG/probes/llama2_chat_7B_popqa_user_noise/llama2_chat_7B_popqa_seed_2025_top_1024_folds_3_val_accs.npy \
--tuning_headwise_path RAG/features/llama2_chat_7B_popqa_user_noise/llama2_chat_7B_popqa_head_wise.npy \
--tuning_labels_path RAG/features/llama2_chat_7B_popqa_user_noise/llama2_chat_7B_popqa_labels.npy \
--scores_csv RAG/probes/llama2_chat_7B_popqa_user_noise/accs_csv.csv \
--alphas 7 --probe_factor_modes false --max_new_tokens 256 --include_strategies topk_35_by_score --results_root RAG/results/llama-2-chat-7b-nq-user_noise/topk_35_by_score_alphas_7 --sample_size 300 --timeout_minutes 10
```

Trivia QA:

```bash
python RAG/generate.py \
--model_name llama2_chat_7B \
--dataset_path RAG/data/TriviaQA/test_noise_test_noise4.jsonl \
--use_chat_template \
--probes_path RAG/probes/llama2_chat_7B_popqa_user_noise/llama2_chat_7B_popqa_seed_2025_top_1024_folds_3_probes.pkl \
--val_accs_path RAG/probes/llama2_chat_7B_popqa_user_noise/llama2_chat_7B_popqa_seed_2025_top_1024_folds_3_val_accs.npy \
--tuning_headwise_path RAG/features/llama2_chat_7B_popqa_user_noise/llama2_chat_7B_popqa_head_wise.npy \
--tuning_labels_path RAG/features/llama2_chat_7B_popqa_user_noise/llama2_chat_7B_popqa_labels.npy \
--scores_csv RAG/probes/llama2_chat_7B_popqa_user_noise/accs_csv.csv \
--alphas 7 --probe_factor_modes false --max_new_tokens 256 --include_strategies topk_35_by_score --results_root RAG/results/llama-2-chat-7b-triviaqa-user_noise/topk_35_by_score_alphas_7 --sample_size 300 --timeout_minutes 10
```

PopQA:

```bash
python RAG/generate.py \
--model_name llama2_chat_7B \
--dataset_path RAG/data/PopQA/test_noise_test_noise4.jsonl \
--use_chat_template \
--probes_path RAG/probes/llama2_chat_7B_popqa_user_noise/llama2_chat_7B_popqa_seed_2025_top_1024_folds_3_probes.pkl \
--val_accs_path RAG/probes/llama2_chat_7B_popqa_user_noise/llama2_chat_7B_popqa_seed_2025_top_1024_folds_3_val_accs.npy \
--tuning_headwise_path RAG/features/llama2_chat_7B_popqa_user_noise/llama2_chat_7B_popqa_head_wise.npy \
--tuning_labels_path RAG/features/llama2_chat_7B_popqa_user_noise/llama2_chat_7B_popqa_labels.npy \
--scores_csv RAG/probes/llama2_chat_7B_popqa_user_noise/accs_csv.csv \
--alphas 7 --probe_factor_modes false --max_new_tokens 256 --include_strategies topk_35_by_score --results_root RAG/results/llama-2-chat-7b-popqa-user_noise/topk_35_by_score_alphas_7 --sample_size 300 --timeout_minutes 10
```

- 产物：`results/llama-2-chat-7b-{dataset_name}-user_noise` 下的逐次 summary 与最终汇总 CSV。

## 超参实验（top-k和alpha）

```bash
python RAG/rag_hparam_search_topk_alpha.py \
--model_name llama2_chat_7B \
--dataset_path RAG/data/PopQA/test_noise_test_noise4.jsonl \
--use_chat_template \
--probes_path RAG/probes/llama2_chat_7B_popqa_user_noise/llama2_chat_7B_popqa_seed_2025_top_1024_folds_3_probes.pkl \
--val_accs_path RAG/probes/llama2_chat_7B_popqa_user_noise/llama2_chat_7B_popqa_seed_2025_top_1024_folds_3_val_accs.npy \
--tuning_headwise_path RAG/features/llama2_chat_7B_popqa_user_noise/llama2_chat_7B_popqa_head_wise.npy \
--tuning_labels_path RAG/features/llama2_chat_7B_popqa_user_noise/llama2_chat_7B_popqa_labels.npy \
--scores_csv RAG/probes/llama2_chat_7B_popqa_user_noise/accs_csv.csv \
--results_root RAG/results/llama2_alpha_search_popqa_noise \
--sample_size 300 \
--max_new_tokens 256
--timeout_minutes 10
```

## 因果追踪实验

```bash
python RAG/causal_trace_experiment.py \
--model_name llama2_chat_7B \
--dataset_path RAG/data/NQ/test_noise_test_noise4.jsonl \
--use_chat_template \
--probes_path RAG/probes/llama2_chat_7B_nq_user/llama2_chat_7B_nq_seed_2025_top_1024_folds_3_probes.pkl \
--val_accs_path RAG/probes/llama2_chat_7B_nq_user/llama2_chat_7B_nq_seed_2025_top_1024_folds_3_val_accs.npy \
--tuning_headwise_path RAG/features/llama2_chat_7B_nq_user/llama2_chat_7B_nq_head_wise.npy \
--tuning_labels_path RAG/features/llama2_chat_7B_nq_user/llama2_chat_7B_nq_labels.npy \
--alpha 5.0 --max_new_tokens 256 --output_csv RAG/results/llama-2-chat-7b-nq-user/causal_layer_trace_pf0.csv
```

### 前 k 层干预超参数搜索实验 (Based on Causal Trace)

- 示例命令：
```bash
python RAG/nq_layer_hparam_search.py \
--model_name llama2_chat_7B \
--dataset_path RAG/data/TriviaQA/test_noise_test_noise4.jsonl \
--use_chat_template \
--probes_path RAG/probes/llama2_chat_7B_nq_user/llama2_chat_7B_nq_seed_2025_top_1024_folds_3_probes.pkl \
--val_accs_path RAG/probes/llama2_chat_7B_nq_user/llama2_chat_7B_nq_seed_2025_top_1024_folds_3_val_accs.npy \
--tuning_headwise_path RAG/features/llama2_chat_7B_nq_user/llama2_chat_7B_nq_head_wise.npy \
--tuning_labels_path RAG/features/llama2_chat_7B_nq_user/llama2_chat_7B_nq_labels.npy \
--sample_size 100 \
--probe_factor_modes false \
--results_root RAG/results/llama-2-chat-7b-triviaqa-user/top-layers-intervention \
--summary_csv top_layer_intervention_results.csv \
--alphas 5 \
--max_new_tokens 256
```

### 细粒度超参数搜索实验
- 示例命令：
```bash
python RAG/nq_fine_grained_hparam_search.py \
--model_name llama2_chat_7B \
--dataset_path RAG/data/TriviaQA/test_noise_test_noise4.jsonl \
--use_chat_template \
--probes_path RAG/probes/llama2_chat_7B_nq_user/llama2_chat_7B_nq_seed_2025_top_1024_folds_3_probes.pkl \
--val_accs_path RAG/probes/llama2_chat_7B_nq_user/llama2_chat_7B_nq_seed_2025_top_1024_folds_3_val_accs.npy \
--tuning_headwise_path RAG/features/llama2_chat_7B_nq_user/llama2_chat_7B_nq_head_wise.npy \
--tuning_labels_path RAG/features/llama2_chat_7B_nq_user/llama2_chat_7B_nq_labels.npy \
--causal_trace_path RAG/results/llama-2-chat-7b-nq-user/causal_layer_trace_pf0.csv \
--head_scores_path RAG/probes/llama2_chat_7B_nq_user/accs_csv.csv \
--sample_size 100 \
--top_k_layers 8 \
--thresholds 0.75,0.7,0.65,0.6,0.55,0.48 \
--results_root RAG/results/llama-2-chat-7b-triviaqa-user/nq_fine_grained \
--summary_csv triviaqa_noise4_result_top_8_layers_score.csv \
--alphas 5 \
--max_new_tokens 256
```

## CoN
- 示例命令：
```bash
python RAG/con_rag.py --model_name llama2_chat_7B --dataset_path RAG/data/TriviaQA/test_noise_test_noise4.jsonl --use_chat_template --max_docs 5 --sample_size 300 --max_new_tokens 256 --results_root RAG/results/llama-2-chat-7b-triviaqa-user/CON_300
```

## naive LLM
- 示例命令：
```bash
python RAG/naive_llm.py \
--model_name llama2_chat_7B \
--dataset_path RAG/data/PopQA/test_noise_test_noise4.jsonl \
--use_chat_template \
--sample_size 300 \
--max_new_tokens 256 \
--results_root RAG/results/llama-2-chat-7b-popqa-user/naive-llm
```

## 评估指标

- `evaluate_rag_em_f1`：按全部 gold 答案进行 EM/F1 计算，脚本在生成结束后自动保存指标文件。

## 环境要求

- Ubuntu 20.04，Python 3.10+ 建议。
- 安装依赖：Transformers、baukit、datasets、numpy、scikit-learn、einops、tqdm、torch（支持 bfloat16/float16）。
- GPU 环境（device_map='auto'）。

## 注意事项

- 路径命名中 `{model}` 必须与实际采集/训练时一致，确保 `head_wise.npy` 与 `labels.npy` 对应同一模型与数据集。
- `--num_heads` 与 `--head_dim` 必须与模型配置一致（如 Llama2-7B chat 为 32×128）。
- 干预强度：`alpha`；
- 若出现空类样本导致 CoM 方向为零向量，代码会回退为零向量，不影响运行但建议检查数据覆盖。