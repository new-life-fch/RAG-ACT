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

## 代码结构与主脚本

- `RAG/llama_get_activations.py`：采集激活，保存至 `RAG/features/`。
- `RAG/train_save_probes.py`：训练所有头的探针，保存探针、准确率矩阵，以及可选的 Top-k 头列表至 `RAG/results_dump/probes/`。
- `RAG/nq_hparam_search.py`：超参数搜索。支持多种头选择策略、强度网格与分层强度映射；默认使用 CoM 方向与 `(1-动态分数)`。
- `RAG/causal_trace_experiment.py`: 因果追踪实验。分别干预LLM每层的所有头，观察结果指标，并根据EM和F1分数相对于标准RAG的增长进行综合排序，输出到一个csv文件。
- `RAG/nq_layer_hparam_search.py`: 基于因果追踪结果的前 k 层超参数搜索实验。根据 `causal_layer_trace.csv` 的结果，对前 k 层（k=1...L）的所有头进行干预，并汇总 Baseline 与 Delta EM/F1 到 CSV 文件。
- `RAG/nq_fine_grained_hparam_search.py`: 基于因果追踪结果的细粒度超参数搜索实验。根据 `causal_layer_trace.csv` 的结果，对前k层的分数阈值前m个头（m=1...M）进行干预，并汇总 Baseline 与 Delta EM/F1 到 CSV 文件。

## 数据与准备

1. 数据构建：
    - `python RAG/utils/3_extract_remaining_testset.py --input RAG/data/PopQA/PopQA_processed.jsonl --val-output RAG/data/PopQA/val_noise_test.jsonl --test-output RAG/data/PopQA/test_noise_test.jsonl --train-num 0 --val-num 0 --test-num 1190 --method random --seed 2025 --noise-levels 0,1,2,3,4,5`

2. 准备模型权重（示例路径）：
   - Llama-2-7b-chat: `/root/shared-nvme/RAG-llm/models/Llama-2-7b-chat-hf`
   - Llama-3-8B-Instruct: `/root/shared-nvme/RAG-llm/models/Llama-3-8B-Instruct`

3. 采集 NQ 激活与标签：
   - 将 `RAG/data/NQ/llama_2_chat_7b_train.jsonl` 或自定义 NQ 格式数据放置到合适位置。
   - 运行示例（使用 chat 模板）：
     - `python RAG/llama_get_activations.py --model_name vicuna_7B_v1.5 --dataset_name popqa --use_chat_template --nq_jsonl RAG/data/PopQA/train_user.jsonl --output_dir RAG/features/vicuna_7B_v1.5_popqa_user_noise --use_noise_contrastive`
   - 产物：`RAG/features/{model}_nq_head_wise.npy`、`RAG/features/{model}_nq_labels.npy`、可选 `tokens.pkl`。

## 训练探针

- 基本命令：
  - `python RAG/train_save_probes.py --model_name vicuna_7B_v1.5 --num_heads 32 --head_dim 128 --feat_dir RAG/features/vicuna_7B_v1.5_popqa_user_noise --save_dir RAG/probes/vicuna_7B_v1.5_popqa_user_noise --cv_final_train full --dataset_name popqa`
- 输出目录：`RAG/results_dump/probes/`
  - `*_probes.pkl`：所有头的探针（长度 = L×H）。
  - `*_val_accs.npy`：验证准确率矩阵，形状 `(L, H)`。
  - `*_top_heads.pkl`：Top-k 头列表（可选）。

## 获取探针的验证集分数
- 基本命令：
  - `python RAG/utils/inspect_probes.py RAG/probes/vicuna_7B_v1.5_popqa_user_noise/vicuna_7B_v1.5_popqa_seed_2025_top_1024_folds_3_val_accs.npy --out-csv RAG/probes/vicuna_7B_v1.5_popqa_user_noise/accs_csv.csv`

## 热力图工具
- 基本命令
  - `python RAG/utils/plot_probe_heatmap.py RAG/probes/vicuna_7B_v1.5_popqa_user_noise/accs_csv.csv --output RAG/probes/vicuna_7B_v1.5_popqa_user_noise/accs_heatmap.png --cmap turbo --bins 12 --discrete`

## 实验

- 示例命令：

  NQ:
```bash
python RAG/nq_hparam_search.py \
--model_name vicuna_7B_v1.5 \
--dataset_path RAG/data/NQ/test_noise_test_noise4.jsonl \
--probes_path RAG/probes/vicuna_7B_v1.5_nq_user/vicuna_7B_v1.5_nq_seed_2025_top_1024_folds_3_probes.pkl \
--val_accs_path RAG/probes/vicuna_7B_v1.5_nq_user/vicuna_7B_v1.5_nq_seed_2025_top_1024_folds_3_val_accs.npy \
--tuning_headwise_path RAG/features/vicuna_7B_v1.5_nq_user/vicuna_7B_v1.5_nq_head_wise.npy \
--tuning_labels_path RAG/features/vicuna_7B_v1.5_nq_user/vicuna_7B_v1.5_nq_labels.npy \
--scores_csv RAG/probes/vicuna_7B_v1.5_nq_user/accs_csv.csv \
--alphas 5 --probe_factor_modes false --max_new_tokens 256 --include_strategies topk_128_by_score --results_root RAG/results/vicuna-7b-v1.5-nq-user/topk_128_by_score_alphas_5 --sample_size 300 --timeout_minutes 6
```

```bash
python RAG/nq_hparam_search.py \
--model_name vicuna_7B_v1.5 \
--dataset_path RAG/data/NQ/test_noise_test_noise4.jsonl \
--probes_path RAG/probes/vicuna_7B_v1.5_popqa_user_noise/vicuna_7B_v1.5_popqa_seed_2025_top_1024_folds_3_probes.pkl \
--val_accs_path RAG/probes/vicuna_7B_v1.5_popqa_user_noise/vicuna_7B_v1.5_popqa_seed_2025_top_1024_folds_3_val_accs.npy \
--tuning_headwise_path RAG/features/vicuna_7B_v1.5_popqa_user_noise/vicuna_7B_v1.5_popqa_head_wise.npy \
--tuning_labels_path RAG/features/vicuna_7B_v1.5_popqa_user_noise/vicuna_7B_v1.5_popqa_labels.npy \
--scores_csv RAG/probes/vicuna_7B_v1.5_popqa_user_noise/accs_csv.csv \
--alphas 7 --probe_factor_modes false --max_new_tokens 256 --include_strategies topk_87_by_score --results_root RAG/results/vicuna-7b-v1.5-nq-user/topk_87_by_score_alphas_7 --sample_size 300 --timeout_minutes 6
```

  Trivia QA:
```bash
python RAG/nq_hparam_search.py \
--model_name vicuna_7B_v1.5 \
--dataset_path RAG/data/TriviaQA/test_noise_test_noise4.jsonl \
--probes_path RAG/probes/vicuna_7B_v1.5_nq_user/vicuna_7B_v1.5_nq_seed_2025_top_1024_folds_3_probes.pkl \
--val_accs_path RAG/probes/vicuna_7B_v1.5_nq_user/vicuna_7B_v1.5_nq_seed_2025_top_1024_folds_3_val_accs.npy \
--tuning_headwise_path RAG/features/vicuna_7B_v1.5_nq_user/vicuna_7B_v1.5_nq_head_wise.npy \
--tuning_labels_path RAG/features/vicuna_7B_v1.5_nq_user/vicuna_7B_v1.5_nq_labels.npy \
--scores_csv RAG/probes/vicuna_7B_v1.5_nq_user/accs_csv.csv \
--alphas 5 --probe_factor_modes false --max_new_tokens 256 --include_strategies layers_8_14 --results_root RAG/results/vicuna-7b-v1.5-triviaqa-user/layers_8_14_alphas_5 --sample_size 300 --timeout_minutes 6
```

```bash
python RAG/nq_hparam_search.py \
--model_name vicuna_7B_v1.5 \
--dataset_path RAG/data/TriviaQA/test_noise_test_noise4.jsonl \
--probes_path RAG/probes/vicuna_7B_v1.5_popqa_user_noise/vicuna_7B_v1.5_popqa_seed_2025_top_1024_folds_3_probes.pkl \
--val_accs_path RAG/probes/vicuna_7B_v1.5_popqa_user_noise/vicuna_7B_v1.5_popqa_seed_2025_top_1024_folds_3_val_accs.npy \
--tuning_headwise_path RAG/features/vicuna_7B_v1.5_popqa_user_noise/vicuna_7B_v1.5_popqa_head_wise.npy \
--tuning_labels_path RAG/features/vicuna_7B_v1.5_popqa_user_noise/vicuna_7B_v1.5_popqa_labels.npy \
--scores_csv RAG/probes/vicuna_7B_v1.5_popqa_user_noise/accs_csv.csv \
--alphas 7 --probe_factor_modes false --max_new_tokens 256 --include_strategies topk_87_by_score --results_root RAG/results/vicuna-7b-v1.5-triviaqa-user/topk_87_by_score_alphas_7 --sample_size 300 --timeout_minutes 6
```

PopQA:
```bash
python RAG/nq_hparam_search.py \
--model_name vicuna_7B_v1.5 \
--dataset_path RAG/data/PopQA/test_noise_test_noise4.jsonl \
--probes_path RAG/probes/vicuna_7B_v1.5_nq_user/vicuna_7B_v1.5_nq_seed_2025_top_1024_folds_3_probes.pkl \
--val_accs_path RAG/probes/vicuna_7B_v1.5_nq_user/vicuna_7B_v1.5_nq_seed_2025_top_1024_folds_3_val_accs.npy \
--tuning_headwise_path RAG/features/vicuna_7B_v1.5_nq_user/vicuna_7B_v1.5_nq_head_wise.npy \
--tuning_labels_path RAG/features/vicuna_7B_v1.5_nq_user/vicuna_7B_v1.5_nq_labels.npy \
--scores_csv RAG/probes/vicuna_7B_v1.5_nq_user/accs_csv.csv \
--alphas 5 --probe_factor_modes false --max_new_tokens 256 --include_strategies layers_8_14 --results_root RAG/results/vicuna-7b-v1.5-popqa-user/layers_8_14_alphas_5 --sample_size 300
```

```bash
python RAG/nq_hparam_search.py \
--model_name vicuna_7B_v1.5 \
--dataset_path RAG/data/PopQA/test_noise_test_noise4.jsonl \
--probes_path RAG/probes/vicuna_7B_v1.5_popqa_user_noise/vicuna_7B_v1.5_popqa_seed_2025_top_1024_folds_3_probes.pkl \
--val_accs_path RAG/probes/vicuna_7B_v1.5_popqa_user_noise/vicuna_7B_v1.5_popqa_seed_2025_top_1024_folds_3_val_accs.npy \
--tuning_headwise_path RAG/features/vicuna_7B_v1.5_popqa_user_noise/vicuna_7B_v1.5_popqa_head_wise.npy \
--tuning_labels_path RAG/features/vicuna_7B_v1.5_popqa_user_noise/vicuna_7B_v1.5_popqa_labels.npy \
--scores_csv RAG/probes/vicuna_7B_v1.5_popqa_user_noise/accs_csv.csv \
--alphas 7 --probe_factor_modes false --max_new_tokens 256 --include_strategies topk_87_by_score --results_root RAG/results/vicuna-7b-v1.5-popqa-user/topk_87_by_score_alphas_7 --sample_size 300 --timeout_minutes 6
```

- 产物：`results_dump/vicuna-7b-v1.5-popqa-user/` 下的逐次 summary 与最终汇总 CSV。

## 超参

```bash
python RAG/nq_hparam_search_topk_alpha.py \
--model_name vicuna_7B_v1.5 \
--dataset_path RAG/data/PopQA/test_noise_test_noise4.jsonl \
--probes_path RAG/probes/vicuna_7B_v1.5_popqa_user_noise/vicuna_7B_v1.5_popqa_seed_2025_top_1024_folds_3_probes.pkl \
--val_accs_path RAG/probes/vicuna_7B_v1.5_popqa_user_noise/vicuna_7B_v1.5_popqa_seed_2025_top_1024_folds_3_val_accs.npy \
--tuning_headwise_path RAG/features/vicuna_7B_v1.5_popqa_user_noise/vicuna_7B_v1.5_popqa_head_wise.npy \
--tuning_labels_path RAG/features/vicuna_7B_v1.5_popqa_user_noise/vicuna_7B_v1.5_popqa_labels.npy \
--scores_csv RAG/probes/vicuna_7B_v1.5_popqa_user_noise/accs_csv.csv \
--results_root RAG/results/vicuna-7b-v1.5-popqa-search_noise-300 \
--sample_size 300 \
--max_new_tokens 256 \
--timeout_minutes 7
```

## 因果追踪实验

```bash
python RAG/causal_trace_experiment.py \
--model_name vicuna_7B_v1.5 \
--dataset_path RAG/data/NQ/test_noise_test_noise4.jsonl \
--probes_path RAG/probes/vicuna_7B_v1.5_nq_user/vicuna_7B_v1.5_nq_seed_2025_top_1024_folds_3_probes.pkl \
--val_accs_path RAG/probes/vicuna_7B_v1.5_nq_user/vicuna_7B_v1.5_nq_seed_2025_top_1024_folds_3_val_accs.npy \
--tuning_headwise_path RAG/features/vicuna_7B_v1.5_nq_user/vicuna_7B_v1.5_nq_head_wise.npy \
--tuning_labels_path RAG/features/vicuna_7B_v1.5_nq_user/vicuna_7B_v1.5_nq_labels.npy \
--alpha 5.0 --max_new_tokens 256 --output_csv RAG/results/vicuna-7b-v1.5-nq-user/causal_layer_trace_pf0.csv
```

## 前 k 层干预超参数搜索实验 (Based on Causal Trace)

- 示例命令：
```bash
python RAG/nq_layer_hparam_search.py \
--model_name vicuna_7B_v1.5 \
--dataset_path RAG/data/TriviaQA/test_noise_test_noise4.jsonl \
--probes_path RAG/probes/vicuna_7B_v1.5_nq_user/vicuna_7B_v1.5_nq_seed_2025_top_1024_folds_3_probes.pkl \
--val_accs_path RAG/probes/vicuna_7B_v1.5_nq_user/vicuna_7B_v1.5_nq_seed_2025_top_1024_folds_3_val_accs.npy \
--tuning_headwise_path RAG/features/vicuna_7B_v1.5_nq_user/vicuna_7B_v1.5_nq_head_wise.npy \
--tuning_labels_path RAG/features/vicuna_7B_v1.5_nq_user/vicuna_7B_v1.5_nq_labels.npy \
--sample_size 100 \
--probe_factor_modes false \
--results_root RAG/results/vicuna-7b-v1.5-triviaqa-user/top-layers-intervention \
--summary_csv top_layer_intervention_results.csv \
--alphas 5 \
--max_new_tokens 256
```

## 细粒度超参数搜索实验
- 示例命令：
```bash
python RAG/nq_fine_grained_hparam_search.py \
--model_name vicuna_7B_v1.5 \
--dataset_path RAG/data/TriviaQA/test_noise_test_noise4.jsonl \
--probes_path RAG/probes/vicuna_7B_v1.5_nq_user/vicuna_7B_v1.5_nq_seed_2025_top_1024_folds_3_probes.pkl \
--val_accs_path RAG/probes/vicuna_7B_v1.5_nq_user/vicuna_7B_v1.5_nq_seed_2025_top_1024_folds_3_val_accs.npy \
--tuning_headwise_path RAG/features/vicuna_7B_v1.5_nq_user/vicuna_7B_v1.5_nq_head_wise.npy \
--tuning_labels_path RAG/features/vicuna_7B_v1.5_nq_user/vicuna_7B_v1.5_nq_labels.npy \
--causal_trace_path RAG/results/vicuna-7b-v1.5-nq-user/causal_layer_trace_pf0.csv \
--head_scores_path RAG/probes/vicuna_7B_v1.5_nq_user/accs_csv.csv \
--sample_size 100 \
--top_k_layers 8 \
--thresholds 0.75,0.7,0.65,0.6,0.55,0.48 \
--results_root RAG/results/vicuna-7b-v1.5-triviaqa-user/nq_fine_grained \
--summary_csv triviaqa_noise4_result_top_8_layers_score.csv \
--alphas 5 \
--max_new_tokens 256
```

## CoN
- 示例命令：
```bash
python RAG/con_rag.py --model_name vicuna_7B_v1.5 --dataset_path RAG/data/NQ/test_noise_test_noise4.jsonl --use_chat_template --max_docs 5 --sample_size 300 --max_new_tokens 256 --results_root RAG/results/vicuna_7B_v1.5-nq-user/CON_300
```
```bash
python RAG/con_rag.py --model_name vicuna_7B_v1.5 --dataset_path RAG/data/TriviaQA/test_noise_test_noise4.jsonl --use_chat_template --max_docs 5 --sample_size 300 --max_new_tokens 256 --results_root RAG/results/vicuna_7B_v1.5-triviaqa-user/CON_300
```

## naive LLM
- 示例命令：
```bash
python RAG/nq_naive_llm.py \
--model_name vicuna_7B_v1.5 \
--dataset_path RAG/data/NQ/test_noise_test_noise4.jsonl \
--sample_size 300 \
--max_new_tokens 256 \
--results_root RAG/results/vicuna-7b-v1.5-nq-user/naive-llm
```

## 评估指标

- `evaluate_nq_em_f1`：按全部 gold 答案进行 EM/F1 计算，脚本在生成结束后自动保存指标文件。

## 环境要求

- Ubuntu 20.04，Python 3.10+ 建议。
- 安装依赖：Transformers、baukit、datasets、numpy、scikit-learn、einops、tqdm、torch（支持 bfloat16/float16）。
- GPU 环境（device_map='auto'）。

## 注意事项

- 路径命名中 `{model}` 必须与实际采集/训练时一致，确保 `head_wise.npy` 与 `labels.npy` 对应同一模型与数据集。
- `--num_heads` 与 `--head_dim` 必须与模型配置一致（如 Llama2-7B chat 为 32×128）。
- 干预强度的三个乘子：`alpha`、`proj_val_std`、`reliability^γ`；
- 若出现空类样本导致 CoM 方向为零向量，代码会回退为零向量，不影响运行但建议检查数据覆盖。

