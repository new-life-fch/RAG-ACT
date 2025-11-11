### 精炼版说明（llama3_8B，RAG/data/train.jsonl 与 RAG/data/test.jsonl）

**目标**
- 在生成阶段对注意力头进行干预，缓解 RAG 对无关片段的敏感性，提升 NQ 的 EM/F1。

**数据与模型**
- 训练集：`RAG/data/train.jsonl`；测试集：`RAG/data/test.jsonl`。
- 模型：`llama3_8B_instruct`（映射到本地 `Llama-3.1-8B-Instruct`）。
- 每行字段：`query`、`answers`（列表）、`wrong_answer`、`retrieve_snippets`（含 `text`）。

**依赖**
- `transformers`、`baukit`、`datasets`、`tqdm`、`einops`、`scikit-learn`。

**工作流程**
- 收集训练集激活（用于调强度）：
  - `python RAG/llama_get_activations.py --model_name llama3_8B_instruct --dataset_name nq --nq_jsonl RAG/data/train.jsonl --use_chat_template`
- 训练并保存探针与前48个头：
  - `python RAG/nq_train_save_probes.py --model_name llama3_8B_instruct --top_k 48 --seed 2025 --num_fold 5 --cv_final_train full`
- 在测试集评估四种设置（均使用 chat 模板与贪心解码）：
  - 标准RAG + 探针干预RAG（含探针分数因子）：
    - `python RAG/nq_generate_with_interventions.py --model_name llama3_8B_instruct --dataset_path RAG/data/test.jsonl --use_chat_template --alpha 15 --top_heads_path results_dump/probes/llama3_8B_instruct_nq_seed_2025_top_48_folds_5_top_heads.pkl --probes_path results_dump/probes/llama3_8B_instruct_nq_seed_2025_top_48_folds_5_probes.pkl --val_accs_path results_dump/probes/llama3_8B_instruct_nq_seed_2025_top_48_folds_5_val_accs.npy --tuning_headwise_path features/llama3_8B_instruct_nq_head_wise.npy --sample_size 300 --sample_seed 2025 --max_new_tokens 256`
  - 随机方向（前48头） + 固定强度（前48头，无探针分数因子）：
    - `python RAG/nq_generate_random_and_fixed.py --model_name llama3_8B_instruct --dataset_path RAG/data/test.jsonl --use_chat_template --alpha 15 --top_heads_path results_dump/probes/llama3_8B_instruct_nq_seed_2025_top_48_folds_5_top_heads.pkl --probes_path results_dump/probes/llama3_8B_instruct_nq_seed_2025_top_48_folds_5_probes.pkl --tuning_headwise_path features/llama3_8B_instruct_nq_head_wise.npy --sample_size 300 --sample_seed 2025 --max_new_tokens 256`

**输出文件**
- 标准RAG：`results_dump/answer_dump/nq_gen_answers_baseline.jsonl`、`results_dump/summary_dump/nq_gen_summary_baseline.json`。
- 探针干预RAG（探针分数因子）：`results_dump/answer_dump/nq_gen_answers_intervene.jsonl`、`results_dump/summary_dump/nq_gen_summary_intervene.json`。
- 随机方向：`results_dump/answer_dump/nq_gen_answers_random_dir.jsonl`、`results_dump/summary_dump/nq_gen_summary_random_dir.json`。
- 固定强度：`results_dump/answer_dump/nq_gen_answers_fixed_strength.jsonl`、`results_dump/summary_dump/nq_gen_summary_fixed_strength.json`。

**干预与强度**
- 钩挂点：`model.layers.{layer}.self_attn.head_out`，在最后 token 加单位化方向。
- 强度：`alpha * proj_val_std * direction`；
  - 探针干预RAG额外乘以 `probe_factor`（验证准确率），随机/固定强度不使用该因子。

**提示词与解码**
- 与 `RAG/utils/generate_dataset.py#L45-91` 一致的系统/用户消息（chat 模板），仅输出直接答案。
- 贪心解码：`do_sample=False`，`max_new_tokens` 默认 256，可根据资源调整。

**注意**
- 维度：默认 `head_dim=128`；如模型不同需显式 `--num_heads` 或调整维度。
- 资源：不足时降低 `--max_new_tokens` 或 `--sample_size`；支持 `device_map='auto'`。

## 超参数搜索（nq_hparam_search.py）

用于在不同头选择策略、干预强度和是否乘探针分数的配置下，自动运行生成与评估，并将结果落盘。

- 层索引说明：采用零基索引。`layers_0_31` 表示选择模型第 0 层到第 31 层（共 32 层），实现为区间 `[0, min(32, L))`，当 `L=32` 时包含第 31 层。
- 头选择策略（示例）：
  - 分数阈值：`score_ge_{0.5,0.6,0.7,0.8,0.9}`。
  - 层区间：`layers_{0_10,10_20,20_31,0_20,10_31,0_31}`。
  - 全局 Top-k：`topk_{64,128,256,512}_by_score`（按 CSV 分数降序）。
  - 分层 Top-m：`per_layer_top_{1,2,4,8,16}`（每层按分数取前 m 个）。
- 干预强度：`alpha ∈ {2, 4, ..., 20}`。
 - 干预强度：默认 `alpha ∈ {1, 3, 5, ..., 19}`（可通过 `--alphas` 手动指定）。
- 是否乘探针分数：分别运行 `use_probe_factor=False/True` 两种。
- Baseline：只跑一次，抽样 100 条。
- 汇总输出：`nq_hparam_search_summary.csv` 按 F1 降序排列。

示例启动命令（100 条样本）：

```
python RAG/nq_hparam_search.py \
  --model_name llama2_chat_7B \
  --dataset_path RAG/data/test.jsonl \
  --use_chat_template \
  --probes_path results_dump/probes/llama2_chat_7B_nq_seed_2025_top_1024_folds_5_probes.pkl \
  --val_accs_path results_dump/probes/llama2_chat_7B_nq_seed_2025_top_1024_folds_5_val_accs.npy \
  --tuning_headwise_path features/llama2_chat_7B_nq_head_wise.npy \
  --scores_csv results_dump/probe_scores/llama2_chat_7B_nq_seed_2025_all_head_scores.csv \
  --sample_size 100 \
  --max_new_tokens 256
```

说明：
- 若你的 `probes.pkl` 覆盖所有层与头（如 `top_1024`），即可支持上述所有选择方案，无需重新训练探针；保证 `scores_csv` 与 `val_accs.npy` 的 `(layer, head)` 编号与 `probes.pkl` 一致即可。

1. 试跑
python RAG/nq_hparam_search.py --model_name llama2_chat_7B --dataset_path RAG/data/test.jsonl --use_chat_template --probes_path results_dump/probes/llama2_chat_7B_nq_seed_2025_top_1024_folds_5_probes.pkl --val_accs_path results_dump/probes/llama2_chat_7B_nq_seed_2025_top_1024_folds_5_val_accs.npy --tuning_headwise_path features/llama2_chat_7B_nq_head_wise.npy --scores_csv results_dump/probe_scores/llama2_chat_7B_nq_seed_2025_all_head_scores.csv --sample_size 100 --max_new_tokens 256 --include_strategies layers_10_31,score_ge_0.7,topk_256_by_score,per_layer_top_4 --alphas 3,5,9 --probe_factor_modes true --limit_per_strategy 1024

2. 补充实验
python RAG/nq_hparam_search.py --model_name llama2_chat_7B --dataset_path RAG/data/test.jsonl --use_chat_template --probes_path results_dump/probes/llama2_chat_7B_nq_seed_2025_top_1024_folds_5_probes.pkl --val_accs_path results_dump/probes/llama2_chat_7B_nq_seed_2025_top_1024_folds_5_val_accs.npy --tuning_headwise_path features/llama2_chat_7B_nq_head_wise.npy --scores_csv results_dump/probe_scores/llama2_chat_7B_nq_seed_2025_all_head_scores.csv --sample_size 100 --max_new_tokens 256 --supplement --timeout_minutes 6

        