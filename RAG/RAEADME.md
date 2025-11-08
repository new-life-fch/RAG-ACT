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