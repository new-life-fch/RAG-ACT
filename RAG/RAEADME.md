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


python RAG/nq_hparam_search.py --model_name llama2_chat_7B --dataset_path RAG/data/test.jsonl --use_chat_template --probes_path results_dump/probes/llama2_chat_7B_nq_seed_2025_top_1024_folds_5_probes.pkl --val_accs_path results_dump/probes/llama2_chat_7B_nq_seed_2025_top_1024_folds_5_val_accs.npy --tuning_headwise_path features/llama2_chat_7B_nq_head_wise.npy --scores_csv results_dump/probe_scores/llama2_chat_7B_nq_seed_2025_all_head_scores.csv --sample_size 100 --max_new_tokens 256 --include_strategies layers_10_31,score_ge_0.7,topk_256_by_score,per_layer_top_4 --alphas 3,5,9 --probe_factor_modes true --limit_per_strategy 1024




          
**文件作用**
- `results_dump/probes/llama2_chat_7B_nq_seed_2025_top_1024_folds_5_top_heads.pkl`
  - 内容：按验证分数选出的头集合，通常是一个列表 `[(layer, head), ...]`（有时包在字典里）。
  - 作用：在生成或干预脚本里直接指定“用哪些头进行干预”，避免现场再做排序或筛选。
  - 典型使用：`nq_generate_with_interventions.py`、`nq_generate_random_and_fixed.py` 这类脚本会读取它，作为 `top_heads` 传入 `get_interventions_dict(...)`。

**为什么在超参搜索脚本里没用到它**
- 在 `nq_hparam_search.py` 里，我们需要同时支持“分数阈值”“层区间”“Top-k（多档）”“分层 Top-m”等多种头选择策略，并且要能限量、白名单筛选。
- 为了统一来源和排序，我们用的是 `results_dump/probe_scores/..._all_head_scores.csv`（三列：`layer, head, score`）来生成各种策略子集；这已经涵盖了从“全部头”的分数排序到“各层分数排序”的需求。
- 因此，该脚本不再依赖某一个固定的 `top_heads.pkl` 集合，而是从 CSV 分数动态构造不同的子集来试验。你仍然保留 `probes.pkl`（方向）和 `val_accs.npy`（探针分数因子），这两者在干预时是必须的。

如果你希望在搜索中也直接复用这份 `top_heads.pkl`，我可以很容易加一个策略，比如 `saved_top_heads`：读取 `--top_heads_path`，把其中的 `(layer, head)` 集合加入到策略列表中与其它策略一并跑。

**results_dump/probes 下文件的作用**
- `..._probes.pkl`
  - 训练好的每个注意力头的逻辑回归探针（sklearn），用于给出该头的“干预方向” `coef_`。
  - 在运行干预时，`get_interventions_dict(...)` 会取对应头的 `coef_` 并单位化作为方向。
- `..._val_accs.npy`
  - 验证集上每个 `(layer, head)` 的评分矩阵（常用准确率，形状 `(L, H)`），用于作为“探针分数因子” `probe_factor`（0–1）调制干预强度。
- `..._top_heads.pkl`
  - 预先挑好的一个头集合（通常 top-K），可在生成脚本中直接作为干预子集使用。
- 可能还有其它辅助文件（如折分信息），但核心是上面三类。

这些文件之间的对应关系需要保持一致：同一模型、相同层数与头数的编号体系。超参脚本用 `probes.pkl + val_accs.npy + scores.csv` 组合，`top_heads.pkl` 在该脚本中不是必需品。

**features 下文件的作用**
- `features/llama2_chat_7B_nq_head_wise.npy`
  - 采集到的头级激活，形状 `(B, L, H*D)`。在干预阶段用于计算沿“干预方向”的标准差 `proj_val_std`（数据驱动尺度），也用于训练探针时作为特征。
- `features/llama2_chat_7B_nq_labels.npy`
  - 对应每个样本的标签（如正确/错误），用于训练探针；超参脚本不再重训探针，所以通常不直接用到。
- `features/llama2_chat_7B_nq_tokens.pkl`（如存在）
  - 采集时的 token 序列，便于检查定位，非必须。

简言之：
- 干预时必须：`probes.pkl`（方向）、`val_accs.npy`（分数因子，可选开关）、`head_wise.npy`（计算标准差）。
- `top_heads.pkl` 是一种“预选的头子集”的快捷文件，供固定策略脚本直接使用；在我们的搜索脚本里，因要统一和扩展多种策略，改用 `scores.csv` 按需生成子集，所以未直接用到它。
        