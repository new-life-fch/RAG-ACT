### 项目说明（RAG_iti，使用自有 NQ 数据集）

**目标**
- 在生成阶段对注意力头进行干预，缓解 RAG 对检索到的无关片段的噪声敏感性，提升 NQ 的 EM/F1。

**此次修改（保持原有数据处理与探针训练不变）**
- 将激活干预从 `baukit.TraceDict` 全面替换为 `pyvene.IntervenableModel`，方法与 `honest_llama` 现版本一致。
- 新增文件：`RAG_iti/interveners.py`（包含 `Collector`、`ITI_Intervener` 与 `wrapper`）。
- 更新依赖：`RAG_iti/requirements.txt` 增加 `pyvene==0.2.2`。
- 代码替换范围（仅更换干预方法，保持功能与评估目标不变）：
  - `RAG_iti/llama_get_activations.py`：改为用 pyvene 的 Collector 收集头部激活（训练集用于强度估计）。
  - `RAG_iti/nq_generate_with_interventions.py`：生成阶段用 IntervenableModel 干预（探针方向 + 验证准确率作为强度因子）。
  - `RAG_iti/nq_generate_random_and_fixed.py`：生成阶段用 IntervenableModel 干预（随机方向 / 固定强度）。
  - `RAG_iti/llama_validate_2fold.py`：如需 TruthfulQA 验证，亦改为 IntervenableModel；不影响本项目基于自有数据的流程。
  - `RAG_iti/llama_utils.py`：移除对 TraceDict 的依赖，前向与生成改为兼容 pyvene 的调用方式（dict 输入、tuple 输出处理）。

**数据与模型**
- 训练集：`RAG/data/train.jsonl`；测试集：`RAG/data/test.jsonl`。
- 每行字段：`query`、`answers`（列表）、`wrong_answer`、`retrieve_snippets`（含 `text`）。
- 默认模型：`llama3_8B_instruct`（映射到本地 `Llama-3.1-8B-Instruct`）。

**依赖安装**
- 建议先激活已有环境：`conda activate iti`
- 安装本项目依赖：`pip install -r RAG_iti/requirements.txt`

**工作流程（使用自有数据与原有探针训练方式）**
- 1) 收集训练集激活（用于调强度 `proj_val_std`）：
  - `python RAG_iti/llama_get_activations.py --model_name llama3_8B_instruct --dataset_name nq --nq_jsonl RAG/data/train.jsonl --use_chat_template`
  - 输出特征保存到：`../RAG-llm/features/llama3_8B_instruct_nq_head_wise.npy` 与 `..._labels.npy`。
- 2) 训练并保存探针与前48个头（沿用你之前的训练与筛选逻辑）：
  - `python RAG_iti/nq_train_save_probes.py --model_name llama3_8B_instruct --top_k 48 --seed 2025 --num_fold 5 --cv_final_train full`
  - 输出：`./results_dump/probes/llama3_8B_instruct_nq_seed_2025_top_48_folds_5_probes.pkl`、`..._top_heads.pkl`、`..._val_accs.npy`。
- 3) 在测试集评估两套设置（均使用 chat 模板与贪心解码）：
  - 标准RAG + 探针干预RAG（使用探针分数因子）：
    - `python RAG_iti/nq_generate_with_interventions.py --model_name llama3_8B_instruct --dataset_path RAG/data/test.jsonl --use_chat_template --alpha 15 --top_heads_path results_dump/probes/llama3_8B_instruct_nq_seed_2025_top_48_folds_5_top_heads.pkl --probes_path results_dump/probes/llama3_8B_instruct_nq_seed_2025_top_48_folds_5_probes.pkl --val_accs_path results_dump/probes/llama3_8B_instruct_nq_seed_2025_top_48_folds_5_val_accs.npy --tuning_headwise_path ../RAG-llm/features/llama3_8B_instruct_nq_head_wise.npy --sample_size 300 --sample_seed 2025 --max_new_tokens 256`
  - 随机方向（前48头） + 固定强度（前48头，无探针分数因子）：
    - `python RAG_iti/nq_generate_random_and_fixed.py --model_name llama3_8B_instruct --dataset_path RAG/data/test.jsonl --use_chat_template --alpha 15 --top_heads_path results_dump/probes/llama3_8B_instruct_nq_seed_2025_top_48_folds_5_top_heads.pkl --probes_path results_dump/probes/llama3_8B_instruct_nq_seed_2025_top_48_folds_5_probes.pkl --tuning_headwise_path ../RAG-llm/features/llama3_8B_instruct_nq_head_wise.npy --sample_size 300 --sample_seed 2025 --max_new_tokens 256`

**输出文件**
- 标准RAG：`results_dump/answer_dump/nq_gen_answers_baseline.jsonl`、`results_dump/summary_dump/nq_gen_summary_baseline.json`。
- 探针干预RAG（探针分数因子）：`results_dump/answer_dump/nq_gen_answers_intervene.jsonl`、`results_dump/summary_dump/nq_gen_summary_intervene.json`。
- 随机方向：`results_dump/answer_dump/nq_gen_answers_random_dir.jsonl`、`results_dump/summary_dump/nq_gen_summary_random_dir.json`。
- 固定强度：`results_dump/answer_dump/nq_gen_answers_fixed_strength.jsonl`、`results_dump/summary_dump/nq_gen_summary_fixed_strength.json`。

**干预与强度设定（与 honest_llama 一致）**
- 钩挂点：`model.layers[{layer}].self_attn.o_proj.input`（head_out 前的线性层输入），仅对最后一个 token 干预。
- 方向：
  - 探针方向：来自你用自有数据训练的 LR 探针系数（或 COM / 随机方向）。
  - 分层聚合：将每层的多头方向拼接为一个 `(#heads * head_dim)` 大向量输入干预。
- 强度：`alpha * proj_val_std * probe_factor`；
  - `proj_val_std` 来自训练集激活沿方向的标准差；
  - `probe_factor` 为验证准确率（随机/固定强度禁用）。

**提示词与解码**
- 使用 chat 模板构造系统/用户消息（强调“只输出直接答案，依据检索片段”），与原有实现一致。
- 贪心解码：`do_sample=False`，`max_new_tokens` 默认 256，可按资源调整。

**注意与建议**
- 维度：默认 `head_dim=128`；若模型与特征维不匹配，请显式传 `--num_heads`。
- 路径：特征输出在 `../RAG-llm/features`；探针与头文件在 `./results_dump/probes`。
- 资源：如显存不足，降低 `--max_new_tokens` 与 `--sample_size`，并保持 `device_map='auto'`。
- 模型精度：LLaMA-3 建议使用 `bfloat16`；已在脚本中自动选择 dtype。

**核对运行**
- 四套设置运行后，查看对应 `summary_dump/*.json` 的 `EM` 与 `F1` 是否符合预期，确保干预方法替换未改变评估逻辑与目标。