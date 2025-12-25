采集激活
python RAG/llama_get_activations.py --model_name llama3_8B_instruct --dataset_name popqa --use_chat_template --nq_jsonl RAG/data/PopQA/train_user.jsonl --output_dir RAG/features/llama3_8B_instruct_popqa_user_noise --use_noise_contrastive

训练探针
python RAG/train_save_probes.py --model_name llama3_8B_instruct --num_heads 32 --head_dim 128 --feat_dir RAG/features/llama3_8B_instruct_popqa_user_noise  --save_dir RAG/probes/llama3_8B_instruct_popqa_user_noise --cv_final_train full --dataset popqa

获取探针分数
python RAG/utils/inspect_probes.py RAG/probes/llama3_8B_instruct_popqa_user_noise/llama3_8B_instruct_popqa_seed_2025_top_1024_folds_3_val_accs.npy --out-csv RAG/probes/llama3_8B_instruct_popqa_user_noise/accs_csv.csv

热力图
python RAG/utils/plot_probe_heatmap.py RAG/probes/llama3_8B_instruct_popqa_user_noise/accs_csv.csv --output RAG/probes/llama3_8B_instruct_popqa_user_noise/accs_heatmap.png --cmap turbo --bins 12 --discrete

因果追踪
python RAG/causal_trace_experiment.py \
  --model_name llama3_8B_instruct \
  --dataset_path RAG/data/NQ/test_noise_test_noise4.jsonl \
  --use_chat_template \
  --probes_path RAG/probes/llama3_8B_instruct_nq_user/llama3_8B_instruct_nq_seed_2025_top_1024_folds_3_probes.pkl \
  --val_accs_path RAG/probes/llama3_8B_instruct_nq_user/llama3_8B_instruct_nq_seed_2025_top_1024_folds_3_val_accs.npy \
  --tuning_headwise_path RAG/features/llama3_8B_instruct_nq_user/llama3_8B_instruct_nq_head_wise.npy \
  --tuning_labels_path RAG/features/llama3_8B_instruct_nq_user/llama3_8B_instruct_nq_labels.npy \
  --alpha 5.0 --max_new_tokens 256 --output_csv RAG/results/llama-3-8b-instruct-nq-user/causal_layer_trace_pf0.csv --sample_size 50

前k层超参数干预
python RAG/nq_layer_hparam_search.py \
--model_name llama3_8B_instruct \
--dataset_path RAG/data/TriviaQA/test_noise_test_noise4.jsonl \
--use_chat_template \
--probes_path RAG/probes/llama3_8B_instruct_nq_user/llama3_8B_instruct_nq_seed_2025_top_1024_folds_3_probes.pkl \
--val_accs_path RAG/probes/llama3_8B_instruct_nq_user/llama3_8B_instruct_nq_seed_2025_top_1024_folds_3_val_accs.npy \
--tuning_headwise_path RAG/features/llama3_8B_instruct_nq_user/llama3_8B_instruct_nq_head_wise.npy \
--tuning_labels_path RAG/features/llama3_8B_instruct_nq_user/llama3_8B_instruct_nq_labels.npy \
--sample_size 100 \
--probe_factor_modes false \
--results_root RAG/results/llama-3-8b-instruct-triviaqa-user/top-layers-intervention \
--summary_csv top_layer_intervention_results.csv \
--alphas 5 \
--max_new_tokens 256

细粒度超参数搜索实验
python RAG/nq_fine_grained_hparam_search.py \
  --model_name llama3_8B_instruct \
  --dataset_path RAG/data/NQ/test_noise_test_noise3.jsonl \
  --probes_path RAG/probes/llama3_8B_instruct_nq_user/llama3_8B_instruct_nq_seed_2025_top_1024_folds_3_probes.pkl \
  --val_accs_path RAG/probes/llama3_8B_instruct_nq_user/llama3_8B_instruct_nq_seed_2025_top_1024_folds_3_val_accs.npy \
  --tuning_headwise_path RAG/features/llama3_8B_instruct_nq_user/llama3_8B_instruct_nq_head_wise.npy \
  --tuning_labels_path RAG/features/llama3_8B_instruct_nq_user/llama3_8B_instruct_nq_labels.npy \
  --causal_trace_path RAG/results/llama-3-8b-instruct-nq-user/causal_layer_trace_pf0.csv \
  --head_scores_path RAG/probes/llama3_8B_instruct_nq_user/accs_csv.csv \
  --sample_size 100 \
  --top_k_layers 7 \
  --thresholds 0.8,0.75,0.7,0.65,0.6,0.55,0.48 \
  --results_root RAG/results/llama-3-8b-instruct-nq-user/nq_fine_grained \
  --summary_csv Rfine_grained_intervention_results_alpha_5.csv \
  --alphas 5 \
  --max_new_tokens 256


NQ:

```bash
python RAG/generate.py \
--model_name llama3_8B_instruct \
--dataset_path RAG/data/NQ/test_noise_test_noise4.jsonl \
--use_chat_template \
--probes_path RAG/probes/llama3_8B_instruct_popqa_user_noise/llama3_8B_instruct_popqa_seed_2025_top_1024_folds_3_probes.pkl \
--val_accs_path RAG/probes/llama3_8B_instruct_popqa_user_noise/llama3_8B_instruct_popqa_seed_2025_top_1024_folds_3_val_accs.npy \
--tuning_headwise_path RAG/features/llama3_8B_instruct_popqa_user_noise/llama3_8B_instruct_popqa_head_wise.npy \
--tuning_labels_path RAG/features/llama3_8B_instruct_popqa_user_noise/llama3_8B_instruct_popqa_labels.npy \
--scores_csv RAG/probes/llama3_8B_instruct_popqa_user_noise/accs_csv.csv \
--alphas 7 --probe_factor_modes false --max_new_tokens 256 --include_strategies topk_112_by_score --results_root RAG/results/llama3/llama-3-8b-instruct-nq-user_noise/topk_112_by_score_alphas_7 --sample_size 300 --timeout_minutes 10
```

Trivia QA:

```bash
python RAG/generate.py \
--model_name llama3_8B_instruct \
--dataset_path RAG/data/TriviaQA/test_noise_test_noise4.jsonl \
--use_chat_template \
--probes_path RAG/probes/llama3_8B_instruct_popqa_user_noise/llama3_8B_instruct_popqa_seed_2025_top_1024_folds_3_probes.pkl \
--val_accs_path RAG/probes/llama3_8B_instruct_popqa_user_noise/llama3_8B_instruct_popqa_seed_2025_top_1024_folds_3_val_accs.npy \
--tuning_headwise_path RAG/features/llama3_8B_instruct_popqa_user_noise/llama3_8B_instruct_popqa_head_wise.npy \
--tuning_labels_path RAG/features/llama3_8B_instruct_popqa_user_noise/llama3_8B_instruct_popqa_labels.npy \
--scores_csv RAG/probes/llama3_8B_instruct_popqa_user_noise/accs_csv.csv \
--alphas 3 --probe_factor_modes false --max_new_tokens 256 --include_strategies topk_60_by_score --results_root RAG/results/llama3/llama-3-8b-instruct-triviaqa-user_noise/topk_60_by_score_alphas_3 --sample_size 300 --timeout_minutes 10
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



## 超参实验（top-k和alpha）

```bash
python RAG/rag_hparam_search_topk_alpha.py \
--model_name llama3_8B_instruct \
--dataset_path RAG/data/PopQA/test_noise_test_noise4.jsonl \
--use_chat_template \
--probes_path RAG/probes/llama3_8B_instruct_popqa_user_noise/llama3_8B_instruct_popqa_seed_2025_top_1024_folds_3_probes.pkl \
--val_accs_path RAG/probes/llama3_8B_instruct_popqa_user_noise/llama3_8B_instruct_popqa_seed_2025_top_1024_folds_3_val_accs.npy \
--tuning_headwise_path RAG/features/llama3_8B_instruct_popqa_user_noise/llama3_8B_instruct_popqa_head_wise.npy \
--tuning_labels_path RAG/features/llama3_8B_instruct_popqa_user_noise/llama3_8B_instruct_popqa_labels.npy \
--scores_csv RAG/probes/llama3_8B_instruct_popqa_user_noise/accs_csv.csv \
--results_root RAG/results/llama3-8b-popqa-search_noise-300 \
--sample_size 300 \
--max_new_tokens 256
--timeout_minutes 10
```

临时命令
python RAG/nq_hparam_search.py \
--model_name llama3_8B_instruct \
--dataset_path RAG/data/NQ/test_noise_test_noise4.jsonl \
--use_chat_template \
--probes_path RAG/probes/llama3_8B_instruct_popqa_user_noise_answer_end/llama3_8B_instruct_popqa_seed_2025_top_1024_folds_3_probes.pkl \
--val_accs_path RAG/probes/llama3_8B_instruct_popqa_user_noise_answer_end/llama3_8B_instruct_popqa_seed_2025_top_1024_folds_3_val_accs.npy \
--tuning_headwise_path RAG/features/llama3_8B_instruct_popqa_user_noise_answer_end/llama3_8B_instruct_popqa_head_wise.npy \
--tuning_labels_path RAG/features/llama3_8B_instruct_popqa_user_noise_answer_end/llama3_8B_instruct_popqa_labels.npy \
--scores_csv RAG/probes/llama3_8B_instruct_popqa_user_noise_answer_end/accs_csv.csv \
--alphas 5 --probe_factor_modes false --max_new_tokens 256 --include_strategies topk_128_by_score --results_root RAG/results/llama-3-8b-instruct-nq-user/topk_128_by_score_alphas_5 --sample_size 100

## CoN
- 示例命令：
```bash
python RAG/con_rag.py --model_name llama3_8B_instruct --dataset_path RAG/data/PopQA/test_noise_test_noise4.jsonl --use_chat_template --max_docs 5 --sample_size 300 --max_new_tokens 256 --results_root RAG/results/llama3/llama-3-8b-instruct-popqa-user-noise/CON_300
```

## naive LLM
- 示例命令：
```bash
python RAG/naive_llm.py \
--model_name llama3_8B_instruct \
--dataset_path RAG/data/PopQA/test_noise_test_noise4.jsonl \
--use_chat_template \
--sample_size 300 \
--max_new_tokens 256 \
--results_root RAG/results/llama3/llama-3-8b-instruct-popqa-user-noise/naive-llm
```
