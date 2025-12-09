import os
import argparse
import numpy as np
from einops import rearrange
import pickle

import sys
sys.path.append('./RAG')
from llama_utils import (
    get_separated_activations_nq,
    train_probes,
    flattened_idx_to_layer_head,
    layer_head_to_flattened_idx,
    save_probes,
)


def main():
    """
    训练并筛选前 top-k 个探针，并保存到磁盘，便于后续实验复用。

    使用从 NQ 数据集生成的特征（路径可通过 --feat_dir 指定）：
    - `{feat_dir}/{model_name}_nq_labels.npy`
    - `{feat_dir}/{model_name}_nq_head_wise.npy`
    - `{feat_dir}/{model_name}_nq_tokens.pkl`（可选，分析定位用）

    处理流程：
    1. 加载 head-wise 激活并 reshape 为 `(B, L, H, D)`；
    2. 基于样本数推断问题数（每题 2 个样本：正确+错误），拆分为每题一组；
    3. 随机划分训练/验证集合索引；
    4. 调用 `train_probes` 训练 LR 探针，按验证集准确率排序选出 top-k；
    5. 保存：探针列表、top-k 头列表、准确率数组（路径可通过 --save_dir 指定）。

    初学者注释：探针是一个简单的线性分类器（逻辑回归），输入是注意力头的激活向量，输出是该向量是否更“像”正确答案的二分类结果。
    """

    parser = argparse.ArgumentParser()
    parser.add_argument('--model_name', type=str, default='llama_7B')
    parser.add_argument('--num_heads', type=int, default=None, help='模型的注意力头数（用于 reshape），默认从特征维度推断')
    parser.add_argument('--head_dim', type=int, default=128, help='每个注意力头的维度（默认 128，与 LLAMA 一致）')
    parser.add_argument('--top_k', type=int, default=1024, help='选择并保存的前 k 个探针')
    parser.add_argument('--seed', type=int, default=2025, help='随机种子')
    parser.add_argument('--val_ratio', type=float, default=0.2, help='验证集比例')
    parser.add_argument('--num_fold', type=int, default=3, help='可选：K折交叉验证的折数，>1时启用；=1时仅随机划分')
    parser.add_argument(
        '--cv_final_train',
        type=str,
        default='full',
        choices=['none', 'full'],
        help='当 --num_fold>1 时，是否在折均值后用全数据重训探针并保存：full=用全数据重训（推荐），none=保存第一折的探针（原行为）'
    )
    # 新增：输入特征文件夹参数
    parser.add_argument('--feat_dir', type=str, default='../RAG-llm/RAG/features', 
                        help='特征文件所在的输入文件夹路径（默认：../RAG-llm/RAG/features）')
    # 新增：输出结果文件夹参数
    parser.add_argument('--save_dir', type=str, default='../RAG-llm/RAG/results_dump/probes',
                        help='探针及结果保存的输出文件夹路径（默认：../RAG-llm/RAG/results_dump/probes）')
    args = parser.parse_args()

    np.random.seed(args.seed)

    # 路径准备（使用参数指定的文件夹）
    labels_path = os.path.join(args.feat_dir, f'{args.model_name}_nq_labels.npy')
    head_path = os.path.join(args.feat_dir, f'{args.model_name}_nq_head_wise.npy')

    if not (os.path.exists(labels_path) and os.path.exists(head_path)):
        raise FileNotFoundError(f'未找到 NQ 特征文件，请先运行 RAG/llama_get_activations.py --dataset_name nq 收集特征，或检查 --feat_dir 参数是否正确。缺失文件：\n- {labels_path}\n- {head_path}')

    labels = np.load(labels_path)
    head_wise = np.load(head_path)

    # 推断维度：head_wise 原形状为 `(B, L, H*D)`，需 reshape 为 `(B, L, H, D)`
    B, L, HD = head_wise.shape
    if args.num_heads is None:
        if HD % args.head_dim != 0:
            raise ValueError('无法从特征维推断 num_heads，请显式传入 --num_heads')
        num_heads = HD // args.head_dim
    else:
        num_heads = args.num_heads

    head_wise = rearrange(head_wise, 'b l (h d) -> b l h d', h=num_heads, d=args.head_dim)

    # NQ：每题两个样本（正确/错误），问题数 = B / 2
    if B % 2 != 0:
        raise ValueError('样本数不是偶数，无法按 NQ 规则每题两样本拆分')
    num_questions = B // 2

    # 按题拆分（用于探针训练）
    separated_head, separated_labels, split_idxs = get_separated_activations_nq(labels, head_wise, num_questions)

    # 支持 K折交叉验证或单次划分
    if args.num_fold and args.num_fold > 1:
        fold_idxs = np.array_split(np.arange(num_questions), args.num_fold)
        accs_folds = []
        # 保留每折的探针列表，最后取平均指标选 top-k
        probes_all = None
        accs_all_flat = None
        for i in range(args.num_fold):
            train_idxs = np.concatenate([fold_idxs[j] for j in range(args.num_fold) if j != i])
            val_idxs = fold_idxs[i]
            probes, accs_np = train_probes(args.seed, train_idxs, val_idxs, separated_head, separated_labels, num_layers=L, num_heads=num_heads)
            accs_folds.append(accs_np)
            # 第一次记录探针（探针训练不跨折合并，以验证为主选头）
            if probes_all is None:
                probes_all = probes
        # 将各折准确率做平均作为最终排序依据
        accs_mean = np.mean(np.stack(accs_folds, axis=0), axis=0)
        scores = accs_mean.reshape(L * num_heads)
        top_flat = np.argsort(scores)[::-1][:args.top_k]
        top_heads = [flattened_idx_to_layer_head(idx, num_heads) for idx in top_flat]
        accs_to_save = accs_mean.reshape(L, num_heads)

        # 交叉验证后最终保存的探针：
        # - full：用“全问题样本”重训所有头的探针，保证与折均分数的选择逻辑一致（推荐）
        # - none：保留第一折训练的探针（原行为）
        if args.cv_final_train == 'full':
            all_q_idxs = np.arange(num_questions)
            # 复用 train_probes：val 同样取全数据以便函数内部计算不报错；返回的 accs 不用于最终保存
            final_probes, _ = train_probes(
                args.seed,
                all_q_idxs,
                all_q_idxs,
                separated_head,
                separated_labels,
                num_layers=L,
                num_heads=num_heads,
            )
            probes_to_save = final_probes
            print('[Info] CV 后使用全数据重训并保存探针（cv_final_train=full）')
        else:
            probes_to_save = probes_all
            print('[Info] CV 后保留第一折训练的探针（cv_final_train=none）')
    else:
        # 单次随机划分
        q_idxs = np.arange(num_questions)
        train_size = int(num_questions * (1 - args.val_ratio))
        train_idxs = np.random.choice(q_idxs, size=train_size, replace=False)
        val_idxs = np.array([x for x in q_idxs if x not in train_idxs])
        probes, accs_np = train_probes(args.seed, train_idxs, val_idxs, separated_head, separated_labels, num_layers=L, num_heads=num_heads)
        scores = accs_np.reshape(L * num_heads)
        top_flat = np.argsort(scores)[::-1][:args.top_k]
        top_heads = [flattened_idx_to_layer_head(idx, num_heads) for idx in top_flat]
        probes_to_save = probes
        accs_to_save = accs_np.reshape(L, num_heads)

    # 保存探针与 top-k 结果（使用参数指定的输出文件夹）
    os.makedirs(args.save_dir, exist_ok=True)
    base = f'{args.model_name}_nq_seed_{args.seed}_top_{args.top_k}'
    if args.num_fold and args.num_fold > 1:
        base += f'_folds_{args.num_fold}'
    probes_path = os.path.join(args.save_dir, base + '_probes.pkl')
    top_heads_path = os.path.join(args.save_dir, base + '_top_heads.pkl')
    accs_path = os.path.join(args.save_dir, base + '_val_accs.npy')

    save_probes(probes_to_save, probes_path)
    with open(top_heads_path, 'wb') as f:
        pickle.dump(top_heads, f)
    np.save(accs_path, accs_to_save)

    print('Saved:')
    print(' - probes:', probes_path)
    print(' - top_heads:', top_heads_path)
    print(' - val_accs:', accs_path)


if __name__ == '__main__':
    main()