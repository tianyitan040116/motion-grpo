"""
手动对 GRPO 训练的 checkpoint 做 validation
用法:
    python eval_grpo.py --checkpoint experiments_grpo/grpo_from_sft/motionllm_grpo_latest.pth
"""
import torch
import argparse
import os
os.environ['TOKENIZERS_PARALLELISM'] = 'false'

from models.mllm import MotionLLM
from utils.evaluation import evaluation_test
from dataset import dataset_TM_eval
from utils.word_vectorizer import WordVectorizer
from models.evaluator_wrapper import EvaluatorModelWrapper
from options.get_eval_option import get_opt


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', type=str, required=True, help='GRPO checkpoint path')
    parser.add_argument('--split', type=str, default='val', choices=['val', 'test'])
    parser.add_argument('--device', type=str, default='cuda:0')
    # VQ-VAE / model args (match training defaults)
    parser.add_argument('--llm-backbone', type=str, default='C:/Users/tianyi/Downloads/gemma-2-2b-it')
    parser.add_argument('--lora-r-t2m', type=int, default=64)
    parser.add_argument('--lora-alpha-t2m', type=int, default=64)
    parser.add_argument('--lora-r-m2t', type=int, default=32)
    parser.add_argument('--lora-alpha-m2t', type=int, default=32)
    parser.add_argument('--lora-dropout', type=float, default=0.1)
    parser.add_argument('--dataname', type=str, default='t2m')
    parser.add_argument('--code-dim', type=int, default=512)
    parser.add_argument('--nb-code', type=int, default=512)
    parser.add_argument('--mu', type=float, default=0.99)
    parser.add_argument('--down-t', type=int, default=2)
    parser.add_argument('--stride-t', type=int, default=2)
    parser.add_argument('--width', type=int, default=512)
    parser.add_argument('--depth', type=int, default=3)
    parser.add_argument('--dilation-growth-rate', type=int, default=3)
    parser.add_argument('--output-emb-width', type=int, default=512)
    parser.add_argument('--vq-act', type=str, default='relu')
    parser.add_argument('--vq-norm', type=str, default=None)
    parser.add_argument('--quantizer', type=str, default='ema_reset')
    parser.add_argument('--beta', type=float, default=1.0)
    parser.add_argument('--vq-path', type=str, default='ckpt/vqvae.pth')
    parser.add_argument('--out-dir', type=str, default='experiments_grpo/grpo_from_sft')
    parser.add_argument('--max-samples', type=int, default=None, help='Limit number of samples for quick test')
    return parser.parse_args()


def main():
    args = get_args()
    print(f"Loading model from: {args.checkpoint}")

    model = MotionLLM(args)
    model.load_model(args.checkpoint)
    model.eval()
    print("Model loaded.")

    w_vectorizer = WordVectorizer('./glove', 'our_vab')
    dataset_opt_path = 'checkpoints/t2m/Comp_v6_KLD005/opt.txt'
    wrapper_opt = get_opt(dataset_opt_path, args.device)
    eval_wrapper = EvaluatorModelWrapper(wrapper_opt)

    loader = dataset_TM_eval.DATALoader(
        args.dataname, args.split, 32, w_vectorizer, unit_length=2**args.down_t
    )

    # Limit samples for quick test
    if args.max_samples is not None:
        original_len = len(loader.dataset)
        # Keep only first N items in data_dict
        keys_to_keep = list(loader.dataset.data_dict.keys())[:args.max_samples]
        loader.dataset.data_dict = {k: loader.dataset.data_dict[k] for k in keys_to_keep}
        # Update name_list to match
        loader.dataset.name_list = keys_to_keep
        print(f"Limited to {args.max_samples} samples (original: {original_len})")

    print(f"Val samples: {len(loader.dataset)}, batches: {len(loader)}")

    fid, div, top1, top2, top3, matching, multi = evaluation_test(
        args.out_dir, loader, model, eval_wrapper=eval_wrapper, draw=False, savenpy=False
    )

    print("\n===== Validation Results =====")
    print(f"FID:      {fid:.4f}")
    print(f"Div:      {div:.4f}")
    print(f"Top1:     {top1:.4f}")
    print(f"Top2:     {top2:.4f}")
    print(f"Top3:     {top3:.4f}")
    print(f"Matching: {matching:.4f}")
    print(f"Multi:    {multi:.4f}")


if __name__ == '__main__':
    main()
