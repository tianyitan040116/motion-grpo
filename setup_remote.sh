#!/bin/bash
# One-shot remote bootstrap for the AutoDL box.
#
# Run this on the remote machine (NOT on your Mac) after the code tree and
# dataset have been uploaded to /root/autodl-tmp/motion-agent. It will:
#
#   1. Install the minimum Python deps the trainer actually imports.
#   2. Download the Motion-Agent ckpt zip (motionllm.pth + vqvae.pth + friends).
#   3. Download the GloVe vectors the evaluator uses.
#   4. Download the t2m/kit evaluator extractors that the reward model loads.
#
# All artifacts land under /root/autodl-tmp/motion-agent/. Re-running is
# safe -- each step skips work that's already done.
#
# Usage:
#   bash setup_remote.sh                      # full bootstrap
#   bash setup_remote.sh --skip-deps          # only download assets
#   bash setup_remote.sh --skip-downloads     # only install deps
#   bash setup_remote.sh --skip-glove         # everything except glove (it's the slowest)
#
# Notes:
# - GPU is NOT required for setup. nvidia-smi is checked at the end as info only.
# - gdown is used for Google Drive links and is installed first if missing.
# - All downloads are unzipped in-place and the source zip removed.

set -euo pipefail

REPO="/root/autodl-tmp/motion-agent"
PIP="/root/miniconda3/bin/pip"
PY="/root/miniconda3/bin/python"

# Parse flags
SKIP_DEPS=0; SKIP_DOWNLOADS=0; SKIP_GLOVE=0; SKIP_CKPT=0; SKIP_EXTRACTOR=0
for arg in "$@"; do
    case "$arg" in
        --skip-deps) SKIP_DEPS=1 ;;
        --skip-downloads) SKIP_DOWNLOADS=1 ;;
        --skip-glove) SKIP_GLOVE=1 ;;
        --skip-ckpt) SKIP_CKPT=1 ;;
        --skip-extractor) SKIP_EXTRACTOR=1 ;;
        *) echo "unknown flag: $arg" >&2; exit 1 ;;
    esac
done

cd "$REPO"
echo "=== motion-agent remote bootstrap @ $REPO ==="
echo

# ---------------------------------------------------------------------------
# 1. Python deps
# ---------------------------------------------------------------------------
if [ "$SKIP_DEPS" -eq 0 ]; then
    echo "[1/4] installing python deps ..."
    # The repo's requirements.txt is a frozen full-env snapshot from the
    # original author's machine (150+ packages). We only need the libs the
    # trainer actually imports.
    $PIP install --quiet --no-input \
        transformers==4.40.0 \
        peft==0.10.0 \
        tqdm \
        matplotlib \
        scipy \
        gdown
    echo "  installed versions:"
    $PIP list 2>/dev/null | grep -i -E "^(torch|transformers|peft|tqdm|matplotlib|scipy|numpy|gdown)\b" | sed 's/^/    /'
    echo
else
    echo "[1/4] skipped (--skip-deps)"
    echo
fi

# ---------------------------------------------------------------------------
# 2. Motion-Agent ckpt zip (motionllm.pth, vqvae.pth, support files)
# ---------------------------------------------------------------------------
if [ "$SKIP_DOWNLOADS" -eq 0 ] && [ "$SKIP_CKPT" -eq 0 ]; then
    echo "[2/4] downloading Motion-Agent ckpt ..."
    if [ -f "$REPO/ckpt/motionllm.pth" ]; then
        echo "  ckpt/motionllm.pth already present, skipping download"
    else
        cd "$REPO"
        # gdown 6+ resolves Drive 'view' links by default; the legacy
        # --fuzzy flag was removed.
        /root/miniconda3/bin/gdown \
            'https://drive.google.com/file/d/1Tagt2xUwv_h0JNMtrM_Ty1rWemkLF5jH/view' \
            -O motion_agent.zip
        unzip -q -o motion_agent.zip
        rm motion_agent.zip
        echo "  unzipped, ckpt/ now contains:"
        ls -la ckpt/ | sed 's/^/    /'
    fi
    echo
else
    echo "[2/4] skipped"
    echo
fi

# ---------------------------------------------------------------------------
# 3. GloVe vectors (~2GB, slow)
# ---------------------------------------------------------------------------
if [ "$SKIP_DOWNLOADS" -eq 0 ] && [ "$SKIP_GLOVE" -eq 0 ]; then
    echo "[3/4] downloading GloVe (~2GB, slow over Google Drive) ..."
    if [ -d "$REPO/glove" ] && [ -f "$REPO/glove/our_vab_data.npy" ]; then
        echo "  glove/ already present, skipping download"
    else
        cd "$REPO"
        /root/miniconda3/bin/gdown \
            'https://drive.google.com/file/d/1bCeS6Sh_mLVTebxIgiUHgdPrroW06mb6/view?usp=sharing' \
            -O glove.zip
        unzip -q -o glove.zip
        rm glove.zip
        echo "  glove/ ready:"
        ls -la glove/ | head -8 | sed 's/^/    /'
    fi
    echo
else
    echo "[3/4] skipped"
    echo
fi

# ---------------------------------------------------------------------------
# 4. T2M evaluator extractors (used by GRPORewardModel matching score)
# ---------------------------------------------------------------------------
if [ "$SKIP_DOWNLOADS" -eq 0 ] && [ "$SKIP_EXTRACTOR" -eq 0 ]; then
    echo "[4/4] downloading t2m + kit evaluator extractors ..."
    if [ -d "$REPO/checkpoints/t2m/Comp_v6_KLD005" ]; then
        echo "  checkpoints/t2m/Comp_v6_KLD005 already present, skipping download"
    else
        # The stub we uploaded earlier lives at
        # checkpoints/t2m/VQVAEV3_CB1024_CMT_H1024_NRES3/meta/{mean,std}.npy.
        # The download will overlay additional subdirs; the stub stays intact.
        mkdir -p "$REPO/checkpoints"
        cd "$REPO/checkpoints"
        /root/miniconda3/bin/gdown \
            'https://drive.google.com/file/d/1FIiqtkt4F-GVWmnBgtZnv9W3cPWS-oM-/view' \
            -O t2m.zip
        /root/miniconda3/bin/gdown \
            'https://drive.google.com/file/d/1KNU8CsMAnxFrwopKBBkC8jEULGLPBHQp/view' \
            -O kit.zip
        unzip -q -o t2m.zip
        unzip -q -o kit.zip
        rm t2m.zip kit.zip
        echo "  checkpoints/ ready:"
        ls "$REPO/checkpoints/" | sed 's/^/    /'
    fi
    echo
else
    echo "[4/4] skipped"
    echo
fi

# ---------------------------------------------------------------------------
# Final summary
# ---------------------------------------------------------------------------
echo "=== summary ==="
cd "$REPO"
for p in ckpt/motionllm.pth ckpt/vqvae.pth glove/our_vab_data.npy \
         checkpoints/t2m/Comp_v6_KLD005 \
         dataset/new_joint_vecs dataset/Mean.npy; do
    if [ -e "$p" ]; then
        if [ -d "$p" ]; then
            count=$(ls "$p" 2>/dev/null | wc -l)
            echo "  OK   $p   ($count entries)"
        else
            sz=$(stat -c%s "$p" 2>/dev/null || stat -f%z "$p")
            echo "  OK   $p   ($sz B)"
        fi
    else
        echo "  MISS $p"
    fi
done
echo
echo "disk usage:"
df -h /root/autodl-tmp | head -2 | sed 's/^/  /'
echo
echo "gpu status (info only):"
if command -v nvidia-smi >/dev/null 2>&1; then
    nvidia-smi -L 2>&1 | sed 's/^/  /'
else
    echo "  nvidia-smi not present"
fi

echo
echo "=== bootstrap done. next: ==="
cat <<EOF
  cd /root/autodl-tmp/motion-agent
  # quick reward audit (no model needed, ~1 min)
  /root/miniconda3/bin/python audit/audit_reward.py --check B --max-detector-samples 200

  # smoke (reward vs GT/noise/shuffle/static, no model needed)
  /root/miniconda3/bin/python audit/smoke_reward.py --n-captions 50

  # real GRPO training with prompt-mix on (needs GPU)
  /root/miniconda3/bin/python train_grpo.py \\
      --sft-checkpoint ckpt/motionllm.pth \\
      --vq-path ckpt/vqvae.pth \\
      --prompt-mix --mix-numeric 0.30 --mix-direction 0.40 --mix-pure 0.30 \\
      --exp-name grpo_p0p1_smoke \\
      --num-samples-per-prompt 4 --batch-size 4 \\
      --learning-rate 2e-6 --epochs 1
EOF
