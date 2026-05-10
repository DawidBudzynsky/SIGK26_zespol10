#!/bin/bash
# Pelny pipeline lab3: ewaluacja + wizualizacje.
# Wywoluj z /root/lab3.
set -e

CKPT_GAN=${CKPT_GAN:-lab3/checkpoints/cgan.pt}
CKPT_L1=${CKPT_L1:-lab3/checkpoints/l1_only.pt}
DATA=${DATA:-lab3/data}

echo "[1/5] eval cGAN"
python3 lab3/evaluate.py --ckpt $CKPT_GAN --out_dir lab3/results --data $DATA

echo "[2/5] eval L1-only"
if [ -f $CKPT_L1 ]; then
  python3 lab3/evaluate.py --ckpt $CKPT_L1 --out_dir lab3/results/l1_only --data $DATA
else
  echo "  (skipped; $CKPT_L1 not found)"
fi

echo "[3/5] metric extremes"
python3 lab3/scripts/visualize.py extremes --ckpt $CKPT_GAN --csv lab3/results/metrics_per_sample.csv --data $DATA

echo "[4/5] loss curves"
python3 lab3/scripts/visualize.py loss --log lab3/checkpoints/cgan.log.txt

echo "[5/5] cGAN vs L1 montage"
if [ -f $CKPT_L1 ]; then
  python3 lab3/scripts/compare_models.py --ckpt_gan $CKPT_GAN --ckpt_l1 $CKPT_L1 --data $DATA
else
  echo "  (skipped; $CKPT_L1 not found)"
fi

echo "done. wyniki: lab3/results/"
