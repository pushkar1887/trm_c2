#!/usr/bin/env bash
# Verifier v2 eval sweep against the PID401-ALIGNED dataset (canonical baseline = 125/400).
# Usage:  bash scripts/run_verifier_v2_eval_sweep.sh <verifier_ckpt.pt>
set -euo pipefail

VERIFIER_CKPT="${1:?pass the verifier checkpoint path as arg 1}"
PY=trm/Scripts/python.exe
CFG=checkpoints/TRM-FVR-Experiments/c2_geomaux_VALID_005_EOS0_AUG1000_step670_evalfull400_pid401/all_config.yaml
CKPT=D:/trm_c2/c2_geomaux_VALID_005_EOS0_AUG1000_step670_evalfull400_pid401/step670_evalfull400_pid401
LEDGER=reports/c2_geomaux_VALID_005_EOS0_AUG1000_step670_evalfull400_pid401/c2_geomaux_VALID_005_EOS0_AUG1000_step670_evalfull400_pid401_17col_ledger.csv
DATA=data/arc-agi-evaluation-full400-seed0-pid401aligned
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# tau=0.0 first, with per-candidate dump for held-out AUROC.
echo "===== tau_commit=0.0 (with candidate dump) ====="
$PY scripts/eval_with_verifier.py \
  --config "$CFG" --checkpoint "$CKPT" \
  --verifier-checkpoint "$VERIFIER_CKPT" \
  --reference-ledger "$LEDGER" \
  --dataset "$DATA" \
  --out-dir reports/verifier_v2_evalaligned_tau0 \
  --tau-commit 0.0 --use-d4 --dump-candidates --global-batch-size 8

for TAU in 0.5 1.0 2.0 3.0; do
  echo "===== tau_commit=$TAU ====="
  $PY scripts/eval_with_verifier.py \
    --config "$CFG" --checkpoint "$CKPT" \
    --verifier-checkpoint "$VERIFIER_CKPT" \
    --reference-ledger "$LEDGER" \
    --dataset "$DATA" \
    --out-dir "reports/verifier_v2_evalaligned_tau${TAU}" \
    --tau-commit "$TAU" --use-d4 --global-batch-size 8
done

echo "===== SWEEP DONE ====="
for d in reports/verifier_v2_evalaligned_tau*; do
  echo "--- $d ---"
  grep -E "exact \(|replacement_loss|verdict|both_fail \(verifier" "$d/rejection_or_keep.md" 2>/dev/null || true
done
