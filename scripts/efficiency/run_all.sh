# Full efficiency benchmark: CALF + CMIE (7 datasets, pred_len=336, bs=32)
# Usage: gpu=0 bash scripts/efficiency/run_all.sh
#
# Or parallel on different GPUs:
#   gpu=0 bash scripts/efficiency/run_calf_all.sh &
#   gpu=1 bash scripts/efficiency/run_cmie_all.sh &

echo "=============================================="
echo " FULL Efficiency Benchmark  (GPU=$gpu)"
echo "=============================================="

echo ""
echo ">>> Phase 1: CALF <<<"
bash scripts/efficiency/run_calf_all.sh

echo ""
echo ">>> Phase 2: CMIE (merge_layers 1..6) <<<"
bash scripts/efficiency/run_cmie_all.sh

echo ""
echo "All done.  Results: result_*.txt  and  checkpoints/*/result.txt"
