# Resource utilization plan (L40S 46GB, 144GB RAM)

Current training uses ~1.3 GB GPU, ~8 GB RAM (main) + ~33 GB (6 workers), 53% GPU util. The machine can do more — either **faster single runs** or **richer training / multiple experiments**.

---

## Option A: Same model, faster training (quick wins)

Goal: fill the GPU and shorten wall-clock time per run.

| Knob | Current | Suggested | Effect |
|------|---------|-----------|--------|
| `--batch` | 32 | **128** or **256** | 4–8× fewer steps per epoch; better GPU occupancy. Try 128 first; if GPU stays &lt;50% util, try 256. |
| `--num_workers` | 6 | **8** or **10** | Keeps GPU fed when batch size is larger. RAM can easily handle 8–10 workers. |
| `--epochs` | 200 | 200 (unchanged) | Or 300 if you want more passes; total time may still be less due to bigger batches. |

**LR:** With 4× batch (32→128), you can try **linear scaling**: `--lr 4e-3`. If loss is unstable, use `--lr 2e-3` or keep `1e-3`.

**Example — higher throughput single run:**
```bash
python snn/training/train_lgmd.py \
  --h5 /tmp/evasion_head_on_events/events.h5 \
       /tmp/evasion_lateral_events/events.h5 \
       /tmp/evasion_high_events/events.h5 \
       /tmp/evasion_low_events/events.h5 \
  --val_h5 /tmp/evasion_diagonal_events/events.h5 \
  --augment --save results/lgmd_weights.pt \
  --batch 128 --num_workers 8 --lr 4e-3 --epochs 200
```

---

## Option B: Richer single run (more capacity / data)

Goal: use more GPU and RAM for a **better model** or **more data**, not just speed.

| Knob | Current | Suggested | Effect |
|------|---------|-----------|--------|
| `--pool` | 4 | **2** | Input 130×173 instead of 65×87 → ~4× spatial elements; more VRAM and compute. |
| `--n_bins` | 50 | **75** or **100** | Longer temporal context per window; more VRAM per sample. |
| `--stride_bins` | 10 | **5** | More overlapping windows → more training samples per recording; higher RAM for pre-encoded cache. |
| `--batch` | 32 | **64** or **32** | With pool=2 or n_bins=100, keep batch smaller to fit VRAM; monitor with `nvidia-smi`. |
| `--epochs` | 200 | **300–400** | More passes over data. |

**Example — higher resolution + more windows:**
```bash
python snn/training/train_lgmd.py \
  --h5 /tmp/evasion_head_on_events/events.h5 \
       /tmp/evasion_lateral_events/events.h5 \
       /tmp/evasion_high_events/events.h5 \
       /tmp/evasion_low_events/events.h5 \
  --val_h5 /tmp/evasion_diagonal_events/events.h5 \
  --augment --save results/lgmd_weights_hr.pt \
  --pool 2 --n_bins 75 --stride_bins 5 --batch 64 \
  --num_workers 8 --epochs 300
```

---

## Option C: Multiple experiments in parallel

Goal: use spare CPU/RAM to run **several configs** at once (e.g. different `--tau`, `--save`, or seeds). GPU runs stay single-process; run 2–3 training jobs **sequentially** or use **CUDA_VISIBLE_DEVICES** if you ever have multiple GPUs.

**Single GPU:** Run one training at a time; use the same machine to:
- **Sweep hyperparameters** one after another (e.g. `tau=3,5,7` or `lr=5e-4,1e-3,2e-3`) with different `--save` paths.
- **Add more data**: generate more profiles or longer runs, then train with Option A or B.

**Example — two saves, same data (run one after the other):**
```bash
# Run 1: default tau
python snn/training/train_lgmd.py --h5 ... --val_h5 ... --augment \
  --save results/lgmd_tau5.pt --epochs 200

# Run 2: different tau (or lr)
python snn/training/train_lgmd.py --h5 ... --val_h5 ... --augment \
  --save results/lgmd_tau3.pt --tau 3 --epochs 200
```

---

## Option D: Mixed precision (future code change)

**torch.cuda.amp** (Automatic Mixed Precision) would:
- Reduce VRAM per batch (FP16 activations).
- Often 1.5–2× faster on L40S.

This needs a small code change in `train_lgmd.py` (GradScaler + autocast). Not required for the plan above but is the next step if you want to push throughput or batch size further.

---

## Suggested order of operations

1. **Now (no code change):** Run with `--batch 128 --num_workers 8 --lr 4e-3`. Check `nvidia-smi` mid-run; if GPU memory is still low, try `--batch 256`.
2. **If that’s stable:** Try Option B (e.g. `--pool 2 --batch 64`) for a higher-capacity run.
3. **Then:** Use the same machine for multiple `--save` runs with different hyperparameters (Option C).
4. **Optional:** Add AMP (Option D) and then push `--batch` or `--pool` further.

---

## Is the machine “overkill”?

- **For the current default config (batch 32, pool 4, 5 H5s):** Yes — the L40S and 144 GB RAM are far from saturated.
- **To get more out of it:** Option A (bigger batch + workers) gives faster iteration; Option B (resolution/windows/epochs) uses more capacity per run; Option C uses the machine for systematic sweeps. Together they make the hardware worthwhile without changing the codebase much.
