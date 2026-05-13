"""
DINO pretraining smoke test — CPU-friendly reduced config.

Uses the 10 HLS patches from self_Vits_SITS/data/hls_test_10/.
Runs 5 epochs to verify the full pipeline end-to-end.
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from training.dino_pretrain import DINOTrainer

DATA_DIR   = "D:/OneDrive - CGIAR/scripts/self_Vits_SITS/data/hls_test_10"
OUTPUT_DIR = "runs/dino_test"

# CPU-friendly config:
#   patch_size=3  ->  num_patches = (48/3)^2 = 256  (vs 2304 with patch_size=1)
#   dim=64, small heads -> tiny attention matrices
#   2 local crops only -> 4 total forward passes per sample

MODEL_CFG = {
    'img_res':        48,
    'patch_size':     3,      # key: reduces spatial patches from 2304 -> 256
    'num_classes':    21,
    'max_seq_len':    24,
    'dim':            64,
    'temporal_depth': 2,
    'spatial_depth':  1,
    'heads':          4,
    'dim_head':       16,
    'dropout':        0.1,
    'emb_dropout':    0.1,
    'pool':           'cls',
    'num_channels':   8,      # 7 bands + 1 DOY
    'scale_dim':      4,
    'depth':          2,
}

trainer = DINOTrainer(
    model_config     = MODEL_CFG,
    patch_dir        = DATA_DIR,
    output_dir       = OUTPUT_DIR,
    n_months         = 6,
    n_bands          = 7,
    epochs           = 5,
    batch_size       = 2,
    lr               = 5e-4,
    weight_decay     = 0.04,
    dino_out_dim     = 256,     # small projection head for CPU
    teacher_momentum = 0.996,
    n_local_crops    = 2,       # 2 global + 2 local = 4 forward passes per sample
    num_workers      = 0,
    device           = "cpu",
)

print(f"Dataset size: {len(trainer.loader.dataset)} patch combinations")
print(f"Batches per epoch: {len(trainer.loader)}")
print("Starting DINO pretraining...\n")
trainer.train()
print("\nDone. Checkpoint saved to:", OUTPUT_DIR)
