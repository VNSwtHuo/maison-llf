from pathlib import Path
import copy
import random

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from torch.utils.data import Dataset, DataLoader

SEED         = 1399
FORECAST_GAP = 7
BATCH_SIZE   = 12
MAX_EPOCHS   = 500
PATIENCE     = 45
LR           = 2e-3
WEIGHT_DECAY = 1e-3
HIDDEN_DIM   = 24
DROPOUT      = 0.3
DECAY_RATE   = 0.08

TOP_X_VALUES = list(range(5, 47))
TOP_Y_VALUES = list(range(0, 8))

DEVICE   = torch.device("cuda" if torch.cuda.is_available() else "mps")
DATA_DIR = Path(__file__).parent
SHAP_DIR = DATA_DIR / f"catboost-shap{SEED}"

sis_targets = [f"sis-{i:02d}" for i in range(1, 7)]
ohs_targets = [f"ohs-{i:02d}" for i in range(1, 13)]

union_sensor: list       = []
union_demo:   list       = []
n_features:   int        = 0
gate_prior:   np.ndarray = np.empty(0)

def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def ordered_union(a: list, b: list) -> list:
    seen = set(a)
    return a + [x for x in b if x not in seen]

seed_everything(SEED)
print(f"Device: {DEVICE}")
print(f"Hyperparameters: HIDDEN_DIM={HIDDEN_DIM}, DROPOUT={DROPOUT}, WEIGHT_DECAY={WEIGHT_DECAY}, PATIENCE={PATIENCE}")
print(f"Search: TOP_X={TOP_X_VALUES[0]}-{TOP_X_VALUES[-1]}, TOP_Y={TOP_Y_VALUES[0]}-{TOP_Y_VALUES[-1]}")
print(f"Total combinations: {len(TOP_X_VALUES) * len(TOP_Y_VALUES)}")
print()

sis_sensor_all  = pd.read_csv(SHAP_DIR / "shap_catboost_sis_feature_importance.csv")["feature"].tolist()
ohs_sensor_all  = pd.read_csv(SHAP_DIR / "shap_catboost_ohs_feature_importance.csv")["feature"].tolist()
sis_demo_all    = pd.read_csv(SHAP_DIR / "shap_catboost_demo_sis_feature_importance.csv")["feature"].tolist()
ohs_demo_all    = pd.read_csv(SHAP_DIR / "shap_catboost_demo_ohs_feature_importance.csv")["feature"].tolist()

sis_sensor_shap = pd.read_csv(SHAP_DIR / "shap_catboost_sis_feature_importance.csv").set_index("feature")["weights"].to_dict()
ohs_sensor_shap = pd.read_csv(SHAP_DIR / "shap_catboost_ohs_feature_importance.csv").set_index("feature")["weights"].to_dict()
sis_demo_shap   = pd.read_csv(SHAP_DIR / "shap_catboost_demo_sis_feature_importance.csv").set_index("feature")["weights"].to_dict()
ohs_demo_shap   = pd.read_csv(SHAP_DIR / "shap_catboost_demo_ohs_feature_importance.csv").set_index("feature")["weights"].to_dict()

raw = pd.read_csv(
    DATA_DIR / "data" / "maison-llf-features.csv",
    parse_dates=["timestamp", "clinical-timestamp"],
).sort_values(["participant", "timestamp", "clinical-timestamp"]).reset_index(drop=True)

id_cols = ["participant", "timestamp", "clinical-timestamp"]
clinical_cols = (
    ["sis", "ohs", "oks", "tug", "chairstand"]
    + [f"sis-{i:02d}" for i in range(1, 7)]
    + [f"ohs-{i:02d}" for i in range(1, 13)]
    + [f"oks-{i:02d}" for i in range(1, 13)]
)
sensor_cols_all = [c for c in raw.columns if c not in id_cols + clinical_cols]

demo_raw     = pd.read_csv(DATA_DIR / "data" / "maison-llf-demographics.csv")
demo_encoded = demo_raw.copy()
for col in [c for c in demo_raw.columns if c != "participant" and not pd.api.types.is_numeric_dtype(demo_raw[c])]:
    cats    = sorted(demo_raw[col].dropna().unique())
    mapping = {c: float(i) for i, c in enumerate(cats)}
    demo_encoded[col] = demo_raw[col].map(mapping)
for col in [c for c in demo_raw.columns if c != "participant" and pd.api.types.is_numeric_dtype(demo_raw[c])]:
    demo_encoded[col] = demo_raw[col].astype(float)
demo_lookup = demo_encoded.set_index("participant").astype(float)

def build_forecast_examples(features_df: pd.DataFrame, demo_lkp: pd.DataFrame, gap_days: int) -> list:
    examples = []
    for patient, pdf in features_df.groupby("participant", sort=True):
        pdf      = pdf.sort_values("timestamp")
        demo_row = demo_lkp.loc[patient].to_dict()
        for clinical_time, assess_rows in pdf.groupby("clinical-timestamp", sort=True):
            cutoff  = clinical_time - pd.Timedelta(days=gap_days)
            history = pdf.loc[pdf["timestamp"] <= cutoff].copy()
            if history.empty:
                continue
            tr = assess_rows.iloc[0]
            examples.append({
                "patient"        : int(patient),
                "clinical_time"  : clinical_time,
                "timestamps"     : history["timestamp"].to_numpy(),
                "x_sensor_frame" : history[sensor_cols_all].copy(),
                "demo_values"    : demo_row,
                "sis"            : tr[sis_targets].astype(float).to_numpy(),
                "ohs"            : tr[ohs_targets].astype(float).to_numpy(),
            })
    return examples

examples      = build_forecast_examples(raw, demo_lookup, FORECAST_GAP)
example_index = pd.DataFrame([
    {"patient": e["patient"], "clinical_time": e["clinical_time"]}
    for e in examples
])

all_patients   = np.array(sorted(example_index["patient"].unique()))
test_pts       = np.array([15, 6, 7])
remaining_pts  = np.setdiff1d(all_patients, test_pts)
rng            = np.random.default_rng(SEED)
rng.shuffle(remaining_pts)
train_pts      = remaining_pts[:12]
val_pts        = remaining_pts[12:]

def example_ids_for(patients) -> np.ndarray:
    return np.flatnonzero(example_index["patient"].isin(patients).to_numpy())

split = {
    "train_ids"     : example_ids_for(train_pts),
    "val_ids"       : example_ids_for(val_pts),
    "test_ids"      : example_ids_for(test_pts),
    "train_patients": train_pts.tolist(),
}

def compute_sensor_stats(train_ids: np.ndarray) -> dict:
    frames  = pd.concat(
        [examples[i]["x_sensor_frame"].reindex(columns=union_sensor) for i in train_ids],
        ignore_index=True,
    )
    median  = frames.median(numeric_only=True)
    imputed = frames.fillna(median)
    mean    = imputed.mean()
    std     = imputed.std(ddof=0).replace(0, 1.0).fillna(1.0)
    return {"median": median, "mean": mean, "std": std}

def compute_demo_stats(train_patients: list) -> dict:
    if not union_demo:
        return {"mean": pd.Series(dtype=float), "std": pd.Series(dtype=float)}
    dt   = demo_lookup.loc[demo_lookup.index.isin(train_patients), union_demo]
    mean = dt.mean()
    std  = dt.std(ddof=0).replace(0, 1.0).fillna(1.0)
    return {"mean": mean, "std": std}

def transform_inputs(example: dict, sensor_stats: dict, demo_stats: dict) -> np.ndarray:
    T      = len(example["timestamps"])
    s_df   = example["x_sensor_frame"].reindex(columns=union_sensor).fillna(sensor_stats["median"])
    s_norm = ((s_df - sensor_stats["mean"]) / sensor_stats["std"]).astype(np.float32).to_numpy()
    if not union_demo:
        return s_norm
    d_vals  = np.array([example["demo_values"][c] for c in union_demo], dtype=np.float32)
    d_norm  = ((d_vals - demo_stats["mean"].to_numpy()) / demo_stats["std"].to_numpy()).astype(np.float32)
    d_tiled = np.tile(d_norm, (T, 1))
    return np.concatenate([s_norm, d_tiled], axis=1)

class ForecastDataset(Dataset):
    def __init__(self, records: list, ids: np.ndarray, sensor_stats: dict, demo_stats: dict):
        self.records      = [records[i] for i in ids]
        self.sensor_stats = sensor_stats
        self.demo_stats   = demo_stats

    def __len__(self):
        return len(self.records)

    def __getitem__(self, idx):
        e = self.records[idx]
        x = transform_inputs(e, self.sensor_stats, self.demo_stats)
        return {
            "x"  : torch.tensor(x.tolist(), dtype=torch.float32),
            "sis": torch.tensor(e["sis"].tolist(), dtype=torch.float32),
            "ohs": torch.tensor(e["ohs"].tolist(), dtype=torch.float32),
        }

def make_collate_fn(n_feat: int):
    def collate_fn(batch: list) -> dict:
        lengths = torch.tensor([len(b["x"]) for b in batch], dtype=torch.long)
        max_len = int(lengths.max())
        x    = torch.zeros(len(batch), max_len, n_feat, dtype=torch.float32)
        mask = torch.zeros(len(batch), max_len, dtype=torch.bool)
        for i, b in enumerate(batch):
            L = int(lengths[i])
            x[i, :L]    = b["x"]
            mask[i, :L] = True
        return {
            "x"      : x,
            "mask"   : mask,
            "lengths": lengths,
            "sis"    : torch.stack([b["sis"] for b in batch]),
            "ohs"    : torch.stack([b["ohs"] for b in batch]),
        }
    return collate_fn

def make_loader(ids: np.ndarray, sensor_stats: dict, demo_stats: dict, shuffle: bool) -> DataLoader:
    return DataLoader(
        ForecastDataset(examples, ids, sensor_stats, demo_stats),
        batch_size=BATCH_SIZE,
        shuffle=shuffle,
        collate_fn=make_collate_fn(n_features),
    )

def inverse_softplus(x: np.ndarray) -> torch.Tensor:
    t = torch.tensor(np.asarray(x).tolist(), dtype=torch.float32).clamp_min(1e-4)
    return torch.log(torch.expm1(t))

class FeatureGate(nn.Module):
    def __init__(self, prior: np.ndarray):
        super().__init__()
        self.raw = nn.Parameter(inverse_softplus(prior))

    def forward(self, x: torch.Tensor):
        weights = torch.nn.functional.softplus(self.raw)
        weights = weights / weights.mean().clamp_min(1e-6)
        return x * weights, weights

class DecayAttention(nn.Module):
    def __init__(self, hidden_dim: int, decay_rate: float):
        super().__init__()
        self.score = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, 1),
        )
        self.raw_decay = nn.Parameter(
            inverse_softplus(np.array([decay_rate], dtype=np.float32))
        )

    def forward(self, encoded: torch.Tensor, mask: torch.Tensor, lengths: torch.Tensor):
        steps  = torch.arange(encoded.size(1), device=encoded.device).unsqueeze(0)
        age    = (lengths.to(encoded.device).unsqueeze(1) - 1 - steps).clamp_min(0).float()
        decay  = torch.nn.functional.softplus(self.raw_decay)
        logits = self.score(encoded).squeeze(-1) - decay * age
        logits = logits.masked_fill(~mask, -torch.finfo(encoded.dtype).max)
        attn   = torch.softmax(logits, dim=1)
        ctx    = torch.bmm(attn.unsqueeze(1), encoded).squeeze(1)
        return ctx, attn, decay

class MaisonTemporalForecaster(nn.Module):
    def __init__(self, n_feat: int, gate_p: np.ndarray, hidden_dim: int,
                 dropout: float, decay_rate: float):
        super().__init__()
        self.feature_gate = FeatureGate(gate_p)
        self.input_projection = nn.Sequential(
            nn.Linear(n_feat, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.gru = nn.GRU(hidden_dim, hidden_dim, batch_first=True)
        self.sis_attention = DecayAttention(hidden_dim, decay_rate)
        self.ohs_attention = DecayAttention(hidden_dim, decay_rate)
        self.sis_head = nn.Sequential(
            nn.LayerNorm(hidden_dim), nn.Dropout(dropout),
            nn.Linear(hidden_dim, len(sis_targets)),
        )
        self.ohs_head = nn.Sequential(
            nn.LayerNorm(hidden_dim), nn.Dropout(dropout),
            nn.Linear(hidden_dim, len(ohs_targets)),
        )

    def forward(self, x: torch.Tensor, mask: torch.Tensor, lengths: torch.Tensor) -> dict:
        gated_x, _    = self.feature_gate(x)
        projected      = self.input_projection(gated_x)
        packed         = pack_padded_sequence(projected, lengths.cpu(), batch_first=True, enforce_sorted=False)
        packed_enc, _  = self.gru(packed)
        encoded, _     = pad_packed_sequence(packed_enc, batch_first=True, total_length=x.size(1))
        sis_ctx, _, _  = self.sis_attention(encoded, mask, lengths)
        ohs_ctx, _, _  = self.ohs_attention(encoded, mask, lengths)
        return {
            "sis": self.sis_head(sis_ctx),
            "ohs": self.ohs_head(ohs_ctx),
        }

def make_model() -> MaisonTemporalForecaster:
    return MaisonTemporalForecaster(
        n_features, gate_prior, HIDDEN_DIM, DROPOUT, DECAY_RATE
    ).to(DEVICE)

_loss_fn = nn.SmoothL1Loss()

def task_loss(output: dict, batch: dict) -> torch.Tensor:
    return _loss_fn(output["sis"], batch["sis"]) + _loss_fn(output["ohs"], batch["ohs"])

def move_batch(batch: dict) -> dict:
    return {k: v.to(DEVICE) if torch.is_tensor(v) else v for k, v in batch.items()}

def run_epoch(model, optimizer, loader, training: bool) -> float:
    model.train(training)
    total, n = 0.0, 0
    for batch in loader:
        batch = move_batch(batch)
        with torch.set_grad_enabled(training):
            output = model(batch["x"], batch["mask"], batch["lengths"])
            loss   = task_loss(output, batch)
            if training:
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
        total += loss.item() * len(batch["x"])
        n     += len(batch["x"])
    return total / n

def full_metrics(rows: list) -> dict:
    truth   = np.array([r[0] for r in rows])
    pred    = np.array([r[1] for r in rows])
    resid   = truth - pred
    mae     = float(np.abs(resid).mean())
    rmse    = float(np.sqrt((resid ** 2).mean()))
    sst     = float(np.square(truth - truth.mean()).sum())
    r2      = float(1 - np.square(resid).sum() / sst) if sst > 0 else float("nan")
    pearson = (
        float(np.corrcoef(truth, pred)[0, 1])
        if len(truth) > 1 and np.std(truth) > 0 and np.std(pred) > 0
        else float("nan")
    )
    return {"mae": mae, "rmse": rmse, "r2": r2, "pearson": pearson}

def evaluate(model, ids: np.ndarray, sensor_stats: dict, demo_stats: dict) -> dict:
    model.eval()
    sis_item_rows, ohs_item_rows   = [], []
    sis_total_rows, ohs_total_rows = [], []
    loader = make_loader(ids, sensor_stats, demo_stats, shuffle=False)
    with torch.no_grad():
        for batch in loader:
            b   = move_batch(batch)
            out = model(b["x"], b["mask"], b["lengths"])
            for i in range(len(batch["sis"])):
                st = batch["sis"][i].cpu().numpy()
                sp = out["sis"][i].cpu().numpy()
                ot = batch["ohs"][i].cpu().numpy()
                op = out["ohs"][i].cpu().numpy()
                for t, p in zip(st, sp):
                    sis_item_rows.append((float(t), float(p)))
                for t, p in zip(ot, op):
                    ohs_item_rows.append((float(t), float(p)))
                sis_total_rows.append((float(st.sum()), float(sp.sum())))
                ohs_total_rows.append((float(ot.sum()), float(op.sum())))

    si = full_metrics(sis_item_rows)
    oi = full_metrics(ohs_item_rows)
    st = full_metrics(sis_total_rows)
    ot = full_metrics(ohs_total_rows)

    return {
        "sis_item_mae"    : si["mae"],
        "sis_item_rmse"   : si["rmse"],
        "sis_item_r2"     : si["r2"],
        "sis_item_pearson": si["pearson"],
        "sis_total_mae"   : st["mae"],
        "sis_total_rmse"  : st["rmse"],
        "sis_total_r2"    : st["r2"],
        "sis_total_pearson": st["pearson"],
        "ohs_item_mae"    : oi["mae"],
        "ohs_item_rmse"   : oi["rmse"],
        "ohs_item_r2"     : oi["r2"],
        "ohs_item_pearson": oi["pearson"],
        "ohs_total_mae"   : ot["mae"],
        "ohs_total_rmse"  : ot["rmse"],
        "ohs_total_r2"    : ot["r2"],
        "ohs_total_pearson": ot["pearson"],
    }

def train_and_eval(split: dict) -> tuple:
    seed_everything(SEED)
    sensor_stats = compute_sensor_stats(split["train_ids"])
    demo_stats   = compute_demo_stats(split["train_patients"])
    train_loader = make_loader(split["train_ids"], sensor_stats, demo_stats, shuffle=True)
    val_loader   = make_loader(split["val_ids"],   sensor_stats, demo_stats, shuffle=False)
    model     = make_model()
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    best_val   = float("inf")
    best_state = copy.deepcopy(model.state_dict())
    wait       = 0
    epoch      = 0
    for epoch in range(1, MAX_EPOCHS + 1):
        run_epoch(model, optimizer, train_loader, training=True)
        val_loss = run_epoch(model, optimizer, val_loader, training=False)
        if val_loss < best_val - 1e-5:
            best_val   = val_loss
            best_state = copy.deepcopy(model.state_dict())
            wait       = 0
        else:
            wait += 1
        if wait >= PATIENCE:
            break
    model.load_state_dict(best_state)
    val_metrics  = evaluate(model, split["val_ids"],  sensor_stats, demo_stats)
    test_metrics = evaluate(model, split["test_ids"], sensor_stats, demo_stats)
    return val_metrics, test_metrics, epoch

def fmt(m: dict, prefix: str) -> dict:
    return {
        f"{prefix}_sis_item_mae"     : round(m["sis_item_mae"],     4),
        f"{prefix}_sis_item_rmse"    : round(m["sis_item_rmse"],    4),
        f"{prefix}_sis_item_r2"      : round(m["sis_item_r2"],      4),
        f"{prefix}_sis_item_pearson" : round(m["sis_item_pearson"], 4),
        f"{prefix}_sis_total_mae"    : round(m["sis_total_mae"],    4),
        f"{prefix}_sis_total_rmse"   : round(m["sis_total_rmse"],   4),
        f"{prefix}_sis_total_r2"     : round(m["sis_total_r2"],     4),
        f"{prefix}_sis_total_pearson": round(m["sis_total_pearson"],4),
        f"{prefix}_ohs_item_mae"     : round(m["ohs_item_mae"],     4),
        f"{prefix}_ohs_item_rmse"    : round(m["ohs_item_rmse"],    4),
        f"{prefix}_ohs_item_r2"      : round(m["ohs_item_r2"],      4),
        f"{prefix}_ohs_item_pearson" : round(m["ohs_item_pearson"], 4),
        f"{prefix}_ohs_total_mae"    : round(m["ohs_total_mae"],    4),
        f"{prefix}_ohs_total_rmse"   : round(m["ohs_total_rmse"],   4),
        f"{prefix}_ohs_total_r2"     : round(m["ohs_total_r2"],     4),
        f"{prefix}_ohs_total_pearson": round(m["ohs_total_pearson"],4),
    }

results   = []
out_csv   = DATA_DIR / f"seed{SEED}_search_combination_xy.csv"
n_combos  = len(TOP_X_VALUES) * len(TOP_Y_VALUES)
combo_idx = 0

for top_x in TOP_X_VALUES:
    for top_y in TOP_Y_VALUES:
        combo_idx += 1

        union_sensor = ordered_union(sis_sensor_all[:top_x], ohs_sensor_all[:top_x])
        union_demo   = ordered_union(sis_demo_all[:top_y], ohs_demo_all[:top_y])
        n_features   = len(union_sensor) + len(union_demo)

        s_sis = np.array([sis_sensor_shap[c] for c in union_sensor], dtype=np.float32)
        s_ohs = np.array([ohs_sensor_shap[c] for c in union_sensor], dtype=np.float32)
        if union_demo:
            d_sis    = np.array([sis_demo_shap[c] for c in union_demo], dtype=np.float32)
            d_ohs    = np.array([ohs_demo_shap[c] for c in union_demo], dtype=np.float32)
            comb_sis = np.concatenate([s_sis, d_sis])
            comb_ohs = np.concatenate([s_ohs, d_ohs])
        else:
            comb_sis = s_sis
            comb_ohs = s_ohs
        sis_p      = (comb_sis / comb_sis.mean()).astype(np.float32)
        ohs_p      = (comb_ohs / comb_ohs.mean()).astype(np.float32)
        gate_prior = ((sis_p + ohs_p) / 2).astype(np.float32)
        gate_prior = (gate_prior / gate_prior.mean()).astype(np.float32)

        val_metrics, test_metrics, epochs = train_and_eval(split)
        rank_score = val_metrics["sis_item_mae"] + val_metrics["ohs_item_mae"]

        row = {
            "top_x"     : top_x,
            "top_y"     : top_y,
            "n_sensor"  : len(union_sensor),
            "n_demo"    : len(union_demo),
            "n_features": n_features,
            "rank_score": round(rank_score, 4),
            **fmt(val_metrics,  "val"),
            **fmt(test_metrics, "test"),
            "epochs"    : epochs,
        }
        results.append(row)

        print(
            f"[{combo_idx:3d}/{n_combos}] "
            f"top_x={top_x:2d} top_y={top_y} n_feat={n_features:2d} | "
            f"VAL  SIS={val_metrics['sis_item_mae']:.3f} OHS={val_metrics['ohs_item_mae']:.3f} rank={rank_score:.3f} | "
            f"TEST SIS={test_metrics['sis_item_mae']:.3f} OHS={test_metrics['ohs_item_mae']:.3f} "
            f"(ep={epochs})"
        )

        pd.DataFrame(results).sort_values("rank_score").reset_index(drop=True).to_csv(out_csv, index=False)

results_df = pd.DataFrame(results).sort_values("rank_score").reset_index(drop=True)
results_df.to_csv(out_csv, index=False)

print()
print("=" * 70)
print(f"Results saved → {out_csv}")
print()
display_cols = [
    "top_x", "top_y", "n_features", "rank_score",
    "val_sis_item_mae", "val_ohs_item_mae",
    "test_sis_item_mae", "test_ohs_item_mae",
    "test_sis_item_pearson", "test_ohs_item_pearson",
    "epochs",
]
print("=== Top 10 by val per-item SIS MAE + OHS MAE ===")
print(results_df.head(10)[display_cols].to_string(index=False))
print()
best = results_df.iloc[0]
print(
    f"BEST  TOP_X_SENSOR={int(best['top_x'])}  "
    f"TOP_Y_DEMO={int(best['top_y'])}  "
    f"n_features={int(best['n_features'])}  "
    f"rank_score(val)={best['rank_score']:.4f}  "
    f"test_SIS_item_MAE={best['test_sis_item_mae']:.4f}  "
    f"test_OHS_item_MAE={best['test_ohs_item_mae']:.4f}"
)
