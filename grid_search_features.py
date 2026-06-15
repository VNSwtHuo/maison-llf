from pathlib import Path
import argparse
import copy
import os
import random
import tempfile

os.environ.setdefault("MPLCONFIGDIR", tempfile.mkdtemp(prefix="matplotlib-"))

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from torch.utils.data import DataLoader, Dataset


TOP_X_SENSOR = 46
TOP_Y_DEMO = 7

SEED = 42
FORECAST_GAP = 7
BATCH_SIZE = 12
MAX_EPOCHS = 500
PATIENCE = 45
LR = 2e-3
WEIGHT_DECAY = 1e-3
HIDDEN_DIM = 24
DROPOUT = 0.3
DECAY_RATE = 0.08
TOTAL_LOSS_WEIGHT = 0.0

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
DATA_DIR = Path.cwd()
SHAP_DIR = DATA_DIR / "catboost-shap"

sis_targets = [f"sis-{i:02d}" for i in range(1, 7)]
ohs_targets = [f"ohs-{i:02d}" for i in range(1, 13)]


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def load_ranked_features(csv_path: Path, n: int) -> list:
    return pd.read_csv(csv_path)["feature"].iloc[:n].tolist()


def ordered_union(list_a: list, list_b: list) -> list:
    seen = set(list_a)
    return list_a + [x for x in list_b if x not in seen]


def build_feature_selection(top_x_sensor: int, top_y_demo: int) -> dict:
    sis_sensor_top = load_ranked_features(
        SHAP_DIR / "shap_catboost_sis_feature_importance.csv", top_x_sensor
    )
    ohs_sensor_top = load_ranked_features(
        SHAP_DIR / "shap_catboost_ohs_feature_importance.csv", top_x_sensor
    )
    union_sensor = ordered_union(sis_sensor_top, ohs_sensor_top)

    sis_demo_top = load_ranked_features(
        SHAP_DIR / "shap_catboost_demo_sis_feature_importance.csv", top_y_demo
    )
    ohs_demo_top = load_ranked_features(
        SHAP_DIR / "shap_catboost_demo_ohs_feature_importance.csv", top_y_demo
    )
    union_demo = ordered_union(sis_demo_top, ohs_demo_top)
    input_cols = union_sensor + union_demo

    sis_prior = build_gate_prior(
        SHAP_DIR / "shap_catboost_sis_feature_importance.csv",
        SHAP_DIR / "shap_catboost_demo_sis_feature_importance.csv",
        union_sensor,
        union_demo,
    )
    ohs_prior = build_gate_prior(
        SHAP_DIR / "shap_catboost_ohs_feature_importance.csv",
        SHAP_DIR / "shap_catboost_demo_ohs_feature_importance.csv",
        union_sensor,
        union_demo,
    )
    gate_prior = ((sis_prior + ohs_prior) / 2).astype(np.float32)
    gate_prior = (gate_prior / gate_prior.mean()).astype(np.float32)

    return {
        "top_x_sensor": top_x_sensor,
        "top_y_demo": top_y_demo,
        "union_sensor": union_sensor,
        "union_demo": union_demo,
        "input_cols": input_cols,
        "n_features": len(input_cols),
        "gate_prior": gate_prior,
    }


def build_gate_prior(
    sensor_csv: Path,
    demo_csv: Path,
    sensor_cols: list,
    demo_cols: list,
    floor: float = 0.02,
) -> np.ndarray:
    sensor_shap = pd.read_csv(sensor_csv).set_index("feature")["weights"].to_dict()
    sensor_prior = np.array([sensor_shap.get(c, floor) for c in sensor_cols], dtype=np.float32)

    demo_shap = pd.read_csv(demo_csv).set_index("feature")["weights"].to_dict()
    demo_prior = np.array([demo_shap.get(c, floor) for c in demo_cols], dtype=np.float32)

    combined = np.concatenate([sensor_prior, demo_prior])
    return (combined / combined.mean()).astype(np.float32)


def load_data() -> tuple[pd.DataFrame, pd.DataFrame, list]:
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

    demo_raw = pd.read_csv(DATA_DIR / "data" / "maison-llf-demographics.csv")
    demo_encoded = demo_raw.copy()
    for col in [
        c
        for c in demo_raw.columns
        if c != "participant" and not pd.api.types.is_numeric_dtype(demo_raw[c])
    ]:
        cats = sorted(demo_raw[col].dropna().unique())
        mapping = {c: float(i) for i, c in enumerate(cats)}
        demo_encoded[col] = demo_raw[col].map(mapping)
    for col in [
        c
        for c in demo_raw.columns
        if c != "participant" and pd.api.types.is_numeric_dtype(demo_raw[c])
    ]:
        demo_encoded[col] = demo_raw[col].astype(float)

    demo_lookup = demo_encoded.set_index("participant").astype(float)
    return raw, demo_lookup, sensor_cols_all


def build_forecast_examples(
    features_df: pd.DataFrame,
    demo_lkp: pd.DataFrame,
    sensor_cols_all: list,
    gap_days: int,
) -> list:
    examples = []
    for patient, pdf in features_df.groupby("participant", sort=True):
        pdf = pdf.sort_values("timestamp")
        demo_row = demo_lkp.loc[patient].to_dict()

        for clinical_time, assess_rows in pdf.groupby("clinical-timestamp", sort=True):
            cutoff = clinical_time - pd.Timedelta(days=gap_days)
            history = pdf.loc[pdf["timestamp"] <= cutoff].copy()
            if history.empty:
                continue
            tr = assess_rows.iloc[0]
            examples.append(
                {
                    "patient": int(patient),
                    "clinical_time": clinical_time,
                    "cutoff": cutoff,
                    "timestamps": history["timestamp"].to_numpy(),
                    "x_sensor_frame": history[sensor_cols_all].copy(),
                    "demo_values": demo_row,
                    "sis": tr[sis_targets].astype(float).to_numpy(),
                    "ohs": tr[ohs_targets].astype(float).to_numpy(),
                    "sis_total": float(tr["sis"]),
                    "ohs_total": float(tr["ohs"]),
                }
            )
    return examples


def make_split(example_index: pd.DataFrame) -> dict:
    all_patients = np.array(sorted(example_index["patient"].unique()))
    rng = np.random.default_rng(SEED)
    shuffled = all_patients.copy()
    rng.shuffle(shuffled)

    train_pts = shuffled[:12]
    val_pts = shuffled[12:15]
    test_pts = shuffled[15:]

    def example_ids_for(patients) -> np.ndarray:
        return np.flatnonzero(example_index["patient"].isin(patients).to_numpy())

    return {
        "train_ids": example_ids_for(train_pts),
        "val_ids": example_ids_for(val_pts),
        "test_ids": example_ids_for(test_pts),
        "train_patients": train_pts.tolist(),
    }


def compute_sensor_stats(examples: list, train_ids: np.ndarray, feature_cfg: dict) -> dict:
    frames = pd.concat(
        [
            examples[i]["x_sensor_frame"].reindex(columns=feature_cfg["union_sensor"])
            for i in train_ids
        ],
        ignore_index=True,
    )
    median = frames.median(numeric_only=True)
    imputed = frames.fillna(median)
    mean = imputed.mean()
    std = imputed.std(ddof=0).replace(0, 1.0).fillna(1.0)
    return {"median": median, "mean": mean, "std": std}


def compute_demo_stats(demo_lookup: pd.DataFrame, train_patients: list, feature_cfg: dict) -> dict:
    dt = demo_lookup.loc[demo_lookup.index.isin(train_patients), feature_cfg["union_demo"]]
    mean = dt.mean()
    std = dt.std(ddof=0).replace(0, 1.0).fillna(1.0)
    return {"mean": mean, "std": std}


def transform_inputs(example: dict, sensor_stats: dict, demo_stats: dict, feature_cfg: dict) -> np.ndarray:
    t_steps = len(example["timestamps"])
    s_df = example["x_sensor_frame"].reindex(columns=feature_cfg["union_sensor"]).fillna(
        sensor_stats["median"]
    )
    s_norm = ((s_df - sensor_stats["mean"]) / sensor_stats["std"]).astype(np.float32).to_numpy()
    d_vals = np.array([example["demo_values"][c] for c in feature_cfg["union_demo"]], dtype=np.float32)
    d_norm = ((d_vals - demo_stats["mean"].to_numpy()) / demo_stats["std"].to_numpy()).astype(
        np.float32
    )
    d_tiled = np.tile(d_norm, (t_steps, 1))
    return np.concatenate([s_norm, d_tiled], axis=1)


class ForecastDataset(Dataset):
    def __init__(
        self,
        records: list,
        ids: np.ndarray,
        sensor_stats: dict,
        demo_stats: dict,
        feature_cfg: dict,
    ):
        self.records = [records[i] for i in ids]
        self.sensor_stats = sensor_stats
        self.demo_stats = demo_stats
        self.feature_cfg = feature_cfg

    def __len__(self):
        return len(self.records)

    def __getitem__(self, idx):
        e = self.records[idx]
        x = transform_inputs(e, self.sensor_stats, self.demo_stats, self.feature_cfg)
        return {
            "x": torch.tensor(x.tolist(), dtype=torch.float32),
            "sis": torch.tensor(e["sis"].tolist(), dtype=torch.float32),
            "ohs": torch.tensor(e["ohs"].tolist(), dtype=torch.float32),
            "patient": e["patient"],
            "clinical_time": e["clinical_time"],
        }


def make_collate_fn(n_features: int):
    def collate_fn(batch: list) -> dict:
        lengths = torch.tensor([len(b["x"]) for b in batch], dtype=torch.long)
        max_len = int(lengths.max())
        x = torch.zeros(len(batch), max_len, n_features, dtype=torch.float32)
        mask = torch.zeros(len(batch), max_len, dtype=torch.bool)
        for i, b in enumerate(batch):
            length = int(lengths[i])
            x[i, :length] = b["x"]
            mask[i, :length] = True
        return {
            "x": x,
            "mask": mask,
            "lengths": lengths,
            "sis": torch.stack([b["sis"] for b in batch]),
            "ohs": torch.stack([b["ohs"] for b in batch]),
            "patient": [b["patient"] for b in batch],
            "clinical_time": [b["clinical_time"] for b in batch],
        }

    return collate_fn


def make_loader(
    examples: list,
    ids: np.ndarray,
    sensor_stats: dict,
    demo_stats: dict,
    feature_cfg: dict,
    shuffle: bool,
) -> DataLoader:
    return DataLoader(
        ForecastDataset(examples, ids, sensor_stats, demo_stats, feature_cfg),
        batch_size=BATCH_SIZE,
        shuffle=shuffle,
        collate_fn=make_collate_fn(feature_cfg["n_features"]),
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
        self.raw_decay = nn.Parameter(inverse_softplus(np.array([decay_rate], dtype=np.float32)))

    def forward(self, encoded: torch.Tensor, mask: torch.Tensor, lengths: torch.Tensor):
        steps = torch.arange(encoded.size(1), device=encoded.device).unsqueeze(0)
        age = (lengths.to(encoded.device).unsqueeze(1) - 1 - steps).clamp_min(0).float()
        decay = torch.nn.functional.softplus(self.raw_decay)
        logits = self.score(encoded).squeeze(-1) - decay * age
        logits = logits.masked_fill(~mask, -torch.finfo(encoded.dtype).max)
        attn = torch.softmax(logits, dim=1)
        ctx = torch.bmm(attn.unsqueeze(1), encoded).squeeze(1)
        return ctx, attn, decay


class MaisonTemporalForecaster(nn.Module):
    def __init__(
        self,
        n_features: int,
        gate_prior: np.ndarray,
        hidden_dim: int,
        dropout: float,
        decay_rate: float,
    ):
        super().__init__()
        self.feature_gate = FeatureGate(gate_prior)
        self.input_projection = nn.Sequential(
            nn.Linear(n_features, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.gru = nn.GRU(hidden_dim, hidden_dim, batch_first=True)
        self.sis_attention = DecayAttention(hidden_dim, decay_rate)
        self.ohs_attention = DecayAttention(hidden_dim, decay_rate)
        self.sis_head = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, len(sis_targets)),
        )
        self.ohs_head = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, len(ohs_targets)),
        )

    def forward(self, x: torch.Tensor, mask: torch.Tensor, lengths: torch.Tensor) -> dict:
        gated_x, gate_w = self.feature_gate(x)
        projected = self.input_projection(gated_x)
        packed = pack_padded_sequence(
            projected, lengths.cpu(), batch_first=True, enforce_sorted=False
        )
        packed_enc, _ = self.gru(packed)
        encoded, _ = pad_packed_sequence(packed_enc, batch_first=True, total_length=x.size(1))
        sis_ctx, sis_attn, sis_decay = self.sis_attention(encoded, mask, lengths)
        ohs_ctx, ohs_attn, ohs_decay = self.ohs_attention(encoded, mask, lengths)
        return {
            "sis": self.sis_head(sis_ctx),
            "ohs": self.ohs_head(ohs_ctx),
            "sis_attention": sis_attn,
            "ohs_attention": ohs_attn,
            "feature_gate_weights": gate_w,
            "sis_decay": sis_decay,
            "ohs_decay": ohs_decay,
        }


def make_model(feature_cfg: dict) -> MaisonTemporalForecaster:
    return MaisonTemporalForecaster(
        feature_cfg["n_features"],
        feature_cfg["gate_prior"],
        HIDDEN_DIM,
        DROPOUT,
        DECAY_RATE,
    ).to(DEVICE)


_item_loss_fn = nn.SmoothL1Loss()
_total_loss_fn = nn.SmoothL1Loss()


def multitask_loss(output: dict, batch: dict) -> torch.Tensor:
    sis_item_loss = _item_loss_fn(output["sis"], batch["sis"])
    ohs_item_loss = _item_loss_fn(output["ohs"], batch["ohs"])
    item_loss = sis_item_loss + ohs_item_loss

    if TOTAL_LOSS_WEIGHT <= 0:
        return item_loss

    sis_total_loss = _total_loss_fn(output["sis"].mean(dim=1), batch["sis"].mean(dim=1))
    ohs_total_loss = _total_loss_fn(output["ohs"].mean(dim=1), batch["ohs"].mean(dim=1))
    total_loss = sis_total_loss + ohs_total_loss
    return item_loss + TOTAL_LOSS_WEIGHT * total_loss


def move_batch(batch: dict) -> dict:
    return {k: v.to(DEVICE) if torch.is_tensor(v) else v for k, v in batch.items()}


def run_epoch(
    model: MaisonTemporalForecaster,
    optimizer: torch.optim.Optimizer,
    loader: DataLoader,
    training: bool,
) -> float:
    model.train(training)
    total, n = 0.0, 0
    for batch in loader:
        batch = move_batch(batch)
        with torch.set_grad_enabled(training):
            output = model(batch["x"], batch["mask"], batch["lengths"])
            loss = multitask_loss(output, batch)
            if training:
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
        total += loss.item() * len(batch["x"])
        n += len(batch["x"])
    return total / n


def train_model(examples: list, demo_lookup: pd.DataFrame, split: dict, feature_cfg: dict) -> dict:
    seed_everything(SEED)
    sensor_stats = compute_sensor_stats(examples, split["train_ids"], feature_cfg)
    demo_stats = compute_demo_stats(demo_lookup, split["train_patients"], feature_cfg)

    train_loader = make_loader(
        examples, split["train_ids"], sensor_stats, demo_stats, feature_cfg, shuffle=True
    )
    val_loader = make_loader(
        examples, split["val_ids"], sensor_stats, demo_stats, feature_cfg, shuffle=False
    )

    model = make_model(feature_cfg)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)

    best_val = float("inf")
    best_state = copy.deepcopy(model.state_dict())
    wait = 0
    history = []

    for epoch in range(1, MAX_EPOCHS + 1):
        train_loss = run_epoch(model, optimizer, train_loader, training=True)
        val_loss = run_epoch(model, optimizer, val_loader, training=False)
        history.append({"epoch": epoch, "train_loss": train_loss, "val_loss": val_loss})

        if val_loss < best_val - 1e-5:
            best_val = val_loss
            best_state = copy.deepcopy(model.state_dict())
            wait = 0
        else:
            wait += 1
        if wait >= PATIENCE:
            break

    model.load_state_dict(best_state)
    sis_decay = float(torch.nn.functional.softplus(model.sis_attention.raw_decay.detach()))
    ohs_decay = float(torch.nn.functional.softplus(model.ohs_attention.raw_decay.detach()))

    return {
        "model": model,
        "sensor_stats": sensor_stats,
        "demo_stats": demo_stats,
        "train_ids": split["train_ids"],
        "val_ids": split["val_ids"],
        "test_ids": split["test_ids"],
        "history": pd.DataFrame(history),
        "best_val": best_val,
        "epochs": len(history),
        "sis_decay": sis_decay,
        "ohs_decay": ohs_decay,
    }


def collect_predictions(examples: list, artifact: dict, feature_cfg: dict, split_name: str) -> pd.DataFrame:
    rows = []
    model = artifact["model"]
    model.eval()
    ids = artifact[f"{split_name}_ids"]
    loader = make_loader(
        examples,
        ids,
        artifact["sensor_stats"],
        artifact["demo_stats"],
        feature_cfg,
        shuffle=False,
    )
    with torch.no_grad():
        for batch in loader:
            dev_batch = move_batch(batch)
            output = model(dev_batch["x"], dev_batch["mask"], dev_batch["lengths"])
            sis_pred = output["sis"].cpu().tolist()
            ohs_pred = output["ohs"].cpu().tolist()
            for i in range(len(batch["patient"])):
                base = {
                    "patient": batch["patient"][i],
                    "clinical_time": pd.Timestamp(batch["clinical_time"][i]),
                    "history_days": int(batch["lengths"][i]),
                }
                for name, truth, pred in zip(sis_targets, batch["sis"][i].tolist(), sis_pred[i]):
                    rows.append(
                        base
                        | {
                            "outcome": "SIS",
                            "item": name,
                            "truth": float(truth),
                            "prediction": float(pred),
                        }
                    )
                for name, truth, pred in zip(ohs_targets, batch["ohs"][i].tolist(), ohs_pred[i]):
                    rows.append(
                        base
                        | {
                            "outcome": "OHS",
                            "item": name,
                            "truth": float(truth),
                            "prediction": float(pred),
                        }
                    )
    return pd.DataFrame(rows)


def regression_metrics(frame: pd.DataFrame) -> pd.Series:
    y = frame["truth"].to_numpy()
    p = frame["prediction"].to_numpy()
    pearson = np.corrcoef(y, p)[0, 1] if (len(y) > 1 and np.std(y) > 0 and np.std(p) > 0) else np.nan
    residual = y - p
    sst = np.square(y - y.mean()).sum()
    r2 = 1 - np.square(residual).sum() / sst if sst > 0 else np.nan
    return pd.Series(
        {
            "n": len(frame),
            "MAE": np.abs(residual).mean(),
            "RMSE": np.sqrt(np.square(residual).mean()),
            "R2": r2,
            "Pearson": pearson,
        }
    )


def grouped_metrics(frame: pd.DataFrame, group_cols: list) -> pd.DataFrame:
    rows = []
    for keys, group in frame.groupby(group_cols):
        keys = keys if isinstance(keys, tuple) else (keys,)
        rows.append(dict(zip(group_cols, keys)) | regression_metrics(group).to_dict())
    return pd.DataFrame(rows).set_index(group_cols)


def summarize_trial(predictions: pd.DataFrame, split_name: str) -> dict:
    overall = regression_metrics(predictions).add_prefix(f"{split_name}_overall_")
    by_outcome = grouped_metrics(predictions, ["outcome"])
    row = overall.to_dict()
    for outcome in ["SIS", "OHS"]:
        if outcome in by_outcome.index:
            for metric, value in by_outcome.loc[outcome].items():
                row[f"{split_name}_{outcome}_{metric}"] = value
    return row


def run_trial(
    examples: list,
    demo_lookup: pd.DataFrame,
    split: dict,
    top_x_sensor: int,
    top_y_demo: int,
) -> dict:
    feature_cfg = build_feature_selection(top_x_sensor, top_y_demo)
    artifact = train_model(examples, demo_lookup, split, feature_cfg)
    val_predictions = collect_predictions(examples, artifact, feature_cfg, "val")
    test_predictions = collect_predictions(examples, artifact, feature_cfg, "test")

    row = {
        "TOP_X_SENSOR": top_x_sensor,
        "TOP_Y_DEMO": top_y_demo,
        "n_sensor_union": len(feature_cfg["union_sensor"]),
        "n_demo_union": len(feature_cfg["union_demo"]),
        "n_features": feature_cfg["n_features"],
        "epochs": artifact["epochs"],
        "best_val_loss": artifact["best_val"],
        "sis_decay": artifact["sis_decay"],
        "ohs_decay": artifact["ohs_decay"],
    }
    row.update(summarize_trial(val_predictions, "val"))
    row.update(summarize_trial(test_predictions, "test"))
    return row


def plot_heatmap(results: pd.DataFrame, metric: str, output_path: Path) -> None:
    table = results.pivot(index="TOP_Y_DEMO", columns="TOP_X_SENSOR", values=metric).sort_index()
    fig, ax = plt.subplots(figsize=(14, 5))
    im = ax.imshow(table.to_numpy(), aspect="auto", origin="lower", cmap="viridis")
    ax.set_xticks(np.arange(len(table.columns)))
    ax.set_xticklabels(table.columns, rotation=90)
    ax.set_yticks(np.arange(len(table.index)))
    ax.set_yticklabels(table.index)
    ax.set_xlabel("TOP_X_SENSOR")
    ax.set_ylabel("TOP_Y_DEMO")
    ax.set_title(metric)
    fig.colorbar(im, ax=ax, label=metric)
    fig.tight_layout()
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


def plot_lines(results: pd.DataFrame, metric: str, output_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(11, 6))
    for top_y_demo, group in results.sort_values("TOP_X_SENSOR").groupby("TOP_Y_DEMO"):
        ax.plot(group["TOP_X_SENSOR"], group[metric], marker="o", linewidth=1.5, label=top_y_demo)
    ax.set_xlabel("TOP_X_SENSOR")
    ax.set_ylabel(metric)
    ax.set_title(metric)
    ax.legend(title="TOP_Y_DEMO", ncol=4)
    fig.tight_layout()
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


def save_plots(results: pd.DataFrame, output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    metrics = [
        "test_overall_MAE",
        "test_SIS_MAE",
        "test_OHS_MAE",
        "test_overall_RMSE",
        "val_overall_MAE",
    ]
    for metric in metrics:
        if metric not in results.columns:
            continue
        plot_heatmap(results, metric, output_dir / f"{metric}_heatmap.png")
        plot_lines(results, metric, output_dir / f"{metric}_lines.png")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, default=Path("grid_search_results"))
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--sensor-min", type=int, default=5)
    parser.add_argument("--sensor-max", type=int, default=46)
    parser.add_argument("--demo-min", type=int, default=0)
    parser.add_argument("--demo-max", type=int, default=7)
    return parser.parse_args()


def main() -> None:
    global TOP_X_SENSOR, TOP_Y_DEMO

    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    results_path = args.output_dir / "grid_search_results.csv"

    seed_everything(SEED)
    plt.style.use("ggplot")
    print(f"Device: {DEVICE}")

    raw, demo_lookup, sensor_cols_all = load_data()
    examples = build_forecast_examples(raw, demo_lookup, sensor_cols_all, FORECAST_GAP)
    example_index = pd.DataFrame(
        [
            {
                "patient": e["patient"],
                "clinical_time": e["clinical_time"],
                "history_days": len(e["timestamps"]),
            }
            for e in examples
        ]
    )
    split = make_split(example_index)

    completed = set()
    rows = []
    if args.resume and results_path.exists():
        previous = pd.read_csv(results_path)
        rows = previous.to_dict("records")
        completed = set(zip(previous["TOP_X_SENSOR"], previous["TOP_Y_DEMO"]))

    total = (args.sensor_max - args.sensor_min + 1) * (args.demo_max - args.demo_min + 1)
    for top_x_sensor in range(args.sensor_min, args.sensor_max + 1):
        for top_y_demo in range(args.demo_min, args.demo_max + 1):
            if (top_x_sensor, top_y_demo) in completed:
                print(f"Skipping TOP_X_SENSOR={top_x_sensor}, TOP_Y_DEMO={top_y_demo}")
                continue

            TOP_X_SENSOR = top_x_sensor
            TOP_Y_DEMO = top_y_demo
            print(
                f"[{len(rows) + 1}/{total}] "
                f"TOP_X_SENSOR={TOP_X_SENSOR}, TOP_Y_DEMO={TOP_Y_DEMO}"
            )
            row = run_trial(examples, demo_lookup, split, TOP_X_SENSOR, TOP_Y_DEMO)
            rows.append(row)
            pd.DataFrame(rows).sort_values(["TOP_X_SENSOR", "TOP_Y_DEMO"]).to_csv(
                results_path, index=False
            )

    results = pd.DataFrame(rows).sort_values(["TOP_X_SENSOR", "TOP_Y_DEMO"])
    results.to_csv(results_path, index=False)
    save_plots(results, args.output_dir)

    best = results.sort_values("test_overall_MAE").head(10)
    best.to_csv(args.output_dir / "top10_by_test_overall_MAE.csv", index=False)
    print("\nTop 10 by test_overall_MAE")
    print(
        best[
            [
                "TOP_X_SENSOR",
                "TOP_Y_DEMO",
                "n_features",
                "epochs",
                "test_overall_MAE",
                "test_SIS_MAE",
                "test_OHS_MAE",
            ]
        ].to_string(index=False)
    )
    print(f"\nSaved results to {results_path}")
    print(f"Saved plots to {args.output_dir}")


if __name__ == "__main__":
    main()
