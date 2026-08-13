"""
ablation_report.py
===================
Farklı RUN_NAME'lerle (bkz. configs/config.py: RUN_NAME) çalıştırılmış training/train_sft.py
koşumlarını KARŞILAŞTIRIR. Her koşum kendi çıktısını
DRIVE_ROOT/eval_outputs/{MODE_TAG}/runs/{RUN_NAME}/ altına yazar:
  - run_config.json     : o koşumun ablation-ilgili hiperparametreleri (bkz. train_sft.py)
  - eval_history.json    : her epoch sonrası Test A/B skorları (bkz. training/callbacks.py)
  - pareto_summary.json  : kabul edilebilir forgetting sınırı içindeki en iyi (Test B CER'i
                            en düşük) epoch (bkz. training/callbacks.py Pareto-kısıtlı seçim)
  - lora_drift/drift_log.jsonl : her N adımda katman başına LoRA update normu (bkz.
                            training/lora_drift.py)

Bu script BU DOSYALARI OKUR (yeni bir training/eval koşumu BAŞLATMAZ) ve iki çıktı üretir:
  1) Bir karşılaştırma tablosu (stdout + CSV) -- her koşumun LR/layer-scope/target-scope/
     rank-alpha-dropout ayarlarını, son epoch ve Pareto-seçili epoch metriklerini yan yana koyar.
  2) (--drift-plot RUN_NAME verilirse) O koşum için katman başına LoRA drift oranının
     (||ΔW||/||W||) training step'e göre grafiği; Test A değerlendirme noktaları (ve o
     andaki forgetting oranı) dikey çizgilerle işaretlenir -- "regresyon başladığında hangi
     katmanların update normunda ani artış oldu?" sorusunu görsel olarak yanıtlamak için.

Kullanım:
    python analysis/ablation_report.py                          # tüm koşumları karşılaştır
    python analysis/ablation_report.py --mode-tag full           # yalnızca tam-mod koşumları
    python analysis/ablation_report.py --drift-plot lr_5e-5      # bir koşumun drift grafiği
"""

import argparse
import csv
import json
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from configs import config  # noqa: E402


def _eval_output_dir(mode_tag: str) -> Path:
    return config.DRIVE_ROOT / "eval_outputs" / mode_tag


def _run_dir(mode_tag: str, run_name: str) -> Path:
    return _eval_output_dir(mode_tag) / "runs" / run_name


def discover_runs(mode_tag: str) -> list[str]:
    runs_root = _eval_output_dir(mode_tag) / "runs"
    if not runs_root.exists():
        return []
    return sorted(p.name for p in runs_root.iterdir() if p.is_dir())


def _read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}


def load_run_summary(mode_tag: str, run_name: str) -> dict | None:
    rdir = _run_dir(mode_tag, run_name)
    eval_history = _read_json(rdir / "eval_history.json")
    if not eval_history:
        return None  # bu koşum henüz hiç epoch tamamlamamış (veya hiç çalıştırılmamış)

    run_config = _read_json(rdir / "run_config.json")
    pareto = _read_json(rdir / "pareto_summary.json")
    history = eval_history.get("history", [])
    last = history[-1] if history else {}
    best = pareto.get("best_epoch")

    return {
        "run_name": run_name,
        "learning_rate": run_config.get("learning_rate"),
        "lora_layer_scope": run_config.get("lora_layer_scope"),
        "lora_target_scope": run_config.get("lora_target_scope"),
        "lora_r": run_config.get("lora_r"),
        "lora_alpha": run_config.get("lora_alpha"),
        "lora_dropout": run_config.get("lora_dropout"),
        "enable_weighted_loss": run_config.get("enable_weighted_loss"),
        "loss_weight_replay": run_config.get("loss_weight_replay"),
        "enable_online_self_distillation": run_config.get("enable_online_self_distillation"),
        "n_epochs_completed": len(history),
        "last_test_a_relative_drop": last.get("relative_drop"),
        "last_test_b_cer": last.get("test_b_cer"),
        "last_test_b_wer": last.get("test_b_wer"),
        "last_test_b_exact_match": last.get("test_b_exact_match"),
        "last_test_b_turkish_char_accuracy": last.get("test_b_turkish_char_accuracy"),
        "pareto_satisfied_constraint": best is not None,
        "pareto_selected_epoch": best.get("epoch") if best else None,
        "pareto_test_b_cer": best.get("test_b_cer") if best else None,
        "pareto_relative_drop": best.get("relative_drop") if best else None,
        "best_adapter_path": pareto.get("best_adapter_path"),
    }


_TABLE_COLUMNS = [
    "run_name",
    "learning_rate",
    "lora_layer_scope",
    "lora_target_scope",
    "lora_r",
    "n_epochs_completed",
    "pareto_satisfied_constraint",
    "pareto_selected_epoch",
    "pareto_test_b_cer",
    "pareto_relative_drop",
    "last_test_b_cer",
    "last_test_a_relative_drop",
    "last_test_b_turkish_char_accuracy",
]


def build_comparison_table(mode_tag: str) -> list[dict]:
    rows = []
    for run_name in discover_runs(mode_tag):
        summary = load_run_summary(mode_tag, run_name)
        if summary is not None:
            rows.append(summary)

    def _sort_key(row: dict):
        # Önce KISIT SAĞLAYAN koşumlar (Pareto-uygun), aralarında Test B CER'i EN DÜŞÜK olan üstte.
        satisfies = row["pareto_satisfied_constraint"]
        cer = row["pareto_test_b_cer"] if satisfies else row["last_test_b_cer"]
        cer = cer if cer is not None else float("inf")
        return (0 if satisfies else 1, cer)

    return sorted(rows, key=_sort_key)


def print_comparison_table(rows: list[dict]) -> None:
    if not rows:
        print("[ablation_report] Karşılaştırılacak koşum bulunamadı (eval_history.json yok).")
        return
    widths = {col: max(len(col), *(len(str(row.get(col, ""))) for row in rows)) for col in _TABLE_COLUMNS}
    header = " | ".join(col.ljust(widths[col]) for col in _TABLE_COLUMNS)
    print(header)
    print("-" * len(header))
    for row in rows:
        print(" | ".join(str(row.get(col, "")).ljust(widths[col]) for col in _TABLE_COLUMNS))


def write_comparison_csv(rows: list[dict], mode_tag: str) -> Path:
    out_path = config.ANALYSIS_OUTPUT_DIR / f"ablation_comparison_{mode_tag}.csv"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    all_columns = sorted({key for row in rows for key in row.keys()})
    with open(out_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=all_columns)
        writer.writeheader()
        writer.writerows(rows)
    print(f"[ablation_report] Karşılaştırma tablosu kaydedildi: {out_path}")
    return out_path


def plot_drift(mode_tag: str, run_name: str, output_path: Path | None = None) -> Path:
    """Katman başına ||ΔW||/||W|| oranını training step'e göre çizer; her epoch
    değerlendirme noktasını (ve o andaki Test A forgetting oranını) dikey çizgiyle işaretler."""
    import matplotlib  # type: ignore

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt  # type: ignore

    rdir = _run_dir(mode_tag, run_name)
    drift_path = rdir / "lora_drift" / "drift_log.jsonl"
    if not drift_path.exists():
        raise FileNotFoundError(
            f"{drift_path} bulunamadı. Bu RUN_NAME ile training/train_sft.py "
            "LORA_DRIFT_LOG_EVERY_N_STEPS>0 iken çalıştırılmış olmalı."
        )

    series: dict[int, list[tuple[int, float]]] = defaultdict(list)
    with open(drift_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            if rec["layer_index"] < 0:
                continue  # embed/merger gibi katman-dışı modülleri bu grafikte atla
            series[rec["layer_index"]].append((rec["global_step"], rec["delta_over_base_ratio"]))

    if not series:
        raise RuntimeError(f"{drift_path} boş görünüyor (decoder katmanına ait kayıt yok).")

    fig, ax = plt.subplots(figsize=(11, 6))
    layer_indices = sorted(series.keys())
    cmap = plt.get_cmap("viridis")
    for i, layer_idx in enumerate(layer_indices):
        # aynı step için birden fazla modül türü (q,k,v,o,gate,up,down) var; step'e göre ortalama al
        per_step: dict[int, list[float]] = defaultdict(list)
        for step, ratio in series[layer_idx]:
            per_step[step].append(ratio)
        xs = sorted(per_step.keys())
        ys = [sum(per_step[s]) / len(per_step[s]) for s in xs]
        color = cmap(i / max(len(layer_indices) - 1, 1))
        label = f"layer {layer_idx}" if layer_idx % 4 == 0 or layer_idx == layer_indices[-1] else None
        ax.plot(xs, ys, color=color, alpha=0.85, linewidth=1, label=label)

    eval_history = _read_json(rdir / "eval_history.json")
    ymax = ax.get_ylim()[1]
    for rec in eval_history.get("history", []):
        step = rec.get("global_step")
        if step is None:
            continue
        ax.axvline(step, color="red", linestyle="--", alpha=0.35)
        ax.text(step, ymax, f"e{rec['epoch']} drop={rec['relative_drop']:.0%}", fontsize=7, color="red", va="top")

    ax.set_xlabel("training step")
    ax.set_ylabel("katman başına ||ΔW|| / ||W||")
    ax.set_title(f"LoRA representation drift — run={run_name} ({mode_tag})")
    ax.legend(loc="upper left", fontsize=7, ncol=2)
    fig.tight_layout()

    output_path = output_path or (config.ANALYSIS_OUTPUT_DIR / f"lora_drift_{mode_tag}_{run_name}.png")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    print(f"[ablation_report] Drift grafiği kaydedildi: {output_path}")
    return output_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Ablation koşumlarını karşılaştırır / drift grafiği çizer.")
    parser.add_argument("--mode-tag", default=config.MODE_TAG, help="'pilot' veya 'full' (varsayılan: mevcut MODE_TAG).")
    parser.add_argument(
        "--drift-plot",
        default=None,
        metavar="RUN_NAME",
        help="Verilirse yalnızca bu koşumun LoRA drift grafiğini çizer (karşılaştırma tablosunu atlamaz).",
    )
    args = parser.parse_args()

    config.ensure_directories()
    rows = build_comparison_table(args.mode_tag)
    print_comparison_table(rows)
    if rows:
        write_comparison_csv(rows, args.mode_tag)

    if args.drift_plot:
        plot_drift(args.mode_tag, args.drift_plot)


if __name__ == "__main__":
    main()
