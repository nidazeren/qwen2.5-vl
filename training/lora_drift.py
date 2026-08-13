"""
lora_drift.py
=============
Her N training step'inde, modeldeki HER LoRA katmanı için "temsil kayması"
(representation drift) büyüklüğünü ölçüp LORA_DRIFT_DIR/drift_log.jsonl'a yazan
bir TrainerCallback. Ölçüt, o katmanın taban ağırlığına (frozen, W) göre şu anki
LoRA delta'sının (ΔW = B@A*scaling) Frobenius normudur; ayrıca ΔW/W oranı da
hesaplanır (mutlak ölçek katmandan katmana çok değiştiği için oran, katmanlar
arası KARŞILAŞTIRMA için daha anlamlıdır).

Amaç: hangi katmanların (özellikle Test A regresyonunun başladığı adımlarda) daha
agresif güncellendiğini training SONRASI tespit edebilmek (bkz. analysis/ablation_report.py)
-- proje notlarındaki "catastrophic forgetting hangi katmanlardan kaynaklanıyor?" sorusu.

Yalnızca standart nn.Linear tabanlı LoRA katmanlarını (attention/MLP projeksiyonları,
merger) VE (varsa) nn.Embedding tabanlı LoRA'yı (embed_tokens) destekler; beklenmedik
bir peft iç yapısıyla karşılaşılırsa o katman SESSİZCE atlanır (bir kez uyarılır),
training asla bu yüzden çökmez -- bu, bir GÖZLEM aracıdır, eğitim akışının kritik
bir parçası değildir.
"""

import json
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from configs import config  # noqa: E402

import torch  # type: ignore
from transformers import TrainerCallback  # type: ignore

_LAYER_INDEX_PATTERN = re.compile(r"\.layers\.(\d+)\.")
_MODULE_TYPE_PATTERN = re.compile(
    r"(q_proj|k_proj|v_proj|o_proj|gate_proj|up_proj|down_proj|embed_tokens|merger\.\w+)$"
)


def _layer_index_of(name: str) -> int:
    """Decoder katman indeksi (ör. '...layers.23...' -> 23); decoder dışı modüller
    (merger/embed_tokens) için -1 (bu modüller belirli bir katmana ait değildir)."""
    match = _LAYER_INDEX_PATTERN.search(name)
    return int(match.group(1)) if match else -1


def _module_type_of(name: str) -> str:
    match = _MODULE_TYPE_PATTERN.search(name)
    return match.group(1) if match else name.rsplit(".", 1)[-1]


@torch.no_grad()
def _linear_lora_norms(module, lora_A, lora_B) -> tuple[float, float]:
    adapter_name = next(iter(lora_A.keys()))
    A = lora_A[adapter_name].weight.detach().float()  # (r, in_features)
    B = lora_B[adapter_name].weight.detach().float()  # (out_features, r)
    scaling = float(module.scaling[adapter_name]) if hasattr(module, "scaling") else 1.0
    delta_w = (B @ A) * scaling
    base_weight = module.base_layer.weight.detach().float()
    return torch.linalg.norm(delta_w).item(), torch.linalg.norm(base_weight).item()


@torch.no_grad()
def _embedding_lora_norms(module, embedding_A, embedding_B) -> tuple[float, float]:
    adapter_name = next(iter(embedding_A.keys()))
    A = embedding_A[adapter_name].detach().float()  # (r, num_embeddings)
    B = embedding_B[adapter_name].detach().float()  # (embedding_dim, r)
    scaling = float(module.scaling[adapter_name]) if hasattr(module, "scaling") else 1.0
    delta_w = (B @ A).T * scaling  # -> (num_embeddings, embedding_dim), base_layer.weight ile aynı şekil
    base_weight = module.base_layer.weight.detach().float()
    return torch.linalg.norm(delta_w).item(), torch.linalg.norm(base_weight).item()


def compute_layer_update_norms(model) -> list[dict]:
    """Modeldeki her LoRA katmanı için delta-W normu ve ΔW/W oranını hesaplar."""
    records: list[dict] = []
    for name, module in model.named_modules():
        lora_A = getattr(module, "lora_A", None)
        lora_B = getattr(module, "lora_B", None)
        embedding_A = getattr(module, "lora_embedding_A", None)
        embedding_B = getattr(module, "lora_embedding_B", None)

        try:
            if lora_A is not None and lora_B is not None and len(lora_A) > 0:
                delta_norm, base_norm = _linear_lora_norms(module, lora_A, lora_B)
            elif embedding_A is not None and embedding_B is not None and len(embedding_A) > 0:
                delta_norm, base_norm = _embedding_lora_norms(module, embedding_A, embedding_B)
            else:
                continue
        except Exception as exc:  # noqa: BLE001 -- bu bir gözlem aracı, training'i asla düşürmemeli
            print(f"[lora_drift] !! '{name}' için drift hesaplanamadı, atlanıyor ({exc}).")
            continue

        records.append(
            {
                "layer_name": name,
                "layer_index": _layer_index_of(name),
                "module_type": _module_type_of(name),
                "delta_norm": delta_norm,
                "base_norm": base_norm,
                "delta_over_base_ratio": delta_norm / base_norm if base_norm > 1e-12 else 0.0,
            }
        )
    return records


class LoRADriftTrackingCallback(TrainerCallback):
    """`config.LORA_DRIFT_LOG_EVERY_N_STEPS` adımda bir (0 ise devre dışı) her LoRA
    katmanının drift ölçütlerini LORA_DRIFT_DIR/drift_log.jsonl'a EKLER (append).
    training/train_sft.py tarafından TestABRegressionCallback ile BİRLİKTE eklenir;
    aynı `global_step`/`epoch` alanları üzerinden (bkz. callbacks.py:eval_history.json)
    drift ile Test A regresyonu arasındaki zamansal ilişki sonradan eşlenebilir."""

    def __init__(self):
        self.log_every = config.LORA_DRIFT_LOG_EVERY_N_STEPS
        self.out_path = config.LORA_DRIFT_DIR / "drift_log.jsonl"

    def on_step_end(self, args, state, control, **kwargs):
        if self.log_every <= 0 or state.global_step % self.log_every != 0:
            return control
        model = kwargs["model"]
        records = compute_layer_update_norms(model)
        if not records:
            return control

        self.out_path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.out_path, "a", encoding="utf-8") as f:
            for rec in records:
                rec["global_step"] = state.global_step
                rec["epoch"] = state.epoch
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")

        return control
