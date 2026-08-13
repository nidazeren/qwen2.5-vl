"""
distillation_trainer.py
========================
config.ENABLE_WEIGHTED_LOSS ve/veya config.ENABLE_ONLINE_SELF_DISTILLATION=True
iken train_sft.py tarafından trl.SFTTrainer YERİNE kullanılan bir alt sınıf.
Her ikisi de KAPALIYKEN training/train_sft.py bu dosyayı hiç import etmez ve
standart SFTTrainer (mevcut/varsayılan davranış, loss_type="nll") kullanılır --
bu dosya SADECE faz 5 (replay loss weighting / online self-distillation)
ablation'ları için devreye girer.

İki bağımsız mekanizma uygular:

  1) AĞIRLIKLI KAYIP (config.ENABLE_WEIGHTED_LOSS): trl'nin varsayılan tek-tip
     ortalama NLL'i yerine, her örneğin kaybı `source` sütununa göre (bkz.
     training/collate.py) bir kovaya (OCR-TR / el yazısı / replay) atanır ve
     config.LOSS_WEIGHT_* ile ağırlıklandırılmış bir ortalama alınır. Batch'ler
     zaten karışım oranına göre örneklendiğinden, bu epoch boyunca
     `toplam_loss ~= w_ocr*OCR_loss + w_hw*handwriting_loss + w_replay*replay_loss`
     beklentisine yakınsar -- veri MİKTARINI değiştirmeden stability kazandırmak
     için (bkz. configs/config.py bölüm 9b).

  2) ONLINE SELF-DISTILLATION (config.ENABLE_ONLINE_SELF_DISTILLATION): frozen
     bir teacher model (ince ayardan ÖNCEKİ taban model, train_sft.py tarafından
     yüklenip buraya geçirilir) HER batch'te AYNI girdiyle çalıştırılır (no_grad);
     student'ın çıktı dağılımı teacher'a KL-diverjans ile yaklaştırılır ve bu terim
     config.LOSS_WEIGHT_DISTILL ile toplam kayba eklenir (Hinton ve ark. distillation
     formülasyonu: KL(teacher||student), sıcaklık T ile yumuşatılmış, T^2 ile ölçeklenmiş).

Her iki mekanizma da örnek başına (token-mean) kayıp gerektirdiği için trl'nin
hazır "nll"/"chunked_nll" kayıp yollarını KULLANMAZ; logits'i elle alıp
CrossEntropyLoss(reduction="none") ile per-token, sonra per-example kayıp hesaplar.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from configs import config  # noqa: E402

import torch  # type: ignore
import torch.nn.functional as F  # type: ignore
from trl import SFTTrainer  # type: ignore

IGNORE_INDEX = -100

# training/collate.py'nin taşıdığı `source` değerlerini üç kovaya (bkz. config.py
# bölüm 9b: LOSS_WEIGHT_OCR_TR/HANDWRITING/REPLAY) eşler. data/build_chat_dataset.py:
# OCR_SOURCE_NAMES/REPLAY_OCR_NAME/REPLAY_GENERAL_NAME ile TUTARLI tutulmalıdır.
_BUCKET_BY_SOURCE = {
    "printed_synthetic": "ocr_tr",
    "scene_text": "ocr_tr",
    "handwriting_synthetic": "handwriting",
    "smhd_english": "handwriting",
    "replay_ocr": "replay",
    "replay_general": "replay",
}


def _bucket_weight(source: str) -> float:
    bucket = _BUCKET_BY_SOURCE.get(source)
    if bucket == "ocr_tr":
        return config.LOSS_WEIGHT_OCR_TR
    if bucket == "handwriting":
        return config.LOSS_WEIGHT_HANDWRITING
    if bucket == "replay":
        return config.LOSS_WEIGHT_REPLAY
    return 1.0  # bilinmeyen/yeni bir kaynak -> nötr ağırlık (ablation'ı sessizce bozmasın diye)


def _per_example_nll(logits: torch.Tensor, labels: torch.Tensor):
    """Her örnek için PADDING/PROMPT-maskelenmiş token'lar HARİÇ (bkz. training/collate.py:
    labels maskeleme) ortalama negatif log-likelihood döner; ayrıca sonraki (distillation)
    hesaplamada yeniden kullanılmak üzere shift edilmiş logits ve geçerlilik maskesini de döner."""
    shift_logits = logits[..., :-1, :].contiguous()
    shift_labels = labels[..., 1:].contiguous()
    per_token_loss = F.cross_entropy(
        shift_logits.view(-1, shift_logits.size(-1)),
        shift_labels.view(-1),
        ignore_index=IGNORE_INDEX,
        reduction="none",
    ).view(shift_labels.size())
    valid_mask = (shift_labels != IGNORE_INDEX).float()
    token_counts = valid_mask.sum(dim=1).clamp(min=1.0)
    per_example = (per_token_loss * valid_mask).sum(dim=1) / token_counts
    return per_example, shift_logits, valid_mask, token_counts


class WeightedDistillationTrainer(SFTTrainer):
    """`teacher_model=None` iken (yalnızca ENABLE_WEIGHTED_LOSS=True) online
    distillation terimi hesaplanmaz; yalnızca ağırlıklı NLL uygulanır."""

    def __init__(self, *args, teacher_model=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.teacher_model = teacher_model
        if self.teacher_model is not None:
            self.teacher_model.eval()
            self.teacher_model.requires_grad_(False)

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        sources = inputs.pop("source", None)
        labels = inputs.pop("labels")

        outputs = model(**inputs)
        per_example_nll, shift_logits, valid_mask, token_counts = _per_example_nll(outputs.logits, labels)

        if config.ENABLE_WEIGHTED_LOSS and sources is not None:
            weights = torch.tensor(
                [_bucket_weight(s) for s in sources],
                device=per_example_nll.device,
                dtype=per_example_nll.dtype,
            )
            total_loss = (per_example_nll * weights).sum() / weights.sum().clamp(min=1e-8)
        else:
            total_loss = per_example_nll.mean()

        if config.ENABLE_ONLINE_SELF_DISTILLATION and self.teacher_model is not None:
            with torch.no_grad():
                teacher_outputs = self.teacher_model(**inputs)
            teacher_shift_logits = teacher_outputs.logits[..., :-1, :].contiguous()

            temperature = config.DISTILLATION_TEMPERATURE
            student_log_probs = F.log_softmax(shift_logits / temperature, dim=-1)
            teacher_probs = F.softmax(teacher_shift_logits.to(shift_logits.dtype) / temperature, dim=-1)
            kl_per_token = F.kl_div(student_log_probs, teacher_probs, reduction="none").sum(dim=-1)
            kl_per_example = (kl_per_token * valid_mask).sum(dim=1) / token_counts
            distill_loss = kl_per_example.mean() * (temperature**2)

            total_loss = total_loss + config.LOSS_WEIGHT_DISTILL * distill_loss

        return (total_loss, outputs) if return_outputs else total_loss
