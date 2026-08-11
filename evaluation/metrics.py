"""
metrics.py
==========
Değerlendirme metrikleri:
  - CER (Character Error Rate) ve WER (Word Error Rate): OCR doğruluğunun birincil
    ölçütleri (`jiwer` kütüphanesi ile, corpus-level yani tüm örnekler üzerinden
    toplu hesaplanır — tekil örnek gürültüsüne karşı daha stabildir).
  - Tam eşleşme oranı (exact match): normalize edilmiş (baş/son boşluk temizlenmiş,
    küçük harfe çevrilmiş) string eşitliği.
  - ROUGE-L F1: yalnızca GENEL görevler (replay_general / Test A'nın genel-görev yarısı)
    için kullanılır; bu örneklerde "doğru cevap" karakter-birebir bir transkripsiyon
    DEĞİL, baseline modelin ürettiği serbest metin açıklamalardır, bu yüzden CER/WER
    yerine örtüşme tabanlı bir metrik daha anlamlıdır.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import jiwer  # type: ignore
from rouge_score import rouge_scorer  # type: ignore

_rouge_scorer = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=False)


def _normalize(text: str) -> str:
    return " ".join(text.strip().lower().split())


def compute_cer(references: list[str], hypotheses: list[str]) -> float:
    """Corpus-level Character Error Rate (0.0 = mükemmel, düşük = iyi)."""
    if not references:
        return 0.0
    return float(jiwer.cer(references, hypotheses))


def compute_wer(references: list[str], hypotheses: list[str]) -> float:
    """Corpus-level Word Error Rate (0.0 = mükemmel, düşük = iyi)."""
    if not references:
        return 0.0
    return float(jiwer.wer(references, hypotheses))


def compute_exact_match(references: list[str], hypotheses: list[str]) -> float:
    """Normalize edilmiş tam eşleşme oranı (0.0-1.0, yüksek = iyi)."""
    if not references:
        return 0.0
    matches = sum(
        1 for ref, hyp in zip(references, hypotheses) if _normalize(ref) == _normalize(hyp)
    )
    return matches / len(references)


def compute_rouge_l(references: list[str], hypotheses: list[str]) -> float:
    """Ortalama ROUGE-L F1 skoru (0.0-1.0, yüksek = iyi) — genel görevler için."""
    if not references:
        return 0.0
    scores = [
        _rouge_scorer.score(ref, hyp)["rougeL"].fmeasure for ref, hyp in zip(references, hypotheses)
    ]
    return sum(scores) / len(scores)


def compute_ocr_metrics(references: list[str], hypotheses: list[str]) -> dict:
    """OCR odaklı kaynaklar (basılı/sahne/el yazısı) için standart metrik seti."""
    return {
        "cer": compute_cer(references, hypotheses),
        "wer": compute_wer(references, hypotheses),
        "exact_match": compute_exact_match(references, hypotheses),
    }


def compute_general_metrics(references: list[str], hypotheses: list[str]) -> dict:
    """Genel-görev (OCR dışı) kaynaklar için metrik seti."""
    return {
        "rouge_l": compute_rouge_l(references, hypotheses),
        "cer": compute_cer(references, hypotheses),  # bilgi amaçlı, birincil ölçüt değil
    }
