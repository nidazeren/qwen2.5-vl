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
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import jiwer  # type: ignore
from rouge_score import rouge_scorer  # type: ignore

_rouge_scorer = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=False)

TURKISH_SPECIAL_CHARS = set("çğıöşüÇĞİÖŞÜ")


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


def compute_turkish_char_accuracy(references: list[str], hypotheses: list[str]) -> float:
    """Türkçe özel karakterlerin (ç,ğ,ı,ö,ş,ü + büyük harfleri) hipotezde ne ölçüde
    KORUNDUĞUNU ölçer (0.0-1.0, yüksek = iyi). CER/WER tüm karakterleri eşit ağırlıklandırır
    ve İngilizce/ASCII karakterlerin çoğunlukta olduğu bir metinde birkaç yanlış "ş"/"ğ"
    toplam skoru neredeyse etkilemez -- bu proje özelinde tam olarak izlenmek istenen
    hata türü budur (bkz. proje notları: "Türkçe karakter doğruluğu"). Bu yüzden her
    referans için Türkçe özel karakterlerin multiset'i (Counter) çıkarılır ve hipotezdeki
    aynı karakter multiset'iyle KESİŞİM oranı (referanstaki her karakterin hipotezde de
    bulunma oranı) hesaplanır -- konumdan bağımsız, kaba ama yorumlanması kolay bir ölçüt.
    Referanslarda hiç Türkçe özel karakter yoksa (ör. saf İngilizce el yazısı örnekleri)
    1.0 (nötr/mükemmel) döner -- bu kaynaklarda metrik tanımsızdır, ceza uygulanmaz."""
    total = 0
    correct = 0
    for ref, hyp in zip(references, hypotheses):
        ref_chars = Counter(ch for ch in ref if ch in TURKISH_SPECIAL_CHARS)
        hyp_chars = Counter(ch for ch in hyp if ch in TURKISH_SPECIAL_CHARS)
        for ch, ref_count in ref_chars.items():
            total += ref_count
            correct += min(ref_count, hyp_chars.get(ch, 0))
    return correct / total if total else 1.0


def compute_ocr_metrics(references: list[str], hypotheses: list[str]) -> dict:
    """OCR odaklı kaynaklar (basılı/sahne/el yazısı) için standart metrik seti."""
    return {
        "cer": compute_cer(references, hypotheses),
        "wer": compute_wer(references, hypotheses),
        "exact_match": compute_exact_match(references, hypotheses),
        "turkish_char_accuracy": compute_turkish_char_accuracy(references, hypotheses),
    }


def compute_general_metrics(references: list[str], hypotheses: list[str]) -> dict:
    """Genel-görev (OCR dışı) kaynaklar için metrik seti."""
    return {
        "rouge_l": compute_rouge_l(references, hypotheses),
        "cer": compute_cer(references, hypotheses),  # bilgi amaçlı, birincil ölçüt değil
    }
