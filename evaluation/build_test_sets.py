"""
build_test_sets.py
===================
Test Seti A (regresyon kontrolü) ve Test Seti B (Türkçe el yazısı/özel karakter
iyileşme kontrolü) İLE Validation setinin ASIL İNŞASI, veri sızıntısını önlemek için
TEK bir yerde — data/build_chat_dataset.py içinde — yapılır (train/val/test_a/test_b
ayrımı orada, aynı örnekleme akışının bir parçası olarak gerçekleşir; ayrı bir yerde
tekrar bölme mantığı yazmak, yanlışlıkla eğitim setiyle örtüşen örnekler üretme riski
taşırdı).

Bu dosya, o ÖNCEDEN İNŞA EDİLMİŞ test/val setlerini disktten okuyan İNCE (thin) bir
erişim katmanıdır; evaluation/evaluate.py ve training/callbacks.py bu fonksiyonları
kullanır. Böylece evaluation/ paketi, data/ paketinin iç (dahili) örnekleme mantığına
bağımlı olmadan, yalnızca nihai (processed) çıktıyı okur.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from configs import config  # noqa: E402

import datasets  # type: ignore


def _load_split(split_name: str) -> datasets.Dataset:
    split_dir = config.PROCESSED_DATA_DIR / split_name
    if not split_dir.exists():
        raise FileNotFoundError(
            f"{split_dir} bulunamadı. Önce data/build_chat_dataset.py çalıştırılmalı "
            "(o script train/val/test_a/test_b ayrımını tek seferde, sızıntısız yapar)."
        )
    return datasets.Dataset.load_from_disk(str(split_dir))


def load_val() -> datasets.Dataset:
    return _load_split("val")


def load_test_a() -> datasets.Dataset:
    """Test Seti A: genel talimat-takip (replay_general) + karışık-kaynak OCR.
    Bu skor eğitim boyunca DÜŞMEMELİDİR (regresyon/katastrofik unutma kontrolü)."""
    return _load_split("test_a")


def load_test_b() -> datasets.Dataset:
    """Test Seti B: Türkçe el yazısı / özel karakter ağırlıklı.
    Bu skor eğitim boyunca YÜKSELMELİDİR (asıl hedeflenen iyileşme)."""
    return _load_split("test_b")
