"""
tokenizer_analysis.py
======================
Qwen2.5-VL'nin BBPE (byte-level BPE) tokenizer'ının Türkçe'ye özgü karakterleri
(ç, ğ, ı, ö, ş, ü — ve büyük harfleri) ne kadar "parçaladığını" ölçer.

Neden önemli: BBPE tokenizer'lar eğitim verisinde az görülen karakter/kelimeleri
birden fazla alt-token'a (hatta byte-fallback token'lara) bölebilir. Bu durum ne kadar
şiddetliyse, modelin bu karakterleri DOĞRU ÜRETMESİ o kadar zorlaşır (her ekstra token,
modelin doğru sırayla art arda üretmesi gereken bir karar noktası demektir). Bu script:

  1) Özel karakter İÇEREN Türkçe kelimeler ile İÇERMEYEN (ya da İngilizce) kelimeleri
     karşılaştırarak "karakter başına token" oranını hesaplar,
  2) Her özel karakterin tek başına kaç token'a bölündüğünü raporlar,
  3) Bir "parçalanma oranı" (fragmentation ratio) üretir ve bu oran belli bir eşiği
     aşarsa, configs/config.py içindeki ENABLE_EMBED_LORA bayrağını True yapmanızı
     ÖNERİR (otomatik değiştirmez — bu bilinçli bir insan kararı olarak bırakılmıştır,
     çünkü embed_tokens'a LoRA eklemek eğitilebilir parametre sayısını artırır).

Çıktı: configs.config.ANALYSIS_OUTPUT_DIR / "tokenizer_analysis_report.json"
"""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from configs import config  # noqa: E402

from transformers import AutoTokenizer  # type: ignore

TURKISH_SPECIAL_CHARS = list("çğıöşüÇĞİÖŞÜ")

# Tokenizer için özel bir veri seti indirilmeden de EDA'nın çalışabilmesi için, özel
# karakter içeren ve içermeyen küçük ama temsili birer Türkçe kelime listesi.
_SPECIAL_CHAR_WORDS = [
    "çocuk", "ağaç", "ışık", "öğretmen", "şeker", "üzüm", "değişiklik", "güneş",
    "çiçek", "gökyüzü", "öğrenci", "üniversite", "şirket", "çalışma", "İstanbul",
    "Öğrenci", "Şehir", "Çanakkale", "yağmur", "büyük", "küçük", "gözlük",
]
_PLAIN_WORDS = [
    "kalem", "masa", "kitap", "ev", "araba", "bardak", "sandalye", "telefon",
    "bilgisayar", "market", "sokak", "kapı", "pencere", "duvar", "zaman", "insan",
    "hayat", "dünya", "para", "iş", "spor", "tarih",
]


def _try_load_dataset_sample(name: str, max_words: int = 500) -> list[str]:
    """Varsa hazırlanmış bir veri setinden (data/prepare_datasets.py çıktısı) gerçek
    Türkçe kelime örnekleri çeker; yoksa boş liste döner (fallback listeler kullanılır)."""
    try:
        sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
        from data import io_utils  # noqa: E402

        ds = io_utils.load_ocr_records(name)
    except Exception:
        return []
    words = []
    for row in ds.select(range(min(len(ds), max_words))):
        words.extend(row["text"].split())
    return words


def _tokens_per_char(tokenizer, text: str) -> float:
    if not text:
        return 0.0
    n_tokens = len(tokenizer.encode(text, add_special_tokens=False))
    return n_tokens / max(len(text), 1)


def analyze_single_characters(tokenizer) -> dict:
    """Her özel karakteri TEK BAŞINA tokenize eder; kaç token'a bölündüğünü raporlar."""
    report = {}
    for ch in TURKISH_SPECIAL_CHARS:
        ids = tokenizer.encode(ch, add_special_tokens=False)
        tokens = tokenizer.convert_ids_to_tokens(ids)
        report[ch] = {"n_tokens": len(ids), "tokens": tokens}
    return report


def analyze_word_lists(tokenizer, special_words: list[str], plain_words: list[str]) -> dict:
    special_text = " ".join(special_words)
    plain_text = " ".join(plain_words)

    special_tpc = _tokens_per_char(tokenizer, special_text)
    plain_tpc = _tokens_per_char(tokenizer, plain_text)

    fragmentation_ratio = special_tpc / plain_tpc if plain_tpc > 0 else float("inf")

    return {
        "n_special_words": len(special_words),
        "n_plain_words": len(plain_words),
        "special_words_tokens_per_char": round(special_tpc, 4),
        "plain_words_tokens_per_char": round(plain_tpc, 4),
        "fragmentation_ratio": round(fragmentation_ratio, 4),
    }


# Bu eşiğin üzerinde bir fragmentation_ratio, "özel karakter içeren kelimeler, içermeyenlere
# göre karakter başına belirgin şekilde daha fazla token harcıyor" anlamına gelir ve
# embed_tokens'a LoRA eklenmesi ÖNERİLİR. 1.15 (~%15 fazla token/karakter) makul, orta
# düzey bir eşik olarak seçilmiştir; kesin bir bilimsel değer değildir — raporu okuyup
# kendi kararınızı verin.
FRAGMENTATION_RECOMMENDATION_THRESHOLD = 1.15


def main() -> None:
    config.ensure_directories()
    print(f"[tokenizer_analysis] Tokenizer yükleniyor: {config.MODEL_ID}")
    tokenizer = AutoTokenizer.from_pretrained(config.MODEL_ID)

    # Gerçek veriden örnek çekmeyi dene; yoksa dahili listelere düş.
    real_words = _try_load_dataset_sample("printed_synthetic")
    special_words = [w for w in real_words if any(c in TURKISH_SPECIAL_CHARS for c in w)] or _SPECIAL_CHAR_WORDS
    plain_words = [w for w in real_words if not any(c in TURKISH_SPECIAL_CHARS for c in w)] or _PLAIN_WORDS

    single_char_report = analyze_single_characters(tokenizer)
    word_list_report = analyze_word_lists(tokenizer, special_words, plain_words)

    ratio = word_list_report["fragmentation_ratio"]
    recommend_embed_lora = ratio >= FRAGMENTATION_RECOMMENDATION_THRESHOLD

    print("\n=== Tek karakter tokenizasyonu ===")
    for ch, info in single_char_report.items():
        print(f"  '{ch}': {info['n_tokens']} token -> {info['tokens']}")

    print("\n=== Kelime listesi karşılaştırması ===")
    for k, v in word_list_report.items():
        print(f"  {k}: {v}")

    print(f"\n=== Parçalanma oranı: {ratio} (eşik: {FRAGMENTATION_RECOMMENDATION_THRESHOLD}) ===")
    if recommend_embed_lora:
        print(
            "ÖNERİ: Türkçe özel karakterler belirgin şekilde daha fazla parçalanıyor. "
            "configs/config.py içinde ENABLE_EMBED_LORA = True yapmanız önerilir."
        )
    else:
        print(
            "ÖNERİ: Parçalanma farkı eşik altında; embed_tokens LoRA'sı GEREKLİ görünmüyor "
            "(ENABLE_EMBED_LORA = False olarak bırakılabilir)."
        )

    report = {
        "model_id": config.MODEL_ID,
        "single_character_tokenization": single_char_report,
        "word_list_comparison": word_list_report,
        "fragmentation_recommendation_threshold": FRAGMENTATION_RECOMMENDATION_THRESHOLD,
        "recommend_embed_lora": recommend_embed_lora,
    }
    out_path = config.ANALYSIS_OUTPUT_DIR / "tokenizer_analysis_report.json"
    out_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n[tokenizer_analysis] Rapor kaydedildi: {out_path}")


if __name__ == "__main__":
    main()
