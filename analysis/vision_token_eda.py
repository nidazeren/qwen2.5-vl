"""
vision_token_eda.py
====================
Qwen2.5-VL, bir görüntüyü kaç "görsel token"e böldüğünü `min_pixels`/`max_pixels`
ayarına göre belirler (görüntü önce bu aralığa göre yeniden boyutlandırılır, sonra
14x14'lük patch'lere, ardından 2x2 birleştirme -merge_size- ile 28x28'lik bloklara
bölünür). Bu script:

  1) Birkaç ADAY (min_pixels, max_pixels) ayarını,
  2) Projedeki gerçek veri kaynaklarından (varsa) örnek görüntüler üzerinde,
  3) Gerçek `Qwen2VLImageProcessor` mantığıyla (resmi formülle) karşılaştırır,

ve her ayar için üretilecek görsel token sayısının dağılımını (min/medyan/ortalama/maks)
raporlar. Böylece configs/config.py içindeki MIN_PIXELS/MAX_PIXELS değerlerini; T4'te
bellek sınırını aşmayacak ama OCR için yeterli çözünürlüğü koruyacak şekilde BİLİNÇLİ
olarak seçebilirsiniz.

Çıktı: configs.config.ANALYSIS_OUTPUT_DIR / "vision_token_eda_report.json"
"""

import json
import statistics
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from configs import config  # noqa: E402

from PIL import Image  # type: ignore
from transformers import AutoProcessor  # type: ignore

# Karşılaştırılacak aday (min_pixels, max_pixels) ayarları; birim: piksel (28*28'in katları).
F = config.IMAGE_PATCH_FACTOR
CANDIDATE_SETTINGS = {
    "dar_T4_dostu (128-512 blok)": (128 * F * F, 512 * F * F),
    "orta_varsayilan (256-768 blok, config.py)": (config.MIN_PIXELS, config.MAX_PIXELS),
    "genis_resmi_varsayilan (256-1280 blok)": (256 * F * F, 1280 * F * F),
}


def _try_load_real_sample_images(max_per_source: int = 20) -> list[Image.Image]:
    """Hazırlanmış veri kaynaklarından (varsa) küçük bir görsel örneklemi çeker."""
    images: list[Image.Image] = []
    try:
        from data import io_utils  # noqa: E402

        for name in ["printed_synthetic", "scene_text", "handwriting_synthetic"]:
            try:
                ds = io_utils.load_ocr_records(name)
            except Exception:
                continue
            for row in ds.select(range(min(len(ds), max_per_source))):
                images.append(row["image"])
    except Exception:
        pass
    return images


def _synthetic_fallback_images() -> list[Image.Image]:
    """Hiç veri hazırlanmamışsa bile EDA'nın çalışabilmesi için, projede sık görülecek
    tipik en-boy oranlarını taklit eden BOŞ (yer tutucu) görseller üretir. Bunlar gerçek
    OCR içeriği taşımaz; yalnızca boyut/token ilişkisini göstermek içindir."""
    sizes = [
        (200, 48),    # esengul3 tarzı kısa kelime görseli
        (800, 48),    # uzun kelime/ifade görseli
        (1024, 768),  # sahne metni / belge fotoğrafı tarzı
        (1600, 1200), # yüksek çözünürlüklü belge sayfası
    ]
    return [Image.new("RGB", size, color=(255, 255, 255)) for size in sizes]


def _n_image_tokens(image_processor, image: Image.Image, min_pixels: int, max_pixels: int) -> int:
    image_processor.min_pixels = min_pixels
    image_processor.max_pixels = max_pixels
    out = image_processor(images=[image], return_tensors="pt")
    grid_thw = out["image_grid_thw"][0]
    merge_size = image_processor.merge_size
    n_tokens = int(grid_thw.prod().item()) // (merge_size ** 2)
    return n_tokens


def main() -> None:
    config.ensure_directories()
    print(f"[vision_token_eda] Processor yükleniyor: {config.MODEL_ID}")
    processor = AutoProcessor.from_pretrained(config.MODEL_ID)
    image_processor = processor.image_processor

    images = _try_load_real_sample_images()
    used_fallback = False
    if not images:
        print("[vision_token_eda] Hazır veri bulunamadı; yer tutucu (sentetik) görseller kullanılacak.")
        images = _synthetic_fallback_images()
        used_fallback = True
    print(f"[vision_token_eda] {len(images)} görsel üzerinde analiz yapılacak (fallback={used_fallback}).")

    report = {"used_fallback_images": used_fallback, "n_images": len(images), "settings": {}}

    for setting_name, (min_px, max_px) in CANDIDATE_SETTINGS.items():
        token_counts = [
            _n_image_tokens(image_processor, img, min_px, max_px) for img in images
        ]
        stats = {
            "min_pixels": min_px,
            "max_pixels": max_px,
            "min_tokens": min(token_counts),
            "median_tokens": statistics.median(token_counts),
            "mean_tokens": round(statistics.fmean(token_counts), 1),
            "max_tokens": max(token_counts),
        }
        report["settings"][setting_name] = stats
        print(f"\n[{setting_name}] min_pixels={min_px}, max_pixels={max_px}")
        print(f"  token sayısı -> min={stats['min_tokens']}, medyan={stats['median_tokens']}, "
              f"ortalama={stats['mean_tokens']}, maks={stats['max_tokens']}")
        print(f"  (config.MAX_SEQ_LENGTH={config.MAX_SEQ_LENGTH} ile karşılaştırın: "
              "görsel token + talimat + hedef metin token'ları bu sınırı aşmamalı)")

    out_path = config.ANALYSIS_OUTPUT_DIR / "vision_token_eda_report.json"
    out_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n[vision_token_eda] Rapor kaydedildi: {out_path}")
    print(
        "\nÖNERİ: 'maks_tokens' değeri MAX_SEQ_LENGTH'in büyük bir kısmını kaplıyorsa "
        "(örn. > %60), T4 pilotunda daha dar bir (min_pixels, max_pixels) aralığı seçin; "
        "A100 tam eğitiminde OCR detayı için daha geniş bir aralık tercih edilebilir."
    )


if __name__ == "__main__":
    main()
