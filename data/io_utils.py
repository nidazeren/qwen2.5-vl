"""
io_utils.py
===========
data/ paketi içindeki birden fazla script (prepare_datasets.py, replay_generation.py,
build_chat_dataset.py) AYNI ortak şemayı ({"image","text","source"}) diske yazıp geri
okuduğu için, bu tekrar eden mantık TEK bir yerde toplanmıştır (DRY).
"""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from configs import config  # noqa: E402

import datasets  # type: ignore
from PIL import Image  # type: ignore


def save_ocr_records(records: list[dict], name: str) -> Path:
    """{'image': PIL.Image, 'text': str, 'source': str, 'prompt': str (opsiyonel)}
    kayıtlarını RAW_DATA_DIR altına bir HF `datasets.Dataset` (Arrow) olarak kaydeder.

    `prompt` alanı yalnızca replay_generation.py tarafından üretilen kaynaklarda
    (replay_ocr, replay_general) DOLU olur: bu kaynaklarda hedef metin ("text") belirli
    bir promptla modelden üretildiği için, build_chat_dataset.py eğitim örneğini
    kurarken AYNI promptu kullanmak ZORUNDADIR (yeni rastgele bir prompt seçilirse,
    kayıtlı hedef metinle eşleşmeyen bir talimat-cevap çifti oluşur). Ground-truth OCR
    kaynaklarında (printed_synthetic, scene_text, handwriting_synthetic, smhd_english)
    bu alan boş bırakılır ve build_chat_dataset.py her seferinde rastgele bir OCR
    promptu seçer (prompt çeşitliliği için)."""
    out_dir = config.RAW_DATA_DIR / name
    ds = datasets.Dataset.from_list(
        [
            {
                "image": r["image"],
                "text": r["text"],
                "source": r["source"],
                "prompt": r.get("prompt", ""),
            }
            for r in records
        ]
    )
    ds = ds.cast_column("image", datasets.Image())
    ds.save_to_disk(str(out_dir))
    print(f"      Kaydedildi: {out_dir} ({len(ds)} örnek)")
    return out_dir


def load_ocr_records(name: str) -> datasets.Dataset:
    """save_ocr_records ile yazılmış bir kaynağı geri okur."""
    in_dir = config.RAW_DATA_DIR / name
    if not in_dir.exists():
        raise FileNotFoundError(
            f"{in_dir} bulunamadı. Önce data/prepare_datasets.py (ve gerekiyorsa "
            "data/replay_generation.py) çalıştırılmalı."
        )
    return datasets.Dataset.load_from_disk(str(in_dir))


def save_image_index(records: list[dict], name: str) -> Path:
    """Yalnızca görsel YOLLARINI (etiketsiz), her satırda bir JSON nesnesi olacak
    şekilde bir .jsonl dosyasına yazar. `records` elemanları en az 'image_path' ve
    'source' anahtarlarını içermelidir (image_path -> str veya Path)."""
    out_path = config.RAW_DATA_DIR / f"{name}.jsonl"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps({"image_path": str(r["image_path"]), "source": r["source"]}) + "\n")
    print(f"      Kaydedildi: {out_path} ({len(records)} görsel yolu)")
    return out_path


def load_image_index(name: str) -> list[dict]:
    """save_image_index ile yazılmış bir .jsonl dosyasını okur."""
    in_path = config.RAW_DATA_DIR / f"{name}.jsonl"
    if not in_path.exists():
        raise FileNotFoundError(f"{in_path} bulunamadı. Önce data/prepare_datasets.py çalıştırılmalı.")
    records = []
    with open(in_path, "r", encoding="utf-8") as f:
        for line in f:
            records.append(json.loads(line))
    return records


def open_image(path_like) -> Image.Image:
    return Image.open(path_like).convert("RGB")
