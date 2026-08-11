"""
prepare_datasets.py
====================
Bu script, projede kullanılan TÜM ham veri kaynaklarını indirir (veya yerelden okur) ve
hepsini ORTAK bir şemaya dönüştürür:

    {"image": PIL.Image, "text": str, "source": str}

`source` alanı SADECE bu script içinde iz sürmek / raporlamak için kullanılır; downstream
(build_chat_dataset.py) bu alanı asla prompt metnine YAZMAZ (bkz. data/prompts.py'deki
"kaynağı ifşa etmeme" kuralı) — sadece karışım oranlarını uygulamak ve Test Seti B'yi
(Türkçe el yazısı ağırlıklı) doğru kaynaklardan seçmek için kullanılır.

Kaynaklar ve NASIL indirildikleri:
  1. esengul3/turkish-word-ocr   -> HuggingFace Hub, `datasets.load_dataset` (parquet).
  2. TS-TR (sahne metni)         -> Kaggle, `kagglehub` ile indirilir. Kaggle bu veri
     setinin iç klasör/etiket formatını herkese açık şekilde belgelemediği için, etiket
     dosyası OTOMATİK ALGILANIR (_discover_ts_tr_annotations). Algılama başarısız olursa
     script, indirilen klasörün içeriğini ekrana yazdırıp size TS_TR_MANUAL_HINT
     değişkenini (bu dosyanın altında) doldurmanızı ister — bu tek seferlik, elle bir
     müdahaledir ve README.md'de de anlatılmıştır.
  3. emredeveloper/turkish-ocr   -> HuggingFace Hub, `huggingface_hub.snapshot_download`
     (annotations.jsonl + images/ standart imagefolder+metadata.jsonl biçiminde OLMADIĞI
     için `load_dataset` yerine elle okunur). `text_type == "handwritten"` filtrelenir.
  4. SMHD                        -> Drive'da manuel yerleştirilmiş yerel klasör (izinli
     dağıtım). USE_SMHD=False iken bu kaynak tamamen atlanır.
  5. OmniDocBench                -> HuggingFace Hub, `snapshot_download`; SADECE görseller
     kullanılır (etiketler değil), çünkü bu kaynak replay/self-distillation için görsel
     havuzu olarak kullanılıyor (bkz. replay_generation.py).

Bu script yalnızca Colab'da (Drive bağlıyken) veya internet erişimi olan bir ortamda
çalıştırılmalıdır; yerel VS Code'da SADECE dosya olarak yazılmıştır, çalıştırılmamıştır.
"""

import io
import json
import random
import sys
from pathlib import Path
from typing import Optional

# Repo kökünü içe aktarma yoluna ekle (notebook'lardan da aynı desen kullanılır).
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from configs import config  # noqa: E402
from data import io_utils  # noqa: E402

# NOT: Bu importlar sadece Colab/gerçek çalıştırma ortamında mevcut olacak paketlerdir
# (requirements.txt). Dosya yerelde SADECE yazılıp incelendiği için burada bilerek
# içe aktarma hatası yakalanmıyor — Colab'da requirements kurulduktan sonra hatasız çalışır.
import datasets  # type: ignore
from PIL import Image  # type: ignore
from huggingface_hub import snapshot_download  # type: ignore


# ---------------------------------------------------------------------------
# TS-TR için elle doldurulabilecek "ipucu" — otomatik algılama başarısız olursa kullanılır.
# Örnek: {"annotation_file": "gt.txt", "delimiter": "\t", "image_col": 0, "text_col": 1}
# None bırakılırsa otomatik algılama denenir.
# ---------------------------------------------------------------------------
TS_TR_MANUAL_HINT: Optional[dict] = None


def _to_pil(image_like) -> Image.Image:
    """`datasets` kütüphanesinin Image() sütunundan veya ham bytes'tan PIL.Image üretir."""
    if isinstance(image_like, Image.Image):
        return image_like.convert("RGB")
    if isinstance(image_like, dict) and "bytes" in image_like and image_like["bytes"]:
        return Image.open(io.BytesIO(image_like["bytes"])).convert("RGB")
    if isinstance(image_like, (str, Path)):
        return Image.open(image_like).convert("RGB")
    raise TypeError(f"Bilinmeyen görsel tipi: {type(image_like)}")


# ---------------------------------------------------------------------------
# 1) esengul3/turkish-word-ocr — sentetik basılı Türkçe kelime OCR
# ---------------------------------------------------------------------------
def load_printed_synthetic() -> list[dict]:
    print(f"[1/5] {config.PRINTED_SYNTHETIC_HF_ID} indiriliyor/yükleniyor...")
    ds = datasets.load_dataset(config.PRINTED_SYNTHETIC_HF_ID, split="train")

    # ÖNEMLİ (performans): PILOT_MODE'da yalnızca birkaç yüz örnek gerekiyorken, tüm
    # 225.000 satırı tek tek PIL görüntüsüne çevirmek (aşağıdaki döngü) dakikalarca
    # sürebilir. Bu yüzden PIL dönüşümünden ÖNCE, `datasets` kütüphanesinin tembel
    # (lazy) select/shuffle mekanizmasıyla veri seti küçültülür; yalnızca SEÇİLEN
    # satırlar için görüntü çözme/kopyalama işlemi yapılır.
    if config.PILOT_MODE and len(ds) > config.PILOT_SAMPLES_PER_SOURCE:
        ds = ds.shuffle(seed=config.RANDOM_SEED).select(range(config.PILOT_SAMPLES_PER_SOURCE))

    records = []
    for row in ds:
        records.append(
            {
                "image": _to_pil(row["image"]),
                "text": row["text"].strip(),
                "source": "printed_synthetic",
            }
        )
    print(f"      -> {len(records)} örnek yüklendi.")
    return records


# ---------------------------------------------------------------------------
# 2) TS-TR — gerçek Türkçe sahne metni (Kaggle)
# ---------------------------------------------------------------------------
def _discover_ts_tr_annotations(root: Path) -> list[dict]:
    """İndirilen TS-TR klasöründe etiket dosyasını otomatik bulmaya çalışır.

    Sahne-metni-tanıma veri setlerinde en yaygın kalıplar: `gt.txt` / `labels.txt` gibi
    "goreli_resim_yolu<TAB veya virgül>metin" satırları içeren düz metin dosyaları,
    ya da `annotations.json` / `annotations.csv` gibi yapılandırılmış dosyalardır.
    Bu fonksiyon önce elle verilmiş TS_TR_MANUAL_HINT'e, yoksa sırasıyla bu kalıplara bakar.
    """
    if TS_TR_MANUAL_HINT is not None:
        ann_path = root / TS_TR_MANUAL_HINT["annotation_file"]
        delimiter = TS_TR_MANUAL_HINT.get("delimiter", "\t")
        img_col = TS_TR_MANUAL_HINT.get("image_col", 0)
        text_col = TS_TR_MANUAL_HINT.get("text_col", 1)
        records = []
        with open(ann_path, "r", encoding="utf-8") as f:
            for line in f:
                parts = line.rstrip("\n").split(delimiter)
                if len(parts) <= max(img_col, text_col):
                    continue
                records.append(
                    {"image_path": root / parts[img_col].strip(), "text": parts[text_col].strip()}
                )
        return records

    # Yapılandırılmış dosyaları ara (json/csv/txt), en olası adlardan başlayarak.
    candidate_names = [
        "gt.txt", "labels.txt", "annotations.txt", "train.txt",
        "annotations.json", "labels.json", "gt.json",
        "annotations.csv", "labels.csv", "gt.csv",
    ]
    found = None
    for name in candidate_names:
        matches = list(root.rglob(name))
        if matches:
            found = matches[0]
            break

    if found is None:
        # Hiçbir bilinen isim bulunamadı: klasör yapısını yazdırıp kullanıcıyı bilgilendir.
        print("      !! TS-TR etiket dosyası otomatik bulunamadı. İndirilen klasör içeriği:")
        for p in sorted(root.rglob("*"))[:60]:
            print("        ", p.relative_to(root))
        raise FileNotFoundError(
            "TS-TR etiket dosyası otomatik algılanamadı. Lütfen yukarıdaki listeye bakıp "
            "data/prepare_datasets.py içindeki TS_TR_MANUAL_HINT değişkenini "
            "(annotation_file, delimiter, image_col, text_col) doldurup tekrar çalıştırın."
        )

    print(f"      TS-TR etiket dosyası bulundu: {found.relative_to(root)}")
    records = []
    if found.suffix == ".json":
        data = json.loads(found.read_text(encoding="utf-8"))
        items = data if isinstance(data, list) else data.get("annotations", data.get("data", []))
        for item in items:
            img_key = next((k for k in ("file_name", "image", "image_path", "filename") if k in item), None)
            txt_key = next((k for k in ("text", "label", "transcript", "gt_text") if k in item), None)
            if img_key is None or txt_key is None:
                continue
            records.append({"image_path": root / item[img_key], "text": str(item[txt_key]).strip()})
    else:
        delimiter = "\t" if found.suffix == ".txt" else ","
        with open(found, "r", encoding="utf-8") as f:
            for line in f:
                line = line.rstrip("\n")
                if not line:
                    continue
                parts = line.split(delimiter)
                if len(parts) < 2:
                    continue
                records.append({"image_path": root / parts[0].strip().strip('"'), "text": parts[-1].strip().strip('"')})
    return records


def load_scene_text() -> list[dict]:
    if not config.USE_SCENE_TEXT:
        print("[2/5] TS-TR atlandı (config.USE_SCENE_TEXT=False, Kaggle gerekmez).")
        return []

    print(f"[2/5] TS-TR ({config.SCENE_TEXT_KAGGLE_SLUG}) Kaggle'dan indiriliyor...")
    import kagglehub  # type: ignore

    local_root = Path(kagglehub.dataset_download(config.SCENE_TEXT_KAGGLE_SLUG))
    raw_records = _discover_ts_tr_annotations(local_root)

    records = []
    skipped = 0
    for r in raw_records:
        img_path = Path(r["image_path"])
        if not img_path.exists():
            skipped += 1
            continue
        try:
            image = _to_pil(img_path)
        except Exception:
            skipped += 1
            continue
        records.append({"image": image, "text": r["text"], "source": "scene_text"})

    if skipped:
        print(f"      !! {skipped} kayıt görsel bulunamadığı için atlandı.")
    print(f"      -> {len(records)} örnek yüklendi.")
    return records


# ---------------------------------------------------------------------------
# 3) emredeveloper/turkish-ocr — Türkçe, el yazısı stilinde render edilmiş sentetik veri
# ---------------------------------------------------------------------------
def load_handwriting_synthetic() -> list[dict]:
    print(f"[3/5] {config.HANDWRITING_SYNTHETIC_HF_ID} indiriliyor/yükleniyor...")
    local_root = Path(
        snapshot_download(repo_id=config.HANDWRITING_SYNTHETIC_HF_ID, repo_type="dataset")
    )
    ann_path = local_root / "annotations.jsonl"
    records = []
    with open(ann_path, "r", encoding="utf-8") as f:
        for line in f:
            item = json.loads(line)
            if item.get("text_type") != "handwritten":
                continue  # yalnızca el yazısı stilindeki %70'lik alt küme kullanılır
            img_path = local_root / "images" / item["image"]
            if not img_path.exists():
                continue
            records.append(
                {
                    "image": _to_pil(img_path),
                    "text": item["transcript"].strip(),
                    "source": "handwriting_synthetic",
                }
            )
    print(f"      -> {len(records)} örnek yüklendi (yalnızca 'handwritten' alt küme).")
    return records


# ---------------------------------------------------------------------------
# 4) SMHD — gerçek (ama İngilizce) el yazısı, manuel/izinli dağıtım
# ---------------------------------------------------------------------------
def load_smhd() -> list[dict]:
    if not config.USE_SMHD:
        print("[4/5] SMHD atlandı (config.USE_SMHD=False).")
        return []

    root = config.SMHD_LOCAL_DIR
    if not root.exists() or not any(root.iterdir()):
        print(
            f"      !! config.USE_SMHD=True ama {root} boş/yok. SMHD atlanıyor. "
            "İzin formu ve indirme talimatı için README.md 'SMHD erişimi' bölümüne bakın."
        )
        return []

    print(f"[4/5] SMHD yerel klasörden okunuyor: {root}")
    records = []
    # Beklenen yapı: root/<kategori>/<belge_adı>.png (veya .jpg) + eşlenik .txt transkripsiyon.
    for img_path in list(root.rglob("*.png")) + list(root.rglob("*.jpg")) + list(root.rglob("*.jpeg")):
        txt_path = img_path.with_suffix(".txt")
        if not txt_path.exists():
            continue
        raw_text = txt_path.read_text(encoding="utf-8", errors="ignore")
        # SMHD dokümantasyonuna göre üstü çizilmiş/silinmiş içerik '#' ile işaretlenir;
        # OCR hedefi olarak yalnızca OKUNABİLİR (üstü çizilmemiş) metni istiyoruz.
        clean_text = " ".join(tok for tok in raw_text.split() if not tok.startswith("#")).strip()
        if not clean_text:
            continue
        records.append({"image": _to_pil(img_path), "text": clean_text, "source": "smhd_english"})

    print(f"      -> {len(records)} örnek yüklendi.")
    return records


# ---------------------------------------------------------------------------
# 5) OmniDocBench — replay için yalnızca GÖRSEL kaynağı (etiket kullanılmaz)
# ---------------------------------------------------------------------------
def load_replay_source_images() -> list[dict]:
    print(f"[5/5] {config.REPLAY_SOURCE_HF_ID} (yalnızca görseller) indiriliyor...")
    local_root = Path(snapshot_download(repo_id=config.REPLAY_SOURCE_HF_ID, repo_type="dataset"))
    image_paths = sorted((local_root / "images").glob("*.png"))
    if not image_paths:
        image_paths = sorted(local_root.rglob("*.png"))
    records = [{"image_path": p, "source": "omnidocbench_replay_source"} for p in image_paths]
    print(f"      -> {len(records)} görsel bulundu (etiketsiz; replay_generation.py bunları kullanacak).")
    return records


# ---------------------------------------------------------------------------
# Kaydetme yardımcıları
# ---------------------------------------------------------------------------
def _maybe_subsample(records: list[dict], rng: random.Random) -> list[dict]:
    """PILOT_MODE açıkken her kaynaktan yalnızca küçük bir alt küme kullanır (hızlı duman testi)."""
    if not config.PILOT_MODE or len(records) <= config.PILOT_SAMPLES_PER_SOURCE:
        return records
    return rng.sample(records, config.PILOT_SAMPLES_PER_SOURCE)


def main() -> None:
    config.ensure_directories()
    rng = random.Random(config.RANDOM_SEED)

    printed = _maybe_subsample(load_printed_synthetic(), rng)
    scene = _maybe_subsample(load_scene_text(), rng)
    handwriting_synth = _maybe_subsample(load_handwriting_synthetic(), rng)
    smhd = _maybe_subsample(load_smhd(), rng)
    replay_images = load_replay_source_images()  # replay'de zaten ayrı örnekleme yapılıyor

    io_utils.save_ocr_records(printed, "printed_synthetic")
    if scene:
        io_utils.save_ocr_records(scene, "scene_text")
    io_utils.save_ocr_records(handwriting_synth, "handwriting_synthetic")
    if smhd:
        io_utils.save_ocr_records(smhd, "smhd_english")
    io_utils.save_image_index(replay_images, "omnidocbench_replay_source")

    print("\nTüm kaynaklar hazırlandı. Sıradaki adım: data/replay_generation.py "
          "(self-distillation) ve ardından data/build_chat_dataset.py.")


if __name__ == "__main__":
    main()
