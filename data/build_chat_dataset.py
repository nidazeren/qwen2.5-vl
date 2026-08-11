"""
build_chat_dataset.py
======================
Bu script, data/prepare_datasets.py ve data/replay_generation.py tarafından üretilmiş
HAM kaynakları alıp:

  1) Test Seti A (regresyon kontrolü) ve Test Seti B (Türkçe el yazısı/özel karakter
     iyileşme kontrolü) ile küçük bir Validation setini, kaynaklardan SIZINTI OLMADAN
     (yani eğitim setiyle örtüşmeyen örneklerden) ayırır,
  2) config.compute_bucket_target_sizes()'e göre eğitim karışımını (%40 basılı / %35 el
     yazısı / %25 replay ve alt-oranları) oluşturur,
  3) Her örnek için user-mesajındaki TALİMATI çözer (replay kaynaklarında SAKLANMIŞ
     promptu aynen kullanır; ground-truth OCR kaynaklarında data/prompts.py havuzundan
     rastgele bir talimat seçer) ve nihai düz (flat) şemaya dönüştürür:

         {"image": PIL.Image, "instruction": str, "answer": str, "source": str}

     NOT: Gerçek "user-image-assistant" sohbet mesajı YAPISI ve kayıp maskeleme
     (loss masking, yalnızca asistan cevabı üzerinden loss) burada DEĞİL, eğitim
     sırasında training/collate.py içinde, her batch anında kurulur. Bunun nedeni,
     PIL.Image nesnelerini iç içe (nested) bir "messages" listesi içinde Arrow
     formatında güvenilir şekilde saklamanın kırılgan olması; düz sütunlar (image,
     instruction, answer, source) hem diskte güvenilir hem de bellek-verimli
     (memory-mapped, tembel görsel çözme) şekilde saklanabilir.

Çıktı, config.PROCESSED_DATA_DIR altına dört ayrı `datasets.Dataset` olarak yazılır:
  train/, val/, test_a/, test_b/
"""

import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from configs import config  # noqa: E402
from data import io_utils, prompts  # noqa: E402

import datasets  # type: ignore

OCR_SOURCE_NAMES = ["printed_synthetic", "scene_text", "handwriting_synthetic"] + (
    ["smhd_english"] if config.USE_SMHD else []
)
REPLAY_OCR_NAME = "replay_ocr"
REPLAY_GENERAL_NAME = "replay_general"

TURKISH_SPECIAL_CHARS = set("çğıöşüÇĞİÖŞÜ")

_FINAL_FEATURES = datasets.Features(
    {
        "image": datasets.Image(),
        "instruction": datasets.Value("string"),
        "answer": datasets.Value("string"),
        "source": datasets.Value("string"),
    }
)


# ---------------------------------------------------------------------------
# Yardımcı fonksiyonlar
# ---------------------------------------------------------------------------
def _load_and_shuffle(name: str) -> datasets.Dataset:
    return io_utils.load_ocr_records(name).shuffle(seed=config.RANDOM_SEED)


def _take(ds: datasets.Dataset, n: int) -> tuple[datasets.Dataset, datasets.Dataset]:
    """Baştan n örneği ayırır; (alınan, kalan) döner. `n`, ds boyutunu aşarsa tüm ds alınır."""
    n = min(n, len(ds))
    return ds.select(range(n)), ds.select(range(n, len(ds)))


def _sample_with_replacement(ds: datasets.Dataset, n: int, rng: random.Random) -> datasets.Dataset:
    """n örnek seçer: havuz yeterince büyükse tekrarsız, küçükse tekrarlı (Dataset.select
    tekrarlı indeksleri destekler, bu yüzden küçük ham havuzlarda bile pipeline kırılmaz)."""
    if len(ds) == 0:
        return ds
    if len(ds) >= n:
        indices = rng.sample(range(len(ds)), n)
    else:
        print(f"      !! Havuzda {len(ds)} örnek var, {n} isteniyor; tekrarlı örnekleme yapılacak.")
        indices = [rng.randrange(len(ds)) for _ in range(n)]
    return ds.select(indices)


def _distribute_evenly(total: int, keys: list[str]) -> dict[str, int]:
    """`total`u `keys` arasında (mümkün olduğunca) eşit dağıtır."""
    if not keys:
        return {}
    base, remainder = divmod(total, len(keys))
    return {k: base + (1 if i < remainder else 0) for i, k in enumerate(keys)}


def _finalize(ds: datasets.Dataset, rng: random.Random) -> datasets.Dataset:
    """{'image','text','source','prompt'} -> {'image','instruction','answer','source'}.

    `prompt` sütunu doluysa (SADECE replay_ocr/replay_general kaynaklarında dolu olur,
    bkz. io_utils.save_ocr_records docstring'i) AYNEN kullanılır — çünkü o metin zaten o
    prompta karşılık üretilmiştir. Diğer tüm (ground-truth) OCR kaynaklarında `prompt`
    her zaman boştur ve OCR_PROMPT_POOL'dan rastgele bir talimat seçilir (genel-görev
    promptuna asla düşülmez, çünkü genel-görev örnekleri zaten kendi promptunu taşır)."""
    if len(ds) == 0:
        return datasets.Dataset.from_dict(
            {"image": [], "instruction": [], "answer": [], "source": []}, features=_FINAL_FEATURES
        )

    def _map_fn(example):
        stored_prompt = example.get("prompt") or ""
        instruction = stored_prompt if stored_prompt else prompts.sample_ocr_prompt(rng)
        return {
            "image": example["image"],
            "instruction": instruction,
            "answer": example["text"],
            "source": example["source"],
        }

    # load_from_cache_file=False: instruction seçimi rng durumuna bağlı (stokastik) olduğu
    # için, `datasets` kütüphanesinin fonksiyon-imzasına dayalı önbelleğe düşüp eski/yanlış
    # bir sonucu geri döndürmesini engelliyoruz.
    return ds.map(
        _map_fn, remove_columns=ds.column_names, features=_FINAL_FEATURES, load_from_cache_file=False
    )


def _contains_turkish_special_char(example) -> bool:
    return any(ch in TURKISH_SPECIAL_CHARS for ch in example["text"])


# ---------------------------------------------------------------------------
# Ana akış
# ---------------------------------------------------------------------------
def main() -> None:
    config.ensure_directories()
    rng = random.Random(config.RANDOM_SEED)
    bucket_sizes = config.compute_bucket_target_sizes()
    print(f"[build_chat_dataset] Hedef kova boyutları: {bucket_sizes}")

    pools = {name: _load_and_shuffle(name) for name in OCR_SOURCE_NAMES}
    pools[REPLAY_OCR_NAME] = _load_and_shuffle(REPLAY_OCR_NAME)
    pools[REPLAY_GENERAL_NAME] = _load_and_shuffle(REPLAY_GENERAL_NAME)

    # -----------------------------------------------------------------
    # 1) VAL: OCR kaynakları arasında (replay HARİÇ) eşit dağıtılmış küçük bir dilim.
    # -----------------------------------------------------------------
    val_parts = []
    val_alloc = _distribute_evenly(config.VAL_SIZE, OCR_SOURCE_NAMES)
    for name, n in val_alloc.items():
        taken, pools[name] = _take(pools[name], n)
        val_parts.append(taken)
    val_raw = datasets.concatenate_datasets(val_parts).shuffle(seed=config.RANDOM_SEED)

    # -----------------------------------------------------------------
    # 2) TEST SETİ B: Türkçe el yazısı / özel karakter ağırlıklı.
    #    handwriting_synthetic'ten, ÖNCELİKLE Türkçe özel karakter (ç,ğ,ı,ö,ş,ü) içeren
    #    örnekler seçilir; yeterli sayıda yoksa geri kalan rastgele örneklerle tamamlanır.
    # -----------------------------------------------------------------
    hs_pool = pools["handwriting_synthetic"]
    special_subset = hs_pool.filter(_contains_turkish_special_char)
    plain_subset = hs_pool.filter(lambda ex: not _contains_turkish_special_char(ex))

    test_b_from_special, special_subset = _take(special_subset, config.TEST_B_SIZE)
    remaining_needed = config.TEST_B_SIZE - len(test_b_from_special)
    test_b_from_plain, plain_subset = _take(plain_subset, remaining_needed)
    test_b_raw = datasets.concatenate_datasets([test_b_from_special, test_b_from_plain]).shuffle(
        seed=config.RANDOM_SEED
    )
    # Kullanılmayan kalanları tekrar birleştirip handwriting_synthetic havuzunu güncelle.
    pools["handwriting_synthetic"] = datasets.concatenate_datasets([special_subset, plain_subset])

    # -----------------------------------------------------------------
    # 3) TEST SETİ A: yarısı genel-görev (replay_general'dan), yarısı karışık-kaynak OCR.
    # -----------------------------------------------------------------
    n_general = round(config.TEST_A_SIZE * config.TEST_A_GENERAL_RATIO)
    n_ocr_mixed = config.TEST_A_SIZE - n_general

    test_a_general, pools[REPLAY_GENERAL_NAME] = _take(pools[REPLAY_GENERAL_NAME], n_general)

    ocr_mixed_alloc = _distribute_evenly(n_ocr_mixed, OCR_SOURCE_NAMES)
    ocr_mixed_parts = []
    for name, n in ocr_mixed_alloc.items():
        taken, pools[name] = _take(pools[name], n)
        ocr_mixed_parts.append(taken)

    test_a_raw = datasets.concatenate_datasets([test_a_general] + ocr_mixed_parts).shuffle(
        seed=config.RANDOM_SEED
    )

    # -----------------------------------------------------------------
    # 4) EĞİTİM KARIŞIMI: kalan havuzlardan compute_bucket_target_sizes() hedeflerine göre
    #    (gerekirse tekrarlı) örnekleme yapılır.
    # -----------------------------------------------------------------
    train_parts = []
    for name in OCR_SOURCE_NAMES + [REPLAY_OCR_NAME, REPLAY_GENERAL_NAME]:
        n = bucket_sizes.get(name, 0)
        if n <= 0:
            continue
        train_parts.append(_sample_with_replacement(pools[name], n, rng))
    train_raw = datasets.concatenate_datasets(train_parts).shuffle(seed=config.RANDOM_SEED)

    # -----------------------------------------------------------------
    # 5) Talimatları çöz (instruction/answer'a dönüştür) ve diske yaz.
    # -----------------------------------------------------------------
    splits = {
        "train": train_raw,
        "val": val_raw,
        "test_a": test_a_raw,
        "test_b": test_b_raw,
    }

    for split_name, raw_ds in splits.items():
        final_ds = _finalize(raw_ds, rng)
        out_dir = config.PROCESSED_DATA_DIR / split_name
        final_ds.save_to_disk(str(out_dir))
        print(f"[build_chat_dataset] {split_name}: {len(final_ds)} örnek -> {out_dir}")

    print("\n[build_chat_dataset] Tamamlandı. Sıradaki adım: training/train_sft.py")


if __name__ == "__main__":
    main()
