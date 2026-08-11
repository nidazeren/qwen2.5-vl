"""
replay_generation.py
=====================
Self-distillation ile REPLAY verisi üretir. Fikir: ince ayardan ÖNCEKİ (henüz hiçbir
LoRA uygulanmamış) taban model, OmniDocBench görselleri üzerinde çalıştırılır; modelin
KENDİ ürettiği çıktılar, o görsel için "doğru cevap" (hedef metin) olarak kaydedilir.
Bu şekilde üretilen veri, ince ayar sırasında eğitim karışımına (%25 replay payı)
eklenerek modelin eski/genel yeteneklerini büyük ölçüde korumasını (katastrofik
unutmayı azaltmayı) hedefler — çünkü model, ince ayar sırasında "kendi eski davranışını
tekrar üretmeyi" de öğrenir.

İki alt havuz üretilir (data/prompts.py'deki iki AYRI prompt havuzuyla):
  - replay_ocr:     OCR görev-promptlarıyla (data/prompts.py: OCR_PROMPT_POOL)
  - replay_general: OCR DIŞI genel görev promptlarıyla (data/prompts.py: GENERAL_PROMPT_POOL)

Kaç görsel işleneceği config.compute_bucket_target_sizes()'ten (TEK doğruluk kaynağı)
okunur; böylece burada üretilen miktar data/build_chat_dataset.py'nin beklediği miktarla
otomatik olarak tutarlı kalır.

NOT: Bu script, eğitimden ÖNCE, BİR KEZ çalıştırılır (train_sft.py sırasında DEĞİL).
Üretilen pseudo-label'lar RAW_DATA_DIR altına kalıcı olarak kaydedilir.
"""

import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from configs import config  # noqa: E402
from data import io_utils, prompts  # noqa: E402
from training.lora_setup import load_base_model_and_processor  # noqa: E402
from training.inference_utils import generate_batch  # noqa: E402

from tqdm import tqdm  # type: ignore


def _pick_image_paths(pool: list[dict], n: int, rng: random.Random) -> list[str]:
    """Havuzdan n adet görsel yolu seçer. Havuz yeterince büyükse TEKRARSIZ seçilir;
    değilse (görsel sayısı istenenden az ise) tekrarlı seçime (rng.choices) düşülür ve
    kullanıcı bilgilendirilir — bu, küçük ham veri havuzlarında bile pipeline'ın
    kırılmadan çalışmasını sağlar."""
    paths = [r["image_path"] for r in pool]
    if len(paths) >= n:
        return rng.sample(paths, n)
    print(
        f"      !! Havuzda yalnızca {len(paths)} görsel var ama {n} isteniyor; "
        "bazı görseller TEKRAR kullanılacak (rng.choices)."
    )
    return rng.choices(paths, k=n) if paths else []


def _generate_pool(
    model, processor, image_paths: list[str], prompt_sampler, rng: random.Random, source_tag: str
) -> list[dict]:
    records = []
    for path in tqdm(image_paths, desc=f"[replay_generation] {source_tag}"):
        image = io_utils.open_image(path)
        prompt = prompt_sampler(rng)
        generated_text = generate_batch(model, processor, [image], [prompt])[0]
        if not generated_text:
            continue  # boş üretimler (nadiren) atlanır
        # `prompt` burada AÇIKÇA kaydedilir: build_chat_dataset.py bu hedef metni
        # kurarken YENİ bir prompt seçmek yerine tam olarak bunu yeniden kullanmalıdır
        # (bkz. io_utils.save_ocr_records docstring'i).
        records.append(
            {"image": image, "text": generated_text, "source": source_tag, "prompt": prompt}
        )
    return records


def main() -> None:
    config.ensure_directories()
    rng = random.Random(config.RANDOM_SEED)

    bucket_sizes = config.compute_bucket_target_sizes()
    n_ocr = bucket_sizes["replay_ocr"]
    n_general = bucket_sizes["replay_general"]
    print(f"[replay_generation] Hedef: replay_ocr={n_ocr}, replay_general={n_general}")

    # Not: yalnızca klasörlerin VAR OLMASINA değil, mevcut örnek sayısının GÜNCEL hedefi
    # (n_ocr/n_general) KARŞILAYIP KARŞILAMADIĞINA bakılır. Bu, pilot moddan (küçük
    # hedefler) tam moda (büyük hedefler) geçildiğinde, pilotun küçük replay verisinin
    # yanlışlıkla "zaten hazır" sanılıp atlanmasını ÖNLER.
    ocr_out_dir = config.RAW_DATA_DIR / "replay_ocr"
    general_out_dir = config.RAW_DATA_DIR / "replay_general"
    if ocr_out_dir.exists() and general_out_dir.exists():
        existing_ocr = len(io_utils.load_ocr_records("replay_ocr"))
        existing_general = len(io_utils.load_ocr_records("replay_general"))
        if existing_ocr >= n_ocr and existing_general >= n_general:
            print(
                f"[replay_generation] Mevcut replay verisi yeterli (replay_ocr={existing_ocr}"
                f">={n_ocr}, replay_general={existing_general}>={n_general}); üretim atlanıyor. "
                f"Yeniden üretmek isterseniz '{ocr_out_dir}' ve '{general_out_dir}' klasörlerini "
                "silip tekrar çalıştırın."
            )
            return
        print(
            f"[replay_generation] Mevcut replay verisi hedefi karşılamıyor "
            f"(replay_ocr={existing_ocr}/{n_ocr}, replay_general={existing_general}/{n_general}); "
            "yeniden üretiliyor."
        )

    image_pool = io_utils.load_image_index("omnidocbench_replay_source")
    if not image_pool:
        raise RuntimeError(
            "omnidocbench_replay_source.jsonl boş. Önce data/prepare_datasets.py çalıştırılmalı."
        )

    # OCR ve genel-görev alt havuzları için MÜMKÜNSE farklı görseller kullanılır
    # (çeşitliliği artırmak için); havuz küçükse örtüşme olabilir, bu sorun değildir
    # çünkü her ikisinde de farklı prompt ve dolayısıyla farklı model çıktısı üretilir.
    ocr_paths = _pick_image_paths(image_pool, n_ocr, rng)
    general_paths = _pick_image_paths(image_pool, n_general, rng)

    print("[replay_generation] Taban model (LoRA UYGULANMADAN) yükleniyor...")
    model, processor = load_base_model_and_processor(load_in_4bit=config.LOAD_IN_4BIT)
    model.eval()

    ocr_records = _generate_pool(
        model, processor, ocr_paths, prompts.sample_ocr_prompt, rng, "replay_ocr"
    )
    general_records = _generate_pool(
        model, processor, general_paths, prompts.sample_general_prompt, rng, "replay_general"
    )

    io_utils.save_ocr_records(ocr_records, "replay_ocr")
    io_utils.save_ocr_records(general_records, "replay_general")

    print("\n[replay_generation] Tamamlandı. Sıradaki adım: data/build_chat_dataset.py")


if __name__ == "__main__":
    main()
