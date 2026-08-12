"""
evaluate.py
===========
Test Seti A ve Test Seti B üzerinde bir modeli (taban model VEYA LoRA uygulanmış model)
çalıştırıp metrikleri hesaplayan ve JSON olarak kaydeden değerlendirme çalıştırıcısı.

Test Seti A İÇİN TEK BİR "kompozit skor" üretilir (callbacks.py'nin regresyon eşiğini
tek bir sayı üzerinden karşılaştırabilmesi için): Test A hem OCR-karışık hem genel-görev
örnekleri içerdiğinden (bkz. evaluation/build_test_sets.py), kompozit skor bu ikisinin
config.TEST_A_GENERAL_RATIO ile ağırlıklandırılmış bir ortalamasıdır:

    composite = (1 - TEST_A_GENERAL_RATIO) * (1 - CER_ocr) + TEST_A_GENERAL_RATIO * ROUGE_L_genel

(1 - CER) kullanılması, CER'in "düşük=iyi" olmasını ROUGE-L ile aynı yöne, yani
"yüksek=iyi" yönüne çevirmek içindir; böylece composite skor da her zaman "yüksek=iyi"
olur ve callbacks.py basitçe "düştü mü?" sorusunu sorabilir.

Test Seti B SAF OCR metrikleriyle (CER/WER/exact-match) raporlanır; bu setin YÜKSELMESİ
(CER'in DÜŞMESİ) beklenir.
"""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from configs import config  # noqa: E402
from evaluation import build_test_sets, metrics  # noqa: E402
from training.inference_utils import generate_batch  # noqa: E402

from tqdm import tqdm  # type: ignore


def _checkpoint_path(tag: str, split_name: str) -> Path:
    """Her (tag, split) çifti için AYRI bir checkpoint dosyası: farklı bir etiketle
    (ör. 'epoch_1') yapılan bir değerlendirme, farklı bir model durumuna ait olduğu
    için 'baseline' checkpoint'iyle asla karışmamalı."""
    path = config.EVAL_OUTPUT_DIR / "checkpoints" / f"{tag}_{split_name}.jsonl"
    path.parent.mkdir(parents=True, exist_ok=True)
    return path


def _load_checkpoint(path: Path) -> dict[int, str]:
    if not path.exists():
        return {}
    entries: dict[int, str] = {}
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            entries[rec["index"]] = rec["hypothesis"]
    return entries


def _generate_for_dataset(model, processor, ds, desc: str, checkpoint_path: Path) -> list[str]:
    """Test A (157) + Test B (200) örneği tek seferde üretmek T4'te uzun sürebilir;
    bağlantı kopması gibi bir kesinti TÜM ilerlemeyi kaybettirmesin diye her üretilen
    hipotez, üretilir üretilmez Drive'daki (KALICI) bir .jsonl checkpoint dosyasına
    satır satır yazılıp flush edilir. Script yeniden başlatıldığında, checkpoint'te
    zaten bulunan indeksler ATLANIR (bkz. replay_generation.py'deki aynı desen)."""
    done = _load_checkpoint(checkpoint_path)
    if done:
        print(
            f"      Önceki bir çalıştırmadan {len(done)} doğrulanmış '{desc}' "
            "kaydı bulundu (checkpoint); bunlar yeniden üretilmeyecek."
        )

    hypotheses: list[str] = [""] * len(ds)
    with open(checkpoint_path, "a", encoding="utf-8") as checkpoint_file:
        for i, row in enumerate(tqdm(ds, desc=desc)):
            if i in done:
                hypotheses[i] = done[i]
                continue
            hyp = generate_batch(model, processor, [row["image"]], [row["instruction"]])[0]
            hypotheses[i] = hyp
            checkpoint_file.write(json.dumps({"index": i, "hypothesis": hyp}) + "\n")
            checkpoint_file.flush()
    return hypotheses


def evaluate_test_a(model, processor, tag: str) -> dict:
    ds = build_test_sets.load_test_a()
    checkpoint_path = _checkpoint_path(tag, "test_a")
    hypotheses = _generate_for_dataset(model, processor, ds, "Test Seti A", checkpoint_path)
    references = ds["answer"]
    sources = ds["source"]

    is_general = [s == "replay_general" for s in sources]
    general_refs = [r for r, g in zip(references, is_general) if g]
    general_hyps = [h for h, g in zip(hypotheses, is_general) if g]
    ocr_refs = [r for r, g in zip(references, is_general) if not g]
    ocr_hyps = [h for h, g in zip(hypotheses, is_general) if not g]

    ocr_metrics = metrics.compute_ocr_metrics(ocr_refs, ocr_hyps)
    general_metrics = metrics.compute_general_metrics(general_refs, general_hyps)

    composite = (1 - config.TEST_A_GENERAL_RATIO) * (1 - ocr_metrics["cer"]) + (
        config.TEST_A_GENERAL_RATIO
    ) * general_metrics["rouge_l"]

    return {
        "n_examples": len(ds),
        "ocr_mixed_metrics": ocr_metrics,
        "general_metrics": general_metrics,
        "composite_score": composite,
    }


def evaluate_test_b(model, processor, tag: str) -> dict:
    ds = build_test_sets.load_test_b()
    checkpoint_path = _checkpoint_path(tag, "test_b")
    hypotheses = _generate_for_dataset(model, processor, ds, "Test Seti B", checkpoint_path)
    references = ds["answer"]
    ocr_metrics = metrics.compute_ocr_metrics(references, hypotheses)
    return {"n_examples": len(ds), **ocr_metrics}


def evaluate_all(model, processor, tag: str) -> dict:
    """Test A + Test B'yi çalıştırır, sonucu EVAL_OUTPUT_DIR/{tag}.json olarak kaydeder."""
    config.ensure_directories()
    print(f"[evaluate] '{tag}' etiketiyle değerlendirme başlıyor...")
    result = {
        "tag": tag,
        "test_a": evaluate_test_a(model, processor, tag),
        "test_b": evaluate_test_b(model, processor, tag),
    }
    out_path = config.EVAL_OUTPUT_DIR / f"{tag}.json"
    out_path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[evaluate] Sonuç kaydedildi: {out_path}")
    print(f"           Test A composite_score = {result['test_a']['composite_score']:.4f}")
    print(f"           Test B CER = {result['test_b']['cer']:.4f}")
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description="Test Seti A/B üzerinde model değerlendirmesi.")
    parser.add_argument(
        "--tag", default="baseline", help="Sonuç dosyasının etiketi (örn. 'baseline', 'epoch_1')."
    )
    parser.add_argument(
        "--adapter-path",
        default=None,
        help="Değerlendirilecek bir LoRA adaptör klasörü (verilmezse SADECE taban model değerlendirilir).",
    )
    args = parser.parse_args()

    from training.lora_setup import load_base_model_and_processor

    model, processor = load_base_model_and_processor(load_in_4bit=config.LOAD_IN_4BIT)

    if args.adapter_path:
        from peft import PeftModel  # type: ignore

        model = PeftModel.from_pretrained(model, args.adapter_path)

    model.eval()
    evaluate_all(model, processor, tag=args.tag)


if __name__ == "__main__":
    main()
