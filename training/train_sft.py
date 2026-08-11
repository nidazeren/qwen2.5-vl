"""
train_sft.py
============
Ana eğitim scripti. `trl.SFTTrainer` kullanır, ama veri seti şemamız ("image",
"instruction", "answer", "source" düz sütunları) TRL'nin varsayılan otomatik VLM
hazırlama mantığıyla birebir örtüşmediği için, TRL'ye kendi hazırlama adımını
ATLAMASI söylenir (`dataset_kwargs={"skip_prepare_dataset": True}`) ve TÜM batch
oluşturma işi training/collate.py'deki özel collator'a bırakılır.

Akış:
  1) Taban modeli + processor'ı donanıma göre (T4->fp16/sdpa, A100->bf16/flash-attn2)
     yükler (training/lora_setup.py).
  2) LoRA'yı uygular (LLM attn+mlp + merger + [opsiyonel] embed/vision).
  3) LoRA B matrisleri PEFT'te SIFIRA başlatıldığı için, LoRA uygulandıktan HEMEN SONRA
     ama eğitim BAŞLAMADAN ÖNCE alınan bir değerlendirme, matematiksel olarak taban
     modelin performansına EŞİTTİR. Bu yüzden ayrıca "çıplak" bir taban model
     yüklemeye gerek kalmadan, eksik olması halinde `baseline.json` burada otomatik
     üretilir (bkz. `_ensure_baseline`).
  4) data/build_chat_dataset.py çıktısı olan train/val setlerini yükler.
  5) SFTTrainer'ı özel collator ve TestABRegressionCallback ile kurar, eğitir.
  6) LoRA adaptörünü ve processor'ı config.CHECKPOINT_DIR altına kaydeder.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from configs import config  # noqa: E402
from training.lora_setup import apply_lora, load_base_model_and_processor  # noqa: E402
from training.collate import Qwen25VLDataCollator  # noqa: E402
from training.callbacks import TestABRegressionCallback  # noqa: E402
from evaluation.evaluate import evaluate_all  # noqa: E402

import torch  # type: ignore
import datasets  # type: ignore
from trl import SFTConfig, SFTTrainer  # type: ignore


def _load_processed_split(name: str) -> datasets.Dataset:
    split_dir = config.PROCESSED_DATA_DIR / name
    if not split_dir.exists():
        raise FileNotFoundError(
            f"{split_dir} bulunamadı. Önce data/prepare_datasets.py -> "
            "data/replay_generation.py -> data/build_chat_dataset.py sırasıyla çalıştırılmalı."
        )
    return datasets.Dataset.load_from_disk(str(split_dir))


def _ensure_baseline(model, processor) -> None:
    """baseline.json yoksa, (LoRA B matrisleri sıfır olduğundan taban modelle
    matematiksel olarak ÖZDEŞ olan) mevcut modeli 'baseline' etiketiyle değerlendirir."""
    baseline_path = config.EVAL_OUTPUT_DIR / "baseline.json"
    if baseline_path.exists():
        print(f"[train_sft] Mevcut baseline bulundu: {baseline_path}")
        return
    print(
        "[train_sft] baseline.json bulunamadı; LoRA B-matrisleri sıfır başlatıldığı için "
        "mevcut (henüz eğitilmemiş) model taban modelle özdeştir -> baseline şimdi üretiliyor."
    )
    was_training = model.training
    model.eval()
    evaluate_all(model, processor, tag="baseline")
    if was_training:
        model.train()


def main() -> None:
    config.ensure_directories()

    print("[train_sft] Model + processor yükleniyor...")
    model, processor = load_base_model_and_processor(load_in_4bit=config.LOAD_IN_4BIT)

    print("[train_sft] LoRA uygulanıyor...")
    model = apply_lora(model)

    _ensure_baseline(model, processor)

    print("[train_sft] Eğitim/validation setleri yükleniyor...")
    train_ds = _load_processed_split("train")
    val_ds = _load_processed_split("val")
    print(f"[train_sft] train={len(train_ds)} örnek, val={len(val_ds)} örnek")

    collator = Qwen25VLDataCollator(processor)

    # Donanıma göre seçilmiş dtype'a uygun karma-hassasiyet bayrağını belirle
    # (load_base_model_and_processor içinde zaten aynı mantıkla model yüklendi;
    # burada TrainingArguments'ın bf16/fp16 bayraklarını modelle TUTARLI tutuyoruz).
    model_dtype = next(model.parameters()).dtype
    use_bf16 = model_dtype == torch.bfloat16
    use_fp16 = model_dtype == torch.float16

    sft_config = SFTConfig(
        output_dir=str(config.CHECKPOINT_DIR / "trainer_output"),
        logging_dir=str(config.LOG_DIR),
        num_train_epochs=config.NUM_TRAIN_EPOCHS,
        per_device_train_batch_size=config.PER_DEVICE_TRAIN_BATCH_SIZE,
        per_device_eval_batch_size=config.PER_DEVICE_TRAIN_BATCH_SIZE,
        gradient_accumulation_steps=config.GRADIENT_ACCUMULATION_STEPS,
        learning_rate=config.LEARNING_RATE,
        lr_scheduler_type=config.LR_SCHEDULER_TYPE,
        warmup_ratio=config.WARMUP_RATIO,
        weight_decay=config.WEIGHT_DECAY,
        bf16=use_bf16,
        fp16=use_fp16,
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        logging_steps=5,
        eval_strategy="epoch",
        save_strategy="epoch",
        save_total_limit=2,
        report_to=["tensorboard"],
        remove_unused_columns=False,  # ÖNEMLİ: özel collator ham 'image/instruction/answer' sütunlarını bekler
        dataset_kwargs={"skip_prepare_dataset": True},  # TRL'nin otomatik VLM hazırlama/tokenizasyon adımını atla
        # NOT: max_length/max_seq_length BİLEREK burada YOK. skip_prepare_dataset=True
        # olduğu için TRL kendi tokenizasyon/packing adımını hiç çalıştırmıyor; kesme
        # (truncation) zaten training/collate.py içindeki Qwen25VLDataCollator tarafından
        # config.MAX_SEQ_LENGTH ile uygulanıyor. Bu alanı burada TEKRAR belirtmek, trl
        # sürümleri arasında değişen alan adlarına (max_length/max_seq_length) bağımlılık
        # yaratıp gereksiz kırılganlık ekler.
    )

    callback = TestABRegressionCallback.from_baseline_report(processor, baseline_tag="baseline")

    trainer = SFTTrainer(
        model=model,
        args=sft_config,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        data_collator=collator,
        processing_class=processor,
        callbacks=[callback],
    )

    print("[train_sft] Eğitim başlıyor...")
    trainer.train()

    final_dir = config.CHECKPOINT_DIR / "final_adapter"
    trainer.model.save_pretrained(str(final_dir))
    processor.save_pretrained(str(final_dir))
    print(f"[train_sft] Eğitim tamamlandı. LoRA adaptörü kaydedildi: {final_dir}")


if __name__ == "__main__":
    main()
