"""
train_sft.py
============
Ana eğitim scripti. `trl.SFTTrainer` (veya faz 5 ablation'ları için
training/distillation_trainer.py:WeightedDistillationTrainer) kullanır, ama veri
seti şemamız ("image", "instruction", "answer", "source" düz sütunları) TRL'nin
varsayılan otomatik VLM hazırlama mantığıyla birebir örtüşmediği için, TRL'ye kendi
hazırlama adımını ATLAMASI söylenir (`dataset_kwargs={"skip_prepare_dataset": True}`)
ve TÜM batch oluşturma işi training/collate.py'deki özel collator'a bırakılır.

Akış:
  1) Taban modeli + processor'ı donanıma göre (T4->fp16/sdpa, A100->bf16/flash-attn2)
     yükler (training/lora_setup.py).
  2) LoRA'yı uygular (LLM attn+mlp + merger + [opsiyonel] embed/vision); HANGİ
     katmanların/projeksiyon türlerinin hedefleneceği config.LORA_LAYER_SCOPE ve
     config.LORA_TARGET_SCOPE ile (ablation koşumu başına) değişebilir.
  3) LoRA B matrisleri PEFT'te SIFIRA başlatıldığı için, LoRA uygulandıktan HEMEN SONRA
     ama eğitim BAŞLAMADAN ÖNCE alınan bir değerlendirme, matematiksel olarak taban
     modelin performansına EŞİTTİR. Bu yüzden ayrıca "çıplak" bir taban model
     yüklemeye gerek kalmadan, eksik olması halinde `baseline.json` burada otomatik
     üretilir (bkz. `_ensure_baseline`) VE TÜM ablation koşumları arasında PAYLAŞILIR.
  4) data/build_chat_dataset.py çıktısı olan train/val setlerini yükler.
  5) Koşumun TÜM ablation-ilgili config değerlerini run_config.json'a yazar (bkz.
     `_write_run_config_snapshot`) -- sonradan analysis/ablation_report.py ile
     koşumlar arası karşılaştırma yapılabilsin diye.
  6) Trainer'ı özel collator, TestABRegressionCallback ve (LORA_DRIFT_LOG_EVERY_N_STEPS>0
     ise) LoRADriftTrackingCallback ile kurar, eğitir. ENABLE_WEIGHTED_LOSS veya
     ENABLE_ONLINE_SELF_DISTILLATION açıksa WeightedDistillationTrainer'a geçer (bkz.
     training/distillation_trainer.py); ikisi de kapalıyken davranış DEĞİŞMEZ (standart
     SFTTrainer, loss_type="nll").
  7) LoRA adaptörünü ve processor'ı config.CHECKPOINT_DIR altına kaydeder (ayrıca
     TestABRegressionCallback, Pareto-kısıtlı en iyi epoch'u training SIRASINDA
     CHECKPOINT_DIR/best_pareto_adapter'a otomatik kaydeder).
"""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from configs import config  # noqa: E402
from training.lora_setup import apply_lora, load_base_model_and_processor  # noqa: E402
from training.collate import Qwen25VLDataCollator  # noqa: E402
from training.callbacks import TestABRegressionCallback  # noqa: E402
from training.lora_drift import LoRADriftTrackingCallback  # noqa: E402
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
    matematiksel olarak ÖZDEŞ olan) mevcut modeli 'baseline' etiketiyle değerlendirir.
    NOT: bu, output_dir'ı AÇIKÇA vermez -> evaluate_all varsayılanı olan (PAYLAŞILAN)
    config.EVAL_OUTPUT_DIR'a yazar; böylece TÜM ablation koşumları aynı baseline'ı
    kullanır (bkz. configs/config.py: EVAL_OUTPUT_DIR vs EVAL_RUN_OUTPUT_DIR notu)."""
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


def _write_run_config_snapshot() -> None:
    """Bu koşumun (RUN_NAME) TÜM ablation-ilgili hiperparametrelerini
    EVAL_RUN_OUTPUT_DIR/run_config.json'a yazar. analysis/ablation_report.py, farklı
    RUN_NAME'ler arasında koşum-koşuluyla-sonuç eşlemesi kurmak için buna bağımlıdır --
    aksi halde "bu run_name hangi LR/layer scope ile eğitildi?" sorusu yalnızca Colab
    hücre geçmişinden takip edilebilirdi (kırılgan)."""
    snapshot = {
        "run_name": config.RUN_NAME,
        "mode_tag": config.MODE_TAG,
        "learning_rate": config.LEARNING_RATE,
        "lora_r": config.LORA_R,
        "lora_alpha": config.LORA_ALPHA,
        "lora_dropout": config.LORA_DROPOUT,
        "lora_layer_scope": config.LORA_LAYER_SCOPE,
        "lora_target_scope": config.LORA_TARGET_SCOPE,
        "num_train_epochs": config.NUM_TRAIN_EPOCHS,
        "mixture_printed": config.MIXTURE_PRINTED,
        "mixture_handwriting": config.MIXTURE_HANDWRITING,
        "mixture_replay": config.MIXTURE_REPLAY,
        "enable_weighted_loss": config.ENABLE_WEIGHTED_LOSS,
        "loss_weight_ocr_tr": config.LOSS_WEIGHT_OCR_TR,
        "loss_weight_handwriting": config.LOSS_WEIGHT_HANDWRITING,
        "loss_weight_replay": config.LOSS_WEIGHT_REPLAY,
        "enable_online_self_distillation": config.ENABLE_ONLINE_SELF_DISTILLATION,
        "distillation_temperature": config.DISTILLATION_TEMPERATURE,
        "loss_weight_distill": config.LOSS_WEIGHT_DISTILL,
        "test_a_regression_relative_threshold": config.TEST_A_REGRESSION_RELATIVE_THRESHOLD,
    }
    out_path = config.EVAL_RUN_OUTPUT_DIR / "run_config.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(snapshot, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[train_sft] Koşum konfigürasyonu kaydedildi: {out_path}")


def main() -> None:
    config.ensure_directories()
    print(f"[train_sft] RUN_NAME={config.RUN_NAME!r}")
    _write_run_config_snapshot()

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
        remove_unused_columns=False,  # ÖNEMLİ: özel collator ham 'image/instruction/answer/source' sütunlarını bekler
        dataset_kwargs={"skip_prepare_dataset": True},  # TRL'nin otomatik VLM hazırlama/tokenizasyon adımını atla
        # ENABLE_EMBED_LORA=True + ensure_weight_tying (bkz. lora_setup.py) `lm_head`'i
        # de LoRA adaptörüne dahil ediyor (embed_tokens ile PAYLAŞILAN ağırlık tutarlı
        # kalsın diye -- bkz. Risk 6 tartışması). TRL'nin varsayılan "chunked_nll" kayıp
        # modu, LoRA'lı bir lm_head ile ÇALIŞMIYOR (ValueError fırlatıyor); standart
        # "nll" moduna geçiyoruz -- doğruluk aynı, yalnızca bellek-verimli chunked
        # implementasyonu kullanılmıyor (küçük veri/model ölçeğimizde önemsiz bir fark).
        # NOT: ENABLE_WEIGHTED_LOSS/ENABLE_ONLINE_SELF_DISTILLATION açıkken bu alan hiç
        # devreye girmez -- WeightedDistillationTrainer kendi compute_loss'unu kullanır
        # (bkz. training/distillation_trainer.py), ama trl'nin loss_type doğrulaması
        # LoRA'lı lm_head ile hâlâ ÇALIŞMADIĞINDAN burada "nll" olarak bırakılması gerekir.
        loss_type="nll",
        # NOT: max_length/max_seq_length BİLEREK burada YOK. skip_prepare_dataset=True
        # olduğu için TRL kendi tokenizasyon/packing adımını hiç çalıştırmıyor; kesme
        # (truncation) zaten training/collate.py içindeki Qwen25VLDataCollator tarafından
        # config.MAX_SEQ_LENGTH ile uygulanıyor. Bu alanı burada TEKRAR belirtmek, trl
        # sürümleri arasında değişen alan adlarına (max_length/max_seq_length) bağımlılık
        # yaratıp gereksiz kırılganlık ekler.
    )

    callback = TestABRegressionCallback.from_baseline_report(processor, baseline_tag="baseline")
    callbacks = [callback]
    if config.LORA_DRIFT_LOG_EVERY_N_STEPS > 0:
        callbacks.append(LoRADriftTrackingCallback())

    trainer_cls = SFTTrainer
    trainer_kwargs = {}
    if config.ENABLE_WEIGHTED_LOSS or config.ENABLE_ONLINE_SELF_DISTILLATION:
        from training.distillation_trainer import WeightedDistillationTrainer  # noqa: E402

        trainer_cls = WeightedDistillationTrainer
        teacher_model = None
        if config.ENABLE_ONLINE_SELF_DISTILLATION:
            print(
                "[train_sft] Online self-distillation açık; frozen teacher model (ince "
                "ayardan ÖNCEKİ taban model) ayrıca yükleniyor -- bu ikinci bir ~3B model "
                "belleğe yüklendiği için VRAM kullanımını artırır."
            )
            teacher_model, _ = load_base_model_and_processor(load_in_4bit=config.LOAD_IN_4BIT)
        trainer_kwargs["teacher_model"] = teacher_model

    trainer = trainer_cls(
        model=model,
        args=sft_config,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        data_collator=collator,
        processing_class=processor,
        callbacks=callbacks,
        **trainer_kwargs,
    )

    print("[train_sft] Eğitim başlıyor...")
    trainer.train()

    final_dir = config.CHECKPOINT_DIR / "final_adapter"
    trainer.model.save_pretrained(str(final_dir))
    processor.save_pretrained(str(final_dir))
    print(f"[train_sft] Eğitim tamamlandı. LoRA adaptörü kaydedildi: {final_dir}")
    print(
        f"[train_sft] Pareto-kısıtlı en iyi checkpoint (varsa): "
        f"{config.CHECKPOINT_DIR / 'best_pareto_adapter'} "
        f"(bkz. {config.EVAL_RUN_OUTPUT_DIR / 'pareto_summary.json'})"
    )


if __name__ == "__main__":
    main()
