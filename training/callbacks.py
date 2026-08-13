"""
callbacks.py
============
Her epoch sonunda Test Seti A ve Test Seti B'yi otomatik değerlendiren, Test Seti
A'da (regresyon/katastrofik unutma göstergesi) belirgin bir kötüleşme tespit edilirse
eğitimi DURDURAN, VE aynı zamanda "kabul edilebilir forgetting sınırı içinde Türkçe
OCR'ı en yükseğe çıkaran" checkpoint'i (Pareto-kısıtlı seçim) otomatik olarak ayrı bir
klasöre kaydeden bir `transformers.TrainerCallback`.

PARETO-KISITLI CHECKPOINT SEÇİMİ: proje hedefi "yalnızca en yüksek Türkçe OCR skorunu
veren model" değil, "Test A forgetting'i epsilon (config.TEST_A_REGRESSION_RELATIVE_THRESHOLD)
sınırını AŞMAMA koşuluyla Türkçe OCR performansını (Test B CER'i minimize ederek) MAKSİMİZE
eden" modeldir. Bu yüzden her epoch sonunda: (a) o epoch bu koşulu sağlıyorsa VE o ana kadarki
en düşük Test B CER'e sahipse, LoRA adaptörü `CHECKPOINT_DIR/best_pareto_adapter`'a kaydedilir
(bir SONRAKİ uygun epoch bunun üzerine yazar); (b) training sonunda `pareto_summary.json` bu
seçimi ve TÜM epoch geçmişini özetler -- HİÇBİR epoch koşulu sağlamasa bile (bu durumda
"uyari" alanı sonraki adımları önerir).

YARI-OTOMATİK tasarım (regresyon tespiti sert-durdurma için hâlâ geçerli): sert eşik
(TEST_A_REGRESSION_RELATIVE_THRESHOLD) aşılırsa eğitim DURUR ve regression_report.json
yazılır; ama learning rate'i veya karışım oranını KENDİSİ OTOMATİK OLARAK DEĞİŞTİRMEZ.
Ablation ayarları artık configs/config.py içinde ortam değişkenleriyle override edilebildiği
için (bkz. config.py: _env_* yardımcıları), önerilen sonraki adımlar da QWEN_OCR_* ortam
değişkenlerine referans verir.
"""

import json
import sys
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from configs import config  # noqa: E402
from evaluation.evaluate import evaluate_all  # noqa: E402

from transformers import TrainerCallback  # type: ignore


class TestABRegressionCallback(TrainerCallback):
    def __init__(self, processor, baseline_test_a_composite: float):
        self.processor = processor
        self.baseline_test_a_composite = baseline_test_a_composite
        self.history: list[dict] = []
        # Kabul edilebilir forgetting sınırı içinde kalan, o ana kadarki en düşük Test B
        # CER'ine sahip epoch kaydı (bkz. modül docstring'i: Pareto-kısıtlı seçim). None ise
        # HENÜZ hiçbir epoch koşulu sağlamadı demektir.
        self.best_epoch: dict | None = None

    @classmethod
    def from_baseline_report(cls, processor, baseline_tag: str = "baseline") -> "TestABRegressionCallback":
        """baseline_tag ile daha önce evaluation/evaluate.py çalıştırılarak üretilmiş
        EVAL_OUTPUT_DIR/{baseline_tag}.json dosyasından Test A kompozit skorunu okur.
        NOT: baseline PAYLAŞILANDIR (RUN_NAME'e göre AYRILMAZ) -- bkz. configs/config.py:
        EVAL_OUTPUT_DIR vs EVAL_RUN_OUTPUT_DIR notu."""
        baseline_path = config.EVAL_OUTPUT_DIR / f"{baseline_tag}.json"
        if not baseline_path.exists():
            raise FileNotFoundError(
                f"{baseline_path} bulunamadı. Eğitimden ÖNCE, taban model üzerinde "
                "`python evaluation/evaluate.py --tag baseline` çalıştırılmalı."
            )
        baseline_report = json.loads(baseline_path.read_text(encoding="utf-8"))
        return cls(processor=processor, baseline_test_a_composite=baseline_report["test_a"]["composite_score"])

    def on_epoch_end(self, args, state, control, **kwargs):
        model = kwargs["model"]
        epoch_int = int(round(state.epoch)) if state.epoch is not None else len(self.history) + 1
        tag = f"epoch_{epoch_int}"

        # `evaluate.py`'nin checkpoint mekanizması, AYNI etiket (ör. "epoch_1") + AYNI
        # referans metinlerle karşılaşırsa eski hipotezleri yeniden kullanır -- bu,
        # evaluate.py'nin TEK BİR çalıştırmasının kesintiye uğrayıp devam etmesi için
        # doğrudur. AMA `train_sft.py` sıfırdan yeniden çalıştırılırsa (ör. önceki
        # deneme regresyon nedeniyle durdu, yeniden eğitiliyor), YENİ eğitilen model
        # ile ESKİ denemenin modeli TAMAMEN FARKLIDIR -- ikisi de "epoch_1" etiketini
        # üretir. Checkpoint'i temizlemezsek, bu yeni (gerçekte farklı) modelin
        # performansı yerine ESKİ modelin (yanlışlıkla) hâlâ geçerli sayılan
        # hipotezleri kullanılıp yanıltıcı bir sonuç üretilebilir. Bu yüzden her
        # epoch değerlendirmesi başlamadan önce, bu etikete ait checkpoint dosyaları
        # SİLİNİR -- her çağrı, altındaki model gerçekten farklı olabileceğinden,
        # her zaman TAZE üretim yapmalıdır.
        for split_name in ("test_a", "test_b"):
            checkpoint_path = config.EVAL_RUN_OUTPUT_DIR / "checkpoints" / f"{tag}_{split_name}.jsonl"
            checkpoint_path.unlink(missing_ok=True)

        was_training = model.training
        model.eval()
        result = evaluate_all(model, self.processor, tag=tag, output_dir=config.EVAL_RUN_OUTPUT_DIR)
        if was_training:
            model.train()

        current = result["test_a"]["composite_score"]
        test_b = result["test_b"]

        relative_change = (current - self.baseline_test_a_composite) / max(
            self.baseline_test_a_composite, 1e-8
        )
        relative_drop = -relative_change  # pozitifse kötüleşme (düşüş) var demektir

        record = {
            "epoch": epoch_int,
            "global_step": state.global_step,
            "test_a_composite": current,
            "relative_drop": relative_drop,
            "test_b_cer": test_b["cer"],
            "test_b_wer": test_b["wer"],
            "test_b_exact_match": test_b["exact_match"],
            "test_b_turkish_char_accuracy": test_b["turkish_char_accuracy"],
        }
        self.history.append(record)

        print(
            f"[callback] Epoch {epoch_int}: Test A composite={current:.4f} "
            f"(baseline={self.baseline_test_a_composite:.4f}, göreli değişim={relative_change:+.2%}), "
            f"Test B CER={test_b['cer']:.4f} WER={test_b['wer']:.4f} "
            f"exact_match={test_b['exact_match']:.4f} turkish_char_acc={test_b['turkish_char_accuracy']:.4f}"
        )

        if relative_drop <= config.TEST_A_REGRESSION_RELATIVE_THRESHOLD and (
            self.best_epoch is None or test_b["cer"] < self.best_epoch["test_b_cer"]
        ):
            self.best_epoch = record
            best_adapter_dir = config.CHECKPOINT_DIR / "best_pareto_adapter"
            model.save_pretrained(str(best_adapter_dir))
            self.processor.save_pretrained(str(best_adapter_dir))
            print(
                f"[callback] >> Yeni Pareto-en-iyi checkpoint: epoch {epoch_int} "
                f"(Test B CER={test_b['cer']:.4f}, forgetting={relative_drop:.2%} <= "
                f"{config.TEST_A_REGRESSION_RELATIVE_THRESHOLD:.0%}) -> {best_adapter_dir}"
            )

        self._write_eval_history()

        if relative_drop > config.TEST_A_REGRESSION_RELATIVE_THRESHOLD:
            control.should_training_stop = True
            self._write_regression_report(epoch_int, current, relative_drop)
            print(
                "\n[callback] !! REGRESYON TESPİT EDİLDİ — eğitim durduruluyor. "
                f"Ayrıntılar: {config.EVAL_RUN_OUTPUT_DIR / 'regression_report.json'}\n"
            )

        return control

    def on_train_end(self, args, state, control, **kwargs):
        """Regresyon nedeniyle erken durmuş olsa da olmasa da, koşumun TAM geçmişini ve
        Pareto-seçimini her zaman yazar -- analysis/ablation_report.py bu dosyaya bağımlıdır."""
        self._write_pareto_summary()
        return control

    def _write_eval_history(self) -> None:
        out_path = config.EVAL_RUN_OUTPUT_DIR / "eval_history.json"
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(
            json.dumps(
                {
                    "run_name": config.RUN_NAME,
                    "baseline_test_a_composite": self.baseline_test_a_composite,
                    "threshold": config.TEST_A_REGRESSION_RELATIVE_THRESHOLD,
                    "history": self.history,
                },
                ensure_ascii=False,
                indent=2,
            ),
            encoding="utf-8",
        )

    def _write_pareto_summary(self) -> None:
        summary = {
            "run_name": config.RUN_NAME,
            "threshold": config.TEST_A_REGRESSION_RELATIVE_THRESHOLD,
            "best_epoch": self.best_epoch,
            "best_adapter_path": (
                str(config.CHECKPOINT_DIR / "best_pareto_adapter") if self.best_epoch is not None else None
            ),
            "history": self.history,
        }
        if self.best_epoch is None:
            summary["uyari"] = (
                "Hiçbir epoch kabul edilebilir forgetting sınırının "
                f"(TEST_A_REGRESSION_RELATIVE_THRESHOLD={config.TEST_A_REGRESSION_RELATIVE_THRESHOLD:.0%}) "
                "İÇİNDE kalmadı; bu koşum için Pareto-uygun bir checkpoint SEÇİLEMEDİ. "
                "QWEN_OCR_LEARNING_RATE'i düşürmeyi, QWEN_OCR_LORA_LAYER_SCOPE'u daraltmayı "
                "(ör. 'last12'), QWEN_OCR_LORA_TARGET_SCOPE='attn_only' denemeyi veya "
                "QWEN_OCR_ENABLE_WEIGHTED_LOSS=1 ile QWEN_OCR_LOSS_WEIGHT_REPLAY'i artırmayı "
                "değerlendirin."
            )
        out_path = config.EVAL_RUN_OUTPUT_DIR / "pareto_summary.json"
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"[callback] Pareto özeti yazıldı: {out_path}")

    def _write_regression_report(self, epoch: int, current_score: float, relative_drop: float) -> None:
        report = {
            "timestamp_utc": datetime.now(timezone.utc).isoformat(),
            "run_name": config.RUN_NAME,
            "stopped_at_epoch": epoch,
            "baseline_test_a_composite": self.baseline_test_a_composite,
            "current_test_a_composite": current_score,
            "relative_drop": relative_drop,
            "threshold": config.TEST_A_REGRESSION_RELATIVE_THRESHOLD,
            "history": self.history,
            "onerilen_sonraki_adimlar": [
                f"QWEN_OCR_LEARNING_RATE degerini dusurun (su an: {config.LEARNING_RATE}).",
                f"QWEN_OCR_LORA_LAYER_SCOPE'u daraltmayi degerlendirin (su an: "
                f"{config.LORA_LAYER_SCOPE!r}) -- ust katmanlara odaklanmak drift'i sinirlayabilir.",
                f"QWEN_OCR_LORA_TARGET_SCOPE='attn_only' deneyin (su an: {config.LORA_TARGET_SCOPE!r}).",
                "MIXTURE_REPLAY oranini artirmadan ONCE, QWEN_OCR_ENABLE_WEIGHTED_LOSS=1 ile "
                "QWEN_OCR_LOSS_WEIGHT_REPLAY'i artirmayi deneyin (configs/config.py bolum 9b).",
            ],
        }
        out_path = config.EVAL_RUN_OUTPUT_DIR / "regression_report.json"
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
