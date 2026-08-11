"""
callbacks.py
============
Her epoch sonunda Test Seti A ve Test Seti B'yi otomatik değerlendiren ve Test Seti
A'da (regresyon/katastrofik unutma göstergesi) belirgin bir kötüleşme tespit edilirse
eğitimi DURDURAN bir `transformers.TrainerCallback`.

YARI-OTOMATİK tasarım (kullanıcının orijinal isteğiyle tutarlı: "düşüş tespit edilirse
eğitim durdurulup learning rate düşürülecek ve/veya karışım oranı değiştirilecek"):
bu callback, regresyon tespit ettiğinde eğitimi durdurur VE net bir "sonraki adım"
raporu (outputs/regression_report.json) yazar; ama learning rate'i veya karışım
oranını KENDİSİ OTOMATİK OLARAK DEĞİŞTİRMEZ. Bunun yerine kullanıcı raporu okur,
configs/config.py içinde LEARNING_RATE ve/veya MIXTURE_* değerlerini elle günceller
ve training/train_sft.py'yi tekrar çalıştırır (genelde son checkpoint'ten devam
edilebilir). Tam otomatik bir yeniden-başlatma döngüsü BİLEREK kurulmamıştır: hangi
hiperparametrenin ne kadar değiştirileceği, gözetimsiz bırakılamayacak kadar önemli
bir karardır.
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

    @classmethod
    def from_baseline_report(cls, processor, baseline_tag: str = "baseline") -> "TestABRegressionCallback":
        """baseline_tag ile daha önce evaluation/evaluate.py çalıştırılarak üretilmiş
        EVAL_OUTPUT_DIR/{baseline_tag}.json dosyasından Test A kompozit skorunu okur."""
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

        was_training = model.training
        model.eval()
        result = evaluate_all(model, self.processor, tag=tag)
        if was_training:
            model.train()

        current = result["test_a"]["composite_score"]
        test_b_cer = result["test_b"]["cer"]
        self.history.append({"epoch": epoch_int, "test_a_composite": current, "test_b_cer": test_b_cer})

        relative_change = (current - self.baseline_test_a_composite) / max(
            self.baseline_test_a_composite, 1e-8
        )
        print(
            f"[callback] Epoch {epoch_int}: Test A composite={current:.4f} "
            f"(baseline={self.baseline_test_a_composite:.4f}, göreli değişim={relative_change:+.2%}), "
            f"Test B CER={test_b_cer:.4f}"
        )

        relative_drop = -relative_change  # pozitifse kötüleşme (düşüş) var demektir
        if relative_drop > config.TEST_A_REGRESSION_RELATIVE_THRESHOLD:
            control.should_training_stop = True
            self._write_regression_report(epoch_int, current, relative_drop)
            print(
                "\n[callback] !! REGRESYON TESPİT EDİLDİ — eğitim durduruluyor. "
                f"Ayrıntılar: {config.EVAL_OUTPUT_DIR / 'regression_report.json'}\n"
            )

        return control

    def _write_regression_report(self, epoch: int, current_score: float, relative_drop: float) -> None:
        report = {
            "timestamp_utc": datetime.now(timezone.utc).isoformat(),
            "stopped_at_epoch": epoch,
            "baseline_test_a_composite": self.baseline_test_a_composite,
            "current_test_a_composite": current_score,
            "relative_drop": relative_drop,
            "threshold": config.TEST_A_REGRESSION_RELATIVE_THRESHOLD,
            "history": self.history,
            "onerilen_sonraki_adimlar": [
                f"configs/config.py icinde LEARNING_RATE degerini dusurun (su an: {config.LEARNING_RATE}).",
                "configs/config.py icinde MIXTURE_REPLAY oranini artirmayi (ve/veya MIXTURE_HANDWRITING'i "
                "azaltmayi) degerlendirin -- model cok agresif adapte oluyor olabilir.",
                "data/build_chat_dataset.py ve training/train_sft.py'yi yeniden calistirin; "
                "TrainingArguments(resume_from_checkpoint=...) ile son checkpoint'ten devam edebilirsiniz.",
            ],
        }
        out_path = config.EVAL_OUTPUT_DIR / "regression_report.json"
        out_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
