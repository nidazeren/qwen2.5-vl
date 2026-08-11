"""
collate.py
==========
SFTTrainer'a verilecek ÖZEL (custom) data collator. Ham örnekleri
({"image","instruction","answer","source"} — bkz. data/build_chat_dataset.py çıktısı)
alıp modelin `forward()`'una verilecek batch'e (input_ids, attention_mask, pixel_values,
image_grid_thw, labels) dönüştürür.

KAYIP MASKELEME (loss masking) mantığı:
  Yalnızca ASİSTAN CEVABI (ground-truth OCR metni / replay hedef metni) üzerinden loss
  hesaplanmalıdır; kullanıcı talimatı, görsel token'lar ve chat şablonu iskelet
  token'ları (<|im_start|>user ... gibi) loss'a KATILMAMALIDIR. Bunu sağlamak için:

    1) Her örnek için SADECE talimat kısmını (add_generation_prompt=True ile, yani
       "...<|im_start|>assistant\n" ile biten metni) o örneğin GERÇEK görseliyle birlikte
       processor'dan geçiririz -> bu bize "prompt_len" (görsel token'lar dahil, o örneğe
       özgü talimat uzunluğu) verir. Görsel token sayısı görüntüye göre DEĞİŞTİĞİ için bu
       uzunluk her örnekte farklı olabilir; bu yüzden düz metin tokenizer'ı ile DEĞİL,
       AYNI processor + AYNI görselle hesaplanır (aksi halde <|image_pad|> yer tutucusu
       gerçek token sayısına genişletilmeden sayılır ve maskeleme sınırı KAYAR).
    2) Tüm batch (talimat+cevap birlikte) TEK bir processor çağrısıyla (padding=True)
       işlenir -> input_ids, pixel_values, image_grid_thw elde edilir.
    3) labels = input_ids.clone(); her satırda [0:prompt_len] aralığı VE padding
       pozisyonları -100 (PyTorch CrossEntropyLoss'un yok saydığı index) ile maskelenir.

Bu desen, sağ-taraf (right) padding varsayar (`processor.tokenizer.padding_side="right"`);
böylece [0:prompt_len] maskesi, padding'den ETKİLENMEZ (padding her zaman sondadır).
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from configs import config  # noqa: E402

import torch  # type: ignore
from qwen_vl_utils import process_vision_info  # type: ignore

IGNORE_INDEX = -100


class Qwen25VLDataCollator:
    """SFTTrainer(data_collator=Qwen25VLDataCollator(processor)) şeklinde kullanılır."""

    def __init__(self, processor, max_seq_length: int = config.MAX_SEQ_LENGTH):
        self.processor = processor
        self.max_seq_length = max_seq_length
        # Eğitimde sağ-taraf padding kullanıyoruz (bkz. modül docstring'i); Qwen
        # tokenizer'ının varsayılanı üretim/inference için "left" olabileceğinden burada
        # AÇIKÇA "right" olarak sabitliyoruz.
        self.processor.tokenizer.padding_side = "right"

    def _build_messages(self, image, instruction: str, answer: str | None):
        user_msg = {
            "role": "user",
            "content": [
                {"type": "image", "image": image},
                {"type": "text", "text": instruction},
            ],
        }
        if answer is None:
            return [user_msg]
        assistant_msg = {"role": "assistant", "content": [{"type": "text", "text": answer}]}
        return [user_msg, assistant_msg]

    def _prompt_length_for_example(self, image, instruction: str) -> int:
        """Yalnızca talimat (+ görsel) kısmının, GERÇEK görsel token genişlemesiyle
        birlikte kaç token tuttuğunu hesaplar (bkz. modül docstring'i, adım 1)."""
        prompt_messages = self._build_messages(image, instruction, answer=None)
        prompt_text = self.processor.apply_chat_template(
            prompt_messages, tokenize=False, add_generation_prompt=True
        )
        image_inputs, _ = process_vision_info(prompt_messages)
        prompt_enc = self.processor(text=[prompt_text], images=image_inputs, return_tensors="pt")
        return int(prompt_enc["input_ids"].shape[1])

    def __call__(self, examples: list[dict]) -> dict:
        full_texts = []
        all_image_inputs = []
        prompt_lengths = []

        for ex in examples:
            image, instruction, answer = ex["image"], ex["instruction"], ex["answer"]

            full_messages = self._build_messages(image, instruction, answer)
            full_text = self.processor.apply_chat_template(
                full_messages, tokenize=False, add_generation_prompt=False
            )
            full_texts.append(full_text)

            image_inputs, _ = process_vision_info(full_messages)
            all_image_inputs.extend(image_inputs)

            prompt_lengths.append(self._prompt_length_for_example(image, instruction))

        batch = self.processor(
            text=full_texts,
            images=all_image_inputs,
            padding=True,
            truncation=True,
            max_length=self.max_seq_length,
            return_tensors="pt",
        )

        input_ids = batch["input_ids"]
        attention_mask = batch["attention_mask"]
        labels = input_ids.clone()

        # Padding pozisyonlarını maskele.
        labels[attention_mask == 0] = IGNORE_INDEX

        # Her satırda talimat (+görsel) kısmını maskele; cevap kısmı DOKUNULMADAN kalır.
        seq_len = input_ids.shape[1]
        for i, prompt_len in enumerate(prompt_lengths):
            cutoff = min(prompt_len, seq_len)  # max_length'te kesilmiş örnekler için güvenlik
            labels[i, :cutoff] = IGNORE_INDEX

        batch["labels"] = labels
        return batch
