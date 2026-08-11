"""
inference_utils.py
===================
Modelden metin üretmek (generate) için ORTAK yardımcı fonksiyon. Hem
data/replay_generation.py (self-distillation pseudo-label üretimi) hem de
evaluation/evaluate.py (Test Seti A/B skorlama) AYNI üretim mantığını kullanır;
bu yüzden burada tek bir yerde tutulur (DRY).

Qwen2.5-VL için resmi/önerilen çıkarım deseni `qwen_vl_utils.process_vision_info`
yardımcı fonksiyonunu kullanır (görselleri chat mesajlarından ayıklayıp processor'ın
beklediği formata getirir).
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch  # type: ignore
from qwen_vl_utils import process_vision_info  # type: ignore


@torch.inference_mode()
def generate_batch(
    model,
    processor,
    images: list,
    prompts: list[str],
    max_new_tokens: int = 512,
) -> list[str]:
    """Her (image, prompt) çifti için modelden bir metin cevabı üretir.

    `images` elemanları PIL.Image veya dosya yolu olabilir. Basitlik ve bellek
    öngörülebilirliği için tek tek (batch_size=1) üretim yapılır — self-distillation
    ve değerlendirme adımları zaten sık çalıştırılan, performans-kritik yollar değildir;
    bu yüzden karmaşık dinamik-batching yerine anlaşılır/hatasız tek-tek işleme tercih edilmiştir.
    """
    assert len(images) == len(prompts), "images ve prompts aynı uzunlukta olmalı"

    outputs: list[str] = []
    for image, prompt in zip(images, prompts):
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": image},
                    {"type": "text", "text": prompt},
                ],
            }
        ]
        chat_text = processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        image_inputs, video_inputs = process_vision_info(messages)
        inputs = processor(
            text=[chat_text],
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt",
        ).to(model.device)

        generated_ids = model.generate(
            **inputs, max_new_tokens=max_new_tokens, do_sample=False
        )
        # Girdi (prompt) token'larını üretilen dizinin başından kırpıp yalnızca YENİ
        # üretilen token'ları decode ediyoruz.
        trimmed_ids = [
            out_ids[len(in_ids):]
            for in_ids, out_ids in zip(inputs["input_ids"], generated_ids)
        ]
        decoded = processor.batch_decode(
            trimmed_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )
        outputs.append(decoded[0].strip())

    return outputs
