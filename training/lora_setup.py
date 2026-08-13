"""
lora_setup.py
=============
Bu dosya iki şeyi yapar:

  1) Temel modeli (Qwen2.5-VL-3B-Instruct) ve işlemciyi (processor) donanıma uygun
     ayarlarla (dtype, attention implementasyonu, opsiyonel 4-bit) yükleyen ORTAK
     yardımcı fonksiyonu sağlar (`load_base_model_and_processor`). Bu fonksiyon
     replay_generation.py, evaluate.py ve train_sft.py tarafından da kullanılır —
     böylece model yükleme mantığı TEK bir yerde yaşar.

  2) LoRA'yı modele uygular (`apply_lora`). Hedef modülleri BULMAK için modelin
     mimarisine göre sabit modül yolları (ör. "model.language_model.layers.0...")
     HARDCODE EDİLMEZ; bunun yerine `model.named_modules()` üzerinde gezilip regex/
     isinstance kontrolleriyle DİNAMİK olarak keşfedilir ve bulunanlar ekrana
     yazdırılır. Bu, transformers kütüphanesi sürümden sürüme modül isimlerini
     değiştirse bile (ör. "model.model.layers" <-> "model.language_model.layers")
     kodun kırılmadan çalışmasını sağlar.

     ÖNEMLİ AYRIM: Qwen2.5-VL'nin görsel encoder'ı da (yanlışlıkla LoRA hedefi olabilecek)
     "mlp.gate_proj", "mlp.up_proj", "mlp.down_proj" isimli katmanlar İÇERİR (bkz.
     huggingface/peft #2615). Bu yüzden LLM hedeflerini seçerken tam modül yolunda
     "visual" GEÇMEYEN katmanlar seçilir; görsel encoder'a dokunulmaz (varsayılan).
"""

import inspect
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from configs import config  # noqa: E402

import torch  # type: ignore
from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration  # type: ignore
from peft import LoraConfig, PeftModel, get_peft_model  # type: ignore


# ---------------------------------------------------------------------------
# 1) Donanıma göre dtype ve attention implementasyonu seçimi
# ---------------------------------------------------------------------------
def select_dtype_and_attn_impl() -> tuple[torch.dtype, str]:
    """T4 (Turing, cc 7.5) ile A100 (Ampere, cc 8.0+) arasındaki farkı otomatik yönetir.

    - T4: bfloat16'ı native desteklemez (yavaş/hatalı davranabilir) ve flash-attention-2
      genelde T4'te çalışmaz/desteklenmez -> fp16 + sdpa kullanılır.
    - A100 ve sonrası: bf16 nativedir, flash-attention-2 kuruluysa kullanılır, kurulu
      değilse yine sdpa'ya (güvenli, hatasız) düşülür.
    - GPU yoksa (CPU): fp32 + sdpa (yalnızca kod incelemesi/hata ayıklama amaçlı; gerçek
      eğitim GPU gerektirir).
    """
    if not torch.cuda.is_available():
        return torch.float32, "sdpa"

    major, _minor = torch.cuda.get_device_capability()
    if major < 8:  # T4 ve öncesi mimariler
        return torch.float16, "sdpa"

    # Ampere (A100) ve sonrası: flash-attention-2 paketi kuruluysa kullan, değilse sdpa.
    try:
        import flash_attn  # noqa: F401

        return torch.bfloat16, "flash_attention_2"
    except ImportError:
        return torch.bfloat16, "sdpa"


def load_base_model_and_processor(load_in_4bit: bool = False):
    """Temel modeli ve processor'ı yükler. `load_in_4bit=True` (pilot/T4 için) iken
    bitsandbytes ile 4-bit nicemleme uygulanır; bu durumda döndürülen model
    `prepare_model_for_kbit_training` ile LoRA eğitimine hazır hale getirilir.

    bitsandbytes 4-bit nicemleme yalnızca CUDA GPU'da çalışır. GPU yoksa (ör. Colab
    GPU kotası dolduğunda "GPU'suz devam et" seçilirse) `load_in_4bit=True` isteği
    SESSİZCE yoksayılır ve tam hassasiyete (fp32) düşülür — aksi halde bitsandbytes
    çökerdi. Not: CPU'da bu 3B parametreli VLM'nin görsel-dil üretimi (generation)
    PRATİK DEĞİLDİR (tek görsel için dakikalar sürebilir); bu yalnızca GPU yokken
    kodun ÇÖKMEDEN çalışmasını garanti eder, hızlı bir alternatif sunmaz.
    """
    if load_in_4bit and not torch.cuda.is_available():
        print(
            "[lora_setup] !! GPU bulunamadı; 4-bit nicemleme (bitsandbytes) yalnızca "
            "CUDA'da çalışır. fp32'ye düşülüyor. UYARI: CPU'da bu model ile üretim "
            "(generation) ÇOK YAVAŞTIR, pratik bir seçenek değildir."
        )
        load_in_4bit = False

    dtype, attn_impl = select_dtype_and_attn_impl()
    print(f"[lora_setup] Seçilen dtype={dtype}, attn_implementation={attn_impl}, 4bit={load_in_4bit}")

    processor = AutoProcessor.from_pretrained(
        config.MODEL_ID,
        min_pixels=config.MIN_PIXELS,
        max_pixels=config.MAX_PIXELS,
    )

    model_kwargs = dict(
        dtype=dtype,
        attn_implementation=attn_impl,
        device_map="auto",
    )

    if load_in_4bit:
        from transformers import BitsAndBytesConfig  # type: ignore

        model_kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=dtype,
            bnb_4bit_use_double_quant=True,
        )

    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(config.MODEL_ID, **model_kwargs)

    if load_in_4bit:
        from peft import prepare_model_for_kbit_training  # type: ignore

        model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=True)

    return model, processor


# ---------------------------------------------------------------------------
# 2) Hedef modüllerin dinamik keşfi
# ---------------------------------------------------------------------------
_ATTN_LEAF_SUFFIXES = ("q_proj", "k_proj", "v_proj", "o_proj")
_MLP_LEAF_SUFFIXES = ("gate_proj", "up_proj", "down_proj")
_TARGET_LEAF_SUFFIXES = _ATTN_LEAF_SUFFIXES + _MLP_LEAF_SUFFIXES

_LAYER_INDEX_PATTERN = re.compile(r"\.layers\.(\d+)\.")


def _active_target_suffixes() -> tuple[str, ...]:
    """config.LORA_TARGET_SCOPE'a göre hangi projeksiyon türlerinin LoRA hedefi
    olacağını belirler ("attn_only" -> yalnızca q,k,v,o; aksi halde attn+mlp)."""
    if config.LORA_TARGET_SCOPE == "attn_only":
        return _ATTN_LEAF_SUFFIXES
    if config.LORA_TARGET_SCOPE != "attn_mlp":
        print(
            f"[lora_setup] !! Bilinmeyen LORA_TARGET_SCOPE={config.LORA_TARGET_SCOPE!r}; "
            "'attn_mlp' (varsayılan) kullanılıyor."
        )
    return _TARGET_LEAF_SUFFIXES


def _parse_layer_index(name: str) -> int | None:
    """Modül yolundan LLM decoder katman indeksini çıkarır (ör.
    'model.language_model.layers.23.self_attn.q_proj' -> 23). Decoder katmanı
    içermeyen (merger/embed gibi) yollarda None döner."""
    match = _LAYER_INDEX_PATTERN.search(name)
    return int(match.group(1)) if match else None


def _resolve_layer_scope(scope: str, num_layers: int) -> set[int]:
    """config.LORA_LAYER_SCOPE ("all" veya "lastN") ile keşfedilen toplam katman
    sayısından, LoRA'nın uygulanacağı katman İNDEKSLERİ kümesini üretir."""
    if num_layers <= 0:
        return set()
    if scope == "all":
        return set(range(num_layers))
    match = re.fullmatch(r"last(\d+)", scope)
    if not match:
        raise ValueError(
            f"Geçersiz LORA_LAYER_SCOPE: {scope!r} (beklenen: 'all' veya 'lastN', ör. 'last12')."
        )
    n = min(int(match.group(1)), num_layers)
    return set(range(num_layers - n, num_layers))


def discover_target_modules(model) -> dict[str, list[str]]:
    """Modelin tüm alt modüllerini gezip dört kategoriye ayırır:
      - "llm": LLM decoder'ın attention+MLP projeksiyon katmanları (görsel encoder HARİÇ)
      - "merger": görsel-özellik projektörü (merger) içindeki Linear katmanlar
      - "embed": metin embed_tokens katmanı (görsel encoder'da eşdeğeri yoktur)
      - "vision": (yalnızca ENABLE_VISION_LORA=True iken doldurulur) son N tam-attention
        vision bloğundaki Linear katmanlar

    "llm" kategorisi İKİ ablation bayrağına göre ayrıca daraltılır:
      - config.LORA_TARGET_SCOPE: "attn_only" iken yalnızca q/k/v/o projeksiyonları
        (MLP'ye LoRA UYGULANMAZ); "attn_mlp" (varsayılan) iken hepsi.
      - config.LORA_LAYER_SCOPE: "lastN" iken yalnızca decoder'ın SON N katmanı
        hedeflenir (alt katmanlar tamamen dondurulmuş kalır, LoRA adaptörü ALMAZLAR);
        "all" (varsayılan) iken tüm katmanlar. Toplam katman sayısı HARDCODE EDİLMEZ,
        keşfedilen modül isimlerindeki en yüksek katman indeksinden (+1) TÜRETİLİR --
        böylece bu dosyanın geri kalanındaki "sabit yol yok" ilkesiyle tutarlı kalır.

    Sonuç, tam nitelikli (dotted) modül isimlerinden oluşan listelerdir; bu isimler
    doğrudan LoraConfig(target_modules=...) içinde TAM EŞLEŞME olarak kullanılacağı için
    kısa isim çakışması (ör. görsel encoder'daki "mlp.gate_proj") riski oluşmaz.
    """
    active_suffixes = _active_target_suffixes()

    # Katman sayısını TÜM projeksiyon türlerine (suffix daraltmasından ÖNCE) bakarak
    # belirliyoruz; aksi halde "attn_only" iken MLP-only bir modelde (varsayımsal) katman
    # sayısı yanlış hesaplanabilirdi. Qwen2.5-VL'de her katman hem attn hem mlp içerdiği
    # için pratikte fark etmez, ama bu daha sağlam bir varsayım.
    all_layer_indices = [
        idx
        for name, module in model.named_modules()
        if isinstance(module, torch.nn.Linear)
        and name.endswith(_TARGET_LEAF_SUFFIXES)
        and "visual" not in name
        and (idx := _parse_layer_index(name)) is not None
    ]
    num_layers = (max(all_layer_indices) + 1) if all_layer_indices else 0
    allowed_layer_indices = _resolve_layer_scope(config.LORA_LAYER_SCOPE, num_layers)

    llm_targets: list[str] = []
    merger_targets: list[str] = []
    embed_targets: list[str] = []

    for name, module in model.named_modules():
        is_linear_like = isinstance(module, (torch.nn.Linear,))
        is_embedding = isinstance(module, torch.nn.Embedding)

        if is_linear_like and name.endswith(active_suffixes) and "visual" not in name:
            layer_idx = _parse_layer_index(name)
            if layer_idx is None or layer_idx in allowed_layer_indices:
                llm_targets.append(name)
        elif is_linear_like and "merger" in name:
            merger_targets.append(name)
        elif is_embedding and name.endswith("embed_tokens") and "visual" not in name:
            embed_targets.append(name)

    vision_targets: list[str] = []
    if config.ENABLE_VISION_LORA:
        vision_targets = _discover_vision_lora_targets(model)

    print(
        f"[lora_setup] LoRA layer scope='{config.LORA_LAYER_SCOPE}' -> {num_layers} katmandan "
        f"{len(allowed_layer_indices)} tanesi hedefleniyor: {sorted(allowed_layer_indices)}"
    )
    print(f"[lora_setup] LoRA target scope='{config.LORA_TARGET_SCOPE}' -> aktif suffix'ler: {active_suffixes}")
    print(f"[lora_setup] Keşfedilen LLM hedef modül sayısı: {len(llm_targets)}")
    if llm_targets:
        print(f"             örnekler: {llm_targets[:3]} ...")
    print(f"[lora_setup] Keşfedilen merger hedef modül sayısı: {len(merger_targets)} -> {merger_targets}")
    if config.ENABLE_EMBED_LORA:
        print(f"[lora_setup] Keşfedilen embed hedef modül sayısı: {len(embed_targets)} -> {embed_targets}")
    if config.ENABLE_VISION_LORA:
        print(f"[lora_setup] Keşfedilen vision hedef modül sayısı: {len(vision_targets)}")

    if not llm_targets:
        raise RuntimeError(
            "LLM hedef modülleri bulunamadı! transformers sürümü beklenenden farklı bir "
            "mimari kullanıyor olabilir; model.named_modules() çıktısını elle inceleyin."
        )
    if not merger_targets:
        raise RuntimeError("Merger hedef modülleri bulunamadı! Model mimarisini kontrol edin.")

    return {
        "llm": llm_targets,
        "merger": merger_targets,
        "embed": embed_targets,
        "vision": vision_targets,
    }


def _discover_vision_lora_targets(model) -> list[str]:
    """Yalnızca `fullatt_block_indexes`'te belirtilen SON N tam-attention vision
    bloğundaki Linear katmanları hedefler; pencereli (windowed) attention bloklarına
    dokunulmaz. Bu fonksiyon yalnızca config.ENABLE_VISION_LORA=True iken çağrılır."""
    vision_config = getattr(model.config, "vision_config", None)
    fullatt_indexes = list(getattr(vision_config, "fullatt_block_indexes", []) or [])
    if not fullatt_indexes:
        print("[lora_setup] !! fullatt_block_indexes bulunamadı; vision LoRA atlanıyor.")
        return []

    selected_indexes = set(fullatt_indexes[-config.VISION_LORA_LAST_N_FULLATT_BLOCKS :])
    pattern = re.compile(r"visual\.blocks\.(\d+)\.")

    targets = []
    for name, module in model.named_modules():
        if not isinstance(module, torch.nn.Linear):
            continue
        match = pattern.search(name)
        if match and int(match.group(1)) in selected_indexes:
            targets.append(name)
    return targets


# ---------------------------------------------------------------------------
# 3) LoRA konfigürasyonu ve uygulanması
# ---------------------------------------------------------------------------
def build_lora_config(model) -> LoraConfig:
    targets = discover_target_modules(model)

    all_target_modules = list(targets["llm"]) + list(targets["merger"])
    rank_pattern: dict[str, int] = {}
    alpha_pattern: dict[str, int] = {}

    # Merger için farklı (daha düşük) rank/alpha -> PEFT'in rank_pattern/alpha_pattern
    # mekanizmasıyla verilir. Anahtarlar, re.escape() ile kaçışlanmış TAM modül isimleridir;
    # böylece yalnızca o belirli modül eşleşir, başka hiçbir katmanı etkilemez.
    for name in targets["merger"]:
        rank_pattern[re.escape(name)] = config.MERGER_LORA_R
        alpha_pattern[re.escape(name)] = config.MERGER_LORA_ALPHA

    if config.ENABLE_EMBED_LORA and targets["embed"]:
        all_target_modules += targets["embed"]
        for name in targets["embed"]:
            rank_pattern[re.escape(name)] = config.EMBED_LORA_R
            alpha_pattern[re.escape(name)] = config.EMBED_LORA_ALPHA

    if config.ENABLE_VISION_LORA and targets["vision"]:
        all_target_modules += targets["vision"]
        for name in targets["vision"]:
            rank_pattern[re.escape(name)] = config.VISION_LORA_R
            alpha_pattern[re.escape(name)] = config.VISION_LORA_ALPHA

    lora_config_kwargs = dict(
        r=config.LORA_R,
        lora_alpha=config.LORA_ALPHA,
        lora_dropout=config.LORA_DROPOUT,
        target_modules=all_target_modules,
        rank_pattern=rank_pattern,
        alpha_pattern=alpha_pattern,
        bias="none",
        task_type="CAUSAL_LM",
    )

    # Qwen2.5-VL, `embed_tokens` (girdi gömme matrisi) ile `lm_head` (çıktı projeksiyonu)
    # AYNI ağırlık tensörünü PAYLAŞIR (tie_word_embeddings=True). ENABLE_EMBED_LORA=True
    # iken embed_tokens'a LoRA uygularsak, bu LoRA farkının (delta) çıktı tahminlerine de
    # (lm_head üzerinden) doğru şekilde yansıması için peft'in `ensure_weight_tying`
    # ayarının açık olması gerekir -- aksi halde LoRA yalnızca GİRDİ tarafında etkili
    # olup ÇIKTI (üretim) olasılıklarını tutarsız şekilde etkileyebilir (bkz. peft
    # #2777). Yalnızca embed_tokens hedeflendiğinde VE kurulu peft sürümü bu parametreyi
    # destekliyorsa eklenir (eski peft sürümleriyle geriye dönük uyumluluk için).
    if config.ENABLE_EMBED_LORA and targets["embed"]:
        if "ensure_weight_tying" in inspect.signature(LoraConfig).parameters:
            lora_config_kwargs["ensure_weight_tying"] = True
        else:
            print(
                "[lora_setup] !! Kurulu peft sürümü 'ensure_weight_tying' desteklemiyor; "
                "embed_tokens LoRA'sı ile lm_head arasındaki bağ tutarlılığı garanti edilemez."
            )

    return LoraConfig(**lora_config_kwargs)


def apply_lora(model) -> PeftModel:
    """Modele LoRA'yı uygular ve eğitilebilir parametre özetini yazdırır."""
    lora_config = build_lora_config(model)
    peft_model = get_peft_model(model, lora_config)
    peft_model.print_trainable_parameters()
    return peft_model


# ---------------------------------------------------------------------------
# Bağımsız çalıştırma: modeli yükleyip LoRA hedef modül keşfini doğrulamak için
# (`python training/lora_setup.py`) — EDA/doğrulama amaçlı.
# ---------------------------------------------------------------------------
def main() -> None:
    model, _processor = load_base_model_and_processor(load_in_4bit=config.LOAD_IN_4BIT)
    peft_model = apply_lora(model)
    print("\n[lora_setup] LoRA başarıyla uygulandı. Model eğitime hazır.")
    del peft_model


if __name__ == "__main__":
    main()
