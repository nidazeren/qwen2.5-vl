"""
config.py
=========
Bu dosya, projedeki TÜM sabitleri ve ayarlanabilir parametreleri tek bir yerde toplar.
Amaç: hiçbir Python dosyasının içine "sihirli sayı" gömülmesin; bir değeri değiştirmek
istediğinizde (ör. LoRA rank'ı, karışım oranı, learning rate) tek yapmanız gereken bu
dosyayı düzenlemek olsun.

Diğer tüm modüller (data/, analysis/, training/, evaluation/, notebooks/) bu dosyadan
`import` eder; böylece pilot (T4) ve tam eğitim (A100) koşumları arasında sadece birkaç
bayrağı değiştirerek geçiş yapabilirsiniz (örn. PILOT_MODE=True/False).

Notebook içinden çalıştırırken repo kökünü PYTHONPATH'e eklemeniz yeterlidir:
    import sys; sys.path.insert(0, "/content/qwen2.5-vl")
    from configs import config
"""

import os
from pathlib import Path

# ---------------------------------------------------------------------------
# 0) ÇALIŞMA MODU (PILOT / TAM) — dosyanın en başında tanımlanır çünkü aşağıdaki
# YOL tanımları (bkz. bölüm 1) bu bayrağa göre dallanır.
# ---------------------------------------------------------------------------
# PILOT_MODE=True: T4 üzerinde küçük bir veri alt kümesiyle, hızlı ve ucuz bir "duman testi"
# (pipeline'ın uçtan uca hatasız çalıştığını doğrulamak) için kullanılır.
# PILOT_MODE=False: A100 üzerinde tam veri ve tam epoch sayısıyla asıl eğitim için kullanılır.
# Bu bayrak notebook'larda import EDİLMEDEN ÖNCE ortam değişkeniyle set edilebilir:
#   os.environ["QWEN_OCR_PILOT_MODE"] = "1"  # veya "0"
PILOT_MODE = os.environ.get("QWEN_OCR_PILOT_MODE", "1") == "1"

# ---------------------------------------------------------------------------
# 1) TEMEL YOLLAR
# ---------------------------------------------------------------------------
# REPO_ROOT: bu dosyanın (configs/config.py) iki üst dizini, yani repo kökü.
# Hem yerel VS Code'da hem Colab'da (repo klonlandıktan sonra) doğru çalışır.
REPO_ROOT = Path(__file__).resolve().parent.parent

# DRIVE_ROOT: Google Drive üzerinde KALICI verinin (checkpoint, log, hazırlanmış
# veri seti önbelleği, değerlendirme sonuçları) tutulacağı klasör. Colab'da
# `drive.mount('/content/drive')` yaptıktan SONRA bu yol gerçek olur.
# Yerelde (VS Code'da sadece dosya yazarken) bu yol hiç kullanılmaz/oluşturulmaz;
# sadece Colab'da import edildiğinde anlam kazanır.
# Ortam değişkeni ile override edilebilir (örn. farklı bir Drive klasörü istenirse).
DRIVE_ROOT = Path(
    os.environ.get("QWEN_OCR_DRIVE_ROOT", "/content/drive/MyDrive/qwen25vl_turkish_ocr")
)

# Ham/indirilen veri setlerinin önbelleklendiği klasör (HF datasets cache, Kaggle indirmeleri,
# replay_generation.py çıktısı dahil — replay_ocr/replay_general de birer "ham kaynak"
# olarak burada, diğer 4 kaynakla aynı şemada tutulur). Bu klasör PILOT/TAM modlar
# arasında PAYLAŞILIR: tam moda geçildiğinde prepare_datasets.py/replay_generation.py
# buradaki içeriği daha büyük hedef sayılarıyla YENİDEN üretir (üzerine yazar).
RAW_DATA_DIR = DRIVE_ROOT / "raw_data"

# MODE_TAG: pilot (T4, küçük veri) ve tam (A100, tam veri) koşumlarının çıktılarını
# birbirinden AYIRIR. Bu ayrım olmadan, pilotta üretilen küçük Test A/B setleri ile
# hesaplanan baseline.json, tam moda geçildiğinde YENİDEN OLUŞTURULAN (farklı içerikli
# ve boyutlu) Test A/B setleriyle KARŞILAŞTIRILIRSA anlamsız/yanıltıcı bir regresyon
# kontrolüne yol açar. Bu yüzden işlenmiş veri, checkpoint, log ve değerlendirme
# çıktıları PILOT_MODE'a göre ayrı alt klasörlere yazılır.
MODE_TAG = "pilot" if PILOT_MODE else "full"

# prepare_datasets.py + build_chat_dataset.py çıktısının (ortak şemaya dönüştürülmüş,
# karıştırılmış, train/val/testA/testB olarak bölünmüş) yazıldığı klasör.
PROCESSED_DATA_DIR = DRIVE_ROOT / "processed_data" / MODE_TAG

# Eğitim çıktıları: LoRA adaptör ağırlıkları, Trainer checkpoint'leri, TensorBoard logları.
CHECKPOINT_DIR = DRIVE_ROOT / "checkpoints" / MODE_TAG
LOG_DIR = DRIVE_ROOT / "logs" / MODE_TAG

# Değerlendirme sonuçları (baseline + her epoch sonrası skorlar, JSON/CSV).
EVAL_OUTPUT_DIR = DRIVE_ROOT / "eval_outputs" / MODE_TAG

# analysis/ scriptlerinin ürettiği raporların (tokenizer analizi, görsel token EDA'sı) yazıldığı klasör.
ANALYSIS_OUTPUT_DIR = DRIVE_ROOT / "analysis_outputs"

# SMHD veri setinin MANUEL olarak indirilip Drive'a yerleştirileceği klasör.
# (SMHD izinli/e-posta ile dağıtılan bir veri seti olduğu için otomatik indirilemez;
#  bkz. README.md "SMHD erişimi" bölümü.) Beklenen iç yapı, orijinal dağıtımdaki gibi
# görüntü alt klasörleri + eşlenik .txt transkripsiyon dosyalarıdır.
SMHD_LOCAL_DIR = DRIVE_ROOT / "manual_datasets" / "SMHD"


def ensure_directories() -> None:
    """Yukarıdaki tüm klasörleri (yoksa) oluşturur. Sadece Colab'da, Drive
    bağlandıktan sonra çağrılmalıdır (notebook'ların ilk hücrelerinden biri budur).
    Yerel VS Code ortamında bilerek çağrılmaz, çünkü DRIVE_ROOT burada gerçek bir yol değildir."""
    for d in [
        RAW_DATA_DIR,
        PROCESSED_DATA_DIR,
        CHECKPOINT_DIR,
        LOG_DIR,
        EVAL_OUTPUT_DIR,
        ANALYSIS_OUTPUT_DIR,
        SMHD_LOCAL_DIR,
    ]:
        d.mkdir(parents=True, exist_ok=True)


# ---------------------------------------------------------------------------
# 2) MODEL
# ---------------------------------------------------------------------------
# İnce ayar yapılacak temel model. Bu proje özellikle bu modelin mimarisine
# (görsel encoder + merger + Qwen2.5 LLM decoder) göre tasarlanmıştır.
MODEL_ID = "Qwen/Qwen2.5-VL-3B-Instruct"

# ---------------------------------------------------------------------------
# 3) VERİ SETİ KAYNAKLARI
# ---------------------------------------------------------------------------
# Her kaynağın nereden geldiği ve nasıl indirildiği prepare_datasets.py içinde detaylıdır.
# Burada sadece kimlikler/yollar tutulur.

# (a) Sentetik basılı Türkçe kelime OCR verisi — HuggingFace Hub, doğrudan `load_dataset` ile.
#     Doğrulanmış şema: {"image": PIL.Image, "text": str}, split'ler train/val/test.
PRINTED_SYNTHETIC_HF_ID = "esengul3/turkish-word-ocr"

# (b) Gerçek Türkçe sahne metni (Scene Text Recognition) — sadece Kaggle'da barındırılıyor.
#     Kaggle API kimlik bilgisi (kaggle.json) gerektirir; bkz. README.md "Kaggle erişimi".
SCENE_TEXT_KAGGLE_SLUG = "serdaryildiz/turkish-scene-text-recognition-dataset"

# (c) Türkçe dilinde, el yazısı STİLİNDE render edilmiş sentetik veri — HuggingFace Hub.
#     %70 el yazısı stili + %30 basılı içerir (prepare_datasets.py sadece el yazısı
#     stilindeki alt kümeyi seçer). MIT lisanslı, serbestçe indirilebilir.
HANDWRITING_SYNTHETIC_HF_ID = "emredeveloper/turkish-ocr"

# (d) SMHD (Student Messy Handwritten Dataset) — GERÇEK ama İNGİLİZCE el yazısı, izinli dağıtım.
#     Türkçe karakter İÇERMEZ; burada amaç Türkçe metin öğretmek değil, modele gerçek/bozuk
#     el yazısı çizgi/strok çeşitliliğine karşı sağlamlık (robustness) kazandırmaktır.
#     USE_SMHD=False iken bu kaynak tamamen devre dışı kalır ve payı HANDWRITING_SYNTHETIC_HF_ID'ye
#     kayar (bkz. aşağıdaki KARIŞIM ORANLARI bölümü) — kod bu durumda da hatasız çalışır.
USE_SMHD = False  # SMHD izniniz gelip Drive'a yerleştirdiğinizde True yapın.

# (e) OmniDocBench — self-distillation (replay) için görsel KAYNAĞI. Ground-truth
#     etiketleri KULLANILMAZ; bunun yerine ince ayardan ÖNCEKİ (baseline) model bu
#     görseller üzerinde çalıştırılıp kendi çıktıları "replay" hedef metni olarak kullanılır
#     (replay_generation.py). Böylece modelin eski/genel yeteneklerini "unutmaması" hedeflenir.
REPLAY_SOURCE_HF_ID = "opendatalab/OmniDocBench"

# ---------------------------------------------------------------------------
# 4) KARIŞIM ORANLARI (Katastrofik unutmayı önlemek için veri karışımı)
# ---------------------------------------------------------------------------
# Üst seviye oranlar (toplamı 1.0 olmalı): %40 basılı, %35 el yazısı, %25 replay.
MIXTURE_PRINTED = 0.40
MIXTURE_HANDWRITING = 0.35
MIXTURE_REPLAY = 0.25

# "%40 basılı" kovası kullanıcının orijinal isteğinde tek bir kaynak (esengul3) olarak
# tanımlanmıştı; TS-TR (gerçek sahne metni) ayrı bir yüzdeye sahip değildi. En mantıklı
# yorum: TS-TR de "basılı/dizgi karakterli" bir kaynak olduğundan (el yazısı değil, gerçek
# fotoğraflarda dizgi/tabela yazısı) bu kovaya esengul3 ile birlikte girer. Aşağıdaki
# alt-oran bunu kontrol eder; isterseniz değiştirin.
PRINTED_SYNTH_VS_SCENE_RATIO = 0.70  # 0.70 esengul3 (sentetik), 0.30 TS-TR (gerçek sahne)

# "%35 el yazısı" kovası: USE_SMHD=True ise iki kaynak arasında bölünür; USE_SMHD=False
# ise (varsayılan) tamamı Türkçe sentetik el yazısı stiline (emredeveloper/turkish-ocr) gider.
HANDWRITING_SYNTH_VS_SMHD_RATIO = 0.60  # USE_SMHD=True iken: 0.60 turkish-ocr, 0.40 SMHD

# "%25 replay" kovası: OCR-replay (belge/metin okuma promptlarıyla) ve genel-görev replay
# (OCR dışı: açıklama, soru-cevap vb.) arasında bölünür. Genel-görev payı, modelin OCR
# dışındaki talimat-takip becerisini korumak için vardır.
REPLAY_OCR_VS_GENERAL_RATIO = 0.60  # 0.60 OCR-replay, 0.40 genel-görev replay

# Nihai eğitim setinin HEDEF toplam örnek sayısı. Tüm kova boyutları (printed_synthetic,
# scene_text, handwriting_synthetic, smhd_english, replay_ocr, replay_general) bu sayı
# ve yukarıdaki oranlardan TÜRETİLİR (bkz. compute_bucket_target_sizes). Böylece hem
# data/replay_generation.py (kaç görsel için pseudo-label üretileceğini bilmeli) hem de
# data/build_chat_dataset.py (her kovadan kaç örnek örnekleneceğini bilmeli) AYNI hedef
# sayılarını kullanır — tek doğruluk kaynağı burasıdır.
TOTAL_TRAIN_TARGET_SIZE = 900 if PILOT_MODE else 18000


def compute_bucket_target_sizes(total: int | None = None) -> dict:
    """Karışım oranlarından, her bir "yaprak" kaynak için HEDEF örnek sayısını hesaplar.

    Örnek: total=1000 iken -> printed_synthetic ~= 1000*0.40*0.70 = 280 gibi.
    `total=None` ise TOTAL_TRAIN_TARGET_SIZE kullanılır (PILOT_MODE'a göre otomatik).
    """
    total = TOTAL_TRAIN_TARGET_SIZE if total is None else total

    printed_total = total * MIXTURE_PRINTED
    handwriting_total = total * MIXTURE_HANDWRITING
    replay_total = total * MIXTURE_REPLAY

    handwriting_synth_ratio = 1.0 if not USE_SMHD else HANDWRITING_SYNTH_VS_SMHD_RATIO

    return {
        "printed_synthetic": round(printed_total * PRINTED_SYNTH_VS_SCENE_RATIO),
        "scene_text": round(printed_total * (1 - PRINTED_SYNTH_VS_SCENE_RATIO)),
        "handwriting_synthetic": round(handwriting_total * handwriting_synth_ratio),
        "smhd_english": round(handwriting_total * (1 - handwriting_synth_ratio)) if USE_SMHD else 0,
        "replay_ocr": round(replay_total * REPLAY_OCR_VS_GENERAL_RATIO),
        "replay_general": round(replay_total * (1 - REPLAY_OCR_VS_GENERAL_RATIO)),
    }

# ---------------------------------------------------------------------------
# 5) TEST SETLERİ (regresyon / iyileşme kontrolü) VE VERİ BÖLME
# ---------------------------------------------------------------------------
RANDOM_SEED = 42

# Her kaynaktan train'e girmeden ÖNCE ayrılan sabit-boyutlu değerlendirme dilimleri.
# Bu sayılar örnek sayısıdır (oran değil) — küçük ama istatistiksel olarak anlamlı tutulmuştur.
TEST_A_SIZE = 200  # Test Seti A: genel talimat-takip + karışık kaynak OCR (DÜŞMEMELİ)
TEST_B_SIZE = 200  # Test Seti B: Türkçe özel karakter / el yazısı ağırlıklı (YÜKSELMELİ)
VAL_SIZE = 200      # Eğitim sırasında ara kontrol için küçük bir validation dilimi

# Test Seti A'nın ne kadarının "genel görev" (replay havuzundan), ne kadarının
# "karışık kaynak OCR" (üç OCR kaynağından karışık) olacağını belirler.
TEST_A_GENERAL_RATIO = 0.5

# ---------------------------------------------------------------------------
# 6) GÖRSEL İŞLEME (min_pixels / max_pixels)
# ---------------------------------------------------------------------------
# Qwen2.5-VL görsel encoder'ı görüntüyü 28x28'lik (patch_size=14 * merge_size=2) bloklara
# böler; min_pixels/max_pixels bu bloklara göre görüntünün yeniden boyutlandırılacağı
# aralığı (piksel cinsinden) belirler ve DOLAYLI olarak üretilecek görsel token sayısını
# kontrol eder. Bu değerler analysis/vision_token_eda.py çıktısına göre ayarlanmalıdır;
# burada makul varsayılanlar verilmiştir (Qwen2-VL/2.5-VL resmi varsayılanlarından daha
# dar bir aralık — T4'te bellek tasarrufu ve OCR için yeterli çözünürlük dengesi için).
IMAGE_PATCH_FACTOR = 28  # 14 (patch_size) * 2 (merge_size); değiştirmeyin.
MIN_PIXELS = 256 * IMAGE_PATCH_FACTOR * IMAGE_PATCH_FACTOR   # ~200.704 piksel
MAX_PIXELS = 768 * IMAGE_PATCH_FACTOR * IMAGE_PATCH_FACTOR   # ~602.112 piksel

# ---------------------------------------------------------------------------
# 7) LoRA AYARLARI
# ---------------------------------------------------------------------------
# LLM decoder'ın attention (q,k,v,o) ve MLP (gate,up,down) katmanlarına uygulanan LoRA.
LORA_R = 16
LORA_ALPHA = 32
LORA_DROPOUT = 0.05

# Merger (görsel özellikleri LLM gömme uzayına projekte eden küçük MLP) için ayrı,
# daha düşük ranklı bir LoRA — merger'ın parametre sayısı zaten çok küçük (~5M) olduğundan
# yüksek rank gereksizdir.
MERGER_LORA_R = 8
MERGER_LORA_ALPHA = 16

# embed_tokens LoRA'sı: SADECE analysis/tokenizer_analysis.py raporu Türkçe karakterlerin
# aşırı parçalandığını gösterirse elle True yapılması önerilir (bkz. rapor çıktısındaki
# öneri satırı). Varsayılan olarak kapalıdır.
ENABLE_EMBED_LORA = False
EMBED_LORA_R = 8
EMBED_LORA_ALPHA = 16

# Görsel encoder LoRA'sı: varsayılan olarak KAPALI (encoder tamamen donuk kalır).
# SADECE değerlendirme sonrası görsel-algı kaynaklı hatalar (ör. karakterleri doğru
# okuyamama değil, karakterleri birbirine karıştırma/görememe) tespit edilirse elle
# açılması önerilir. Açıldığında yalnızca `fullatt_block_indexes`'te belirtilen SON
# `VISION_LORA_LAST_N_FULLATT_BLOCKS` adet tam-attention bloğuna LoRA uygulanır
# (pencereli/windowed attention bloklarına DOKUNULMAZ).
ENABLE_VISION_LORA = False
VISION_LORA_R = 8
VISION_LORA_ALPHA = 16
VISION_LORA_LAST_N_FULLATT_BLOCKS = 1

# ---------------------------------------------------------------------------
# 8) EĞİTİM HİPERPARAMETRELERİ
# ---------------------------------------------------------------------------
# (PILOT_MODE bayrağı bu dosyanın en başında, bölüm 0'da tanımlıdır.)

# Pilot modda her kaynaktan kaç örnek kullanılacağı (hızlı çalışsın diye).
PILOT_SAMPLES_PER_SOURCE = 300

LEARNING_RATE = 1e-4
NUM_TRAIN_EPOCHS = 1 if PILOT_MODE else 3
LR_SCHEDULER_TYPE = "cosine"
WARMUP_RATIO = 0.03
WEIGHT_DECAY = 0.01

# Pilot modda T4'ün 16 GB belleğine sığmak için küçük batch + yüksek gradient accumulation;
# tam eğitimde A100'ün daha büyük belleğinden yararlanmak için daha büyük batch.
PER_DEVICE_TRAIN_BATCH_SIZE = 1 if PILOT_MODE else 2
GRADIENT_ACCUMULATION_STEPS = 8 if PILOT_MODE else 16

MAX_SEQ_LENGTH = 1024  # Chat şablonu + görsel token'lar + hedef metin için üst sınır.

# Pilot modda 4-bit (bitsandbytes) yükleme ÖNERİLİR (T4'te 3B model + aktivasyonlar için
# bellek tasarrufu); tam eğitimde A100'de buna gerek yoktur (LoRA zaten ana ağırlıkları
# dondurur, ama 4-bit ekstra bellek tasarrufu sağlar — hız/doğruluk dengesi için tam
# eğitimde kapalı tutulması önerilir).
LOAD_IN_4BIT = PILOT_MODE

# ---------------------------------------------------------------------------
# 9) TEST A/B CALLBACK (REGRESYON KONTROLÜ) AYARLARI
# ---------------------------------------------------------------------------
# Test Seti A "kompozit skoru" (bkz. evaluation/evaluate.py: OCR-karışık CER'i ve
# genel-görev ROUGE-L'ini TEST_A_GENERAL_RATIO ile ağırlıklandırıp TEK bir "yüksek=iyi"
# sayıya indirger), baseline'a göre bu ORANDAN fazla KÖTÜLEŞİRSE (göreli düşerse) eğitim
# durdurulur. Örn. 0.15 => kompozit skorda göreli %15 düşüş.
TEST_A_REGRESSION_RELATIVE_THRESHOLD = 0.15

# ---------------------------------------------------------------------------
# 10) PROMPT ÇEŞİTLİLİĞİ
# ---------------------------------------------------------------------------
# Not: Gerçek prompt metinleri data/prompts.py içindedir. Burada sadece kaç farklı
# şablonun havuzda bulunduğu belgelenir (bilgi amaçlı, kod tarafından kullanılmaz).
OCR_PROMPT_POOL_SIZE_HINT = 14
GENERAL_PROMPT_POOL_SIZE_HINT = 10
