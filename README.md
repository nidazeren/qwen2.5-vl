# Qwen2.5-VL-3B-Instruct — Türkçe OCR İnce Ayarı (LoRA)

Bu proje, [Qwen/Qwen2.5-VL-3B-Instruct](https://huggingface.co/Qwen/Qwen2.5-VL-3B-Instruct)
modelinin genel görsel-dil anlama, talimat takip etme ve çok dilli yeteneklerini
BOZMADAN, **Türkçe karakterlerde** (ç, ğ, ı, ö, ş, ü) ve **Türkçe el yazısında** OCR
doğruluğunu LoRA tabanlı denetimli ince ayar (SFT) ile artırmayı hedefler.

Kod **hiç yerelde çalıştırılmamıştır** — yalnızca dosya olarak yazılmıştır. Kurulum ve
çalıştırma **Google Colab** üzerinden yapılır: pilot denemeler **T4 GPU**'da, tam eğitim
**A100 GPU**'da (Google Drive'a kalıcı checkpoint/log kaydıyla).

## Hızlı Başlangıç

1. Bu repoyu Colab'da klonlayın ve `notebooks/00_pilot_t4_setup_eda.ipynb`'yi T4 GPU ile
   baştan sona çalıştırın (kurulum, EDA, baseline, küçük-ölçek pilot eğitim).
2. Pilot hatasız tamamlandıysa, `notebooks/01_full_training_a100.ipynb`'yi A100 GPU ile
   çalıştırın (tam veri, tam eğitim).

Her iki notebook da hücreleri **sırayla** çalıştırmanızı bekler; her hücrenin ne yaptığı
üstündeki markdown açıklamasında yazılıdır.

## Proje Yapısı

```
configs/config.py          Tüm sabitler (tek doğruluk kaynağı): model id, veri seti
                            kimlikleri, karışım oranları, LoRA hiperparametreleri,
                            min/max pixels, USE_SMHD, PILOT_MODE, Drive yolları.
data/prompts.py             OCR görev-prompt havuzu + replay için genel-görev prompt havuzu.
data/prepare_datasets.py    5 ham kaynağı indirir/normalize eder -> {image,text,source}.
data/io_utils.py            Ortak diske-yazma/okuma yardımcıları (data/ paketi içi).
data/replay_generation.py   Self-distillation ile replay (katastrofik unutmayı önleme) verisi üretir.
data/build_chat_dataset.py  Train/Val/TestA/TestB ayrımı (sızıntısız) + karışım oranı uygulaması.
analysis/tokenizer_analysis.py  Türkçe karakter BBPE parçalanma analizi (embed_tokens LoRA kararı için).
analysis/vision_token_eda.py    min_pixels/max_pixels -> görsel token sayısı EDA'sı.
training/lora_setup.py      Model+processor yükleme, LoRA hedef modüllerinin dinamik keşfi ve uygulanması.
training/collate.py         Özel data collator: chat şablonu + loss masking (-100).
training/callbacks.py       Epoch-sonu Test A/B değerlendirmesi + regresyon tespitinde durdurma.
training/inference_utils.py Ortak metin üretme (generate) yardımcı fonksiyonu.
training/train_sft.py       Ana eğitim scripti (trl.SFTTrainer).
evaluation/metrics.py       CER/WER/tam-eşleşme/ROUGE-L metrikleri.
evaluation/build_test_sets.py  Önceden inşa edilmiş Test A/B/Val setlerinin ince erişim katmanı.
evaluation/evaluate.py      Test A/B üzerinde model değerlendirme çalıştırıcısı.
notebooks/00_pilot_t4_setup_eda.ipynb    T4: kurulum, EDA, baseline, pilot eğitim.
notebooks/01_full_training_a100.ipynb    A100: tam veri, tam eğitim.
```

## Veri Kaynakları ve Önemli Notlar

| Kaynak | Nereden | Not |
|---|---|---|
| `esengul3/turkish-word-ocr` | HuggingFace Hub | Sentetik, basılı, 250K Türkçe kelime görseli. |
| TS-TR (sahne metni) | Kaggle (`serdaryildiz/...`) | Gerçek sahne fotoğrafları; Kaggle API kimlik bilgisi gerektirir. **Varsayılan olarak KAPALI** (`USE_SCENE_TEXT=False`) — aşağıya bakın. |
| `emredeveloper/turkish-ocr` | HuggingFace Hub | Türkçe, %70 el yazısı stilinde render edilmiş sentetik veri. |
| SMHD | GitHub (`hiqmatNisa/SMHD`) | **Gerçek ama İNGİLİZCE el yazısı**, izin formu gerektirir (aşağıya bakın). Varsayılan olarak KAPALI (`USE_SMHD=False`). |
| `opendatalab/OmniDocBench` | HuggingFace Hub | Yalnızca GÖRSEL kaynağı olarak (self-distillation replay için); etiketleri kullanılmaz. |

### Neden TS-TR (Kaggle) varsayılan olarak kapalı?

Kaggle API kimlik bilgisi kurulumu (kaggle.json indirme/Colab Secrets) bazı ortamlarda
sorun çıkarabildiğinden, `configs/config.py` içinde `USE_SCENE_TEXT=False` (varsayılan)
iken bu kaynak **tamamen atlanır** — Kaggle'a hiç bağlanılmaz, kimlik bilgisi gerekmez.
"%40 basılı" payının tamamı otomatik olarak `esengul3/turkish-word-ocr`'a kayar
(bkz. `compute_bucket_target_sizes`), pipeline eksiksiz çalışır.

**TS-TR'yi (gerçek sahne metni) sonradan eklemek isterseniz:**
1. `configs/config.py` içinde `USE_SCENE_TEXT = True` yapın.
2. Kaggle API kimlik bilgisi kurun (aşağıdaki "Kaggle API kimlik bilgisi" bölümü).
3. Değişikliği GitHub'a push edip Colab'da `git pull` çekip veri hazırlama adımlarını
   tekrar çalıştırın.

### Neden SMHD varsayılan olarak kapalı?

Planlama sırasında yapılan araştırma, kullanıcının orijinal isteğindeki "Student Messy
Handwritten Dataset (SMHD)"nin aslında **İngilizce** (RMIT/Avustralya, öğrenci
özet/deneme metinleri) olduğunu ve **e-posta ile izin formu** gerektirdiğini ortaya
çıkardı — doğrudan Türkçe el yazısı verisi değildir. `configs/config.py` içinde
`USE_SMHD=False` iken kod, el yazısı payının tamamını `emredeveloper/turkish-ocr`
(Türkçe, sentetik el yazısı stili) kaynağına kaydırarak **hatasız çalışır**.

**SMHD'yi kullanmak isterseniz:**
1. https://github.com/hiqmatNisa/SMHD adresindeki talimatları takip edip izin formunu
   doldurun (`hiqmat.nisa@gmail.com`).
2. İndirdiğiniz veriyi Drive'da `config.SMHD_LOCAL_DIR` (varsayılan:
   `.../qwen25vl_turkish_ocr/manual_datasets/SMHD`) altına, orijinal klasör yapısıyla
   (görsel + eşlenik `.txt` transkripsiyon) yerleştirin.
3. `configs/config.py` içinde `USE_SMHD = True` yapın.

Bu veri İngilizce olduğundan, amacı Türkçe metin öğretmek DEĞİL, modele gerçek/bozuk
el yazısı çizgi çeşitliliğine karşı ek sağlamlık kazandırmaktır (bkz. config.py'deki
`HANDWRITING_SYNTH_VS_SMHD_RATIO`).

### Kaggle API kimlik bilgisi (yalnızca USE_SCENE_TEXT=True yaparsanız gerekir)

1. https://www.kaggle.com adresinde hesabınıza girin -> **Settings > API > Create New Token**.
2. Ya `kaggle.json` dosyasını `notebooks/00_pilot_t4_setup_eda.ipynb`'nin ilgili
   hücresinde yükleyin, ya da (indirme sorun çıkarırsa) Colab'ın sol kenar çubuğundaki
   **Secrets (🔑)** panelinden `KAGGLE_USERNAME`/`KAGGLE_KEY` ekleyin — hücre ikisini de
   otomatik dener. Bir kez yaparsanız notebook Drive'a kaydeder, sonraki çalıştırmalarda
   tekrar istemez.

TS-TR'nin iç klasör/etiket formatı Kaggle'da ayrıntılı belgelenmediğinden,
`data/prepare_datasets.py` bunu OTOMATİK ALGILAMAYA çalışır (yaygın `gt.txt`/`labels.txt`/
`annotations.json` kalıpları). Algılama başarısız olursa, script indirilen klasör
içeriğini yazdırıp `data/prepare_datasets.py` içindeki `TS_TR_MANUAL_HINT` değişkenini
doldurmanızı ister (tek seferlik, elle bir müdahale).

## Karışım Oranları (Katastrofik Unutmayı Önleme)

`configs/config.py` içinde merkezi olarak tanımlıdır ve `compute_bucket_target_sizes()`
ile her kaynağın tam hedef örnek sayısına dönüştürülür:

- **%40 basılı**: `esengul3` (sentetik, varsayılan %100) [+ TS-TR (gerçek sahne metni, %30), yalnızca `USE_SCENE_TEXT=True` iken]
- **%35 el yazısı**: `turkish-ocr` (Türkçe, sentetik el yazısı stili) [+ SMHD, opsiyonel]
- **%25 replay** (self-distillation): OCR-replay (%60) + genel-görev replay (%40)

> Kullanıcının orijinal isteğinde TS-TR'ye ayrı bir yüzde verilmemişti; en mantıklı
> yorum olarak "%40 basılı" kovasına esengul3 ile birlikte dahil edildi (ikisi de
> dizgi/basılı karakter içerir, el yazısı değildir). `PRINTED_SYNTH_VS_SCENE_RATIO`
> ile bu alt-oranı değiştirebilirsiniz. TS-TR varsayılan olarak KAPALI olduğundan
> (`USE_SCENE_TEXT=False`), bu kova şu an tamamen `esengul3`'ten geliyor.

## LoRA Tasarımı

- LLM decoder attention (q,k,v,o) + MLP (gate,up,down): `r=16, alpha=32` (varsayılan).
- Merger (görsel->LLM projektör): ayrı, daha düşük rank (`r=8`).
- `embed_tokens`: **varsayılan kapalı**; `analysis/tokenizer_analysis.py` raporu Türkçe
  karakterlerin aşırı parçalandığını gösterirse `ENABLE_EMBED_LORA=True` yapmanız önerilir.
- Görsel encoder: **varsayılan tamamen donuk**; yalnızca değerlendirme sonrası
  görsel-algı kaynaklı hatalar tespit edilirse `ENABLE_VISION_LORA=True` ile
  `fullatt_block_indexes`'teki son bloklara hafif LoRA açılabilir.

Hedef modüller `model.named_modules()` üzerinde **dinamik olarak** keşfedilir (regex +
`isinstance` kontrolleriyle, "visual" geçen hiçbir modül LLM hedeflerine dahil edilmez);
bu, transformers sürümleri arasında modül isimleri değişse bile kodun kırılmamasını sağlar.

## Test Seti A / B ve Regresyon Kontrolü

- **Test Seti A** (düşmemeli): genel talimat-takip (replay_general) + karışık-kaynak OCR.
- **Test Seti B** (yükselmeli): Türkçe özel karakter/el yazısı ağırlıklı (handwriting_synthetic'ten).
- Her epoch sonunda `training/callbacks.py`, Test A'nın "kompozit skorunu" baseline'a göre
  kontrol eder; `configs.TEST_A_REGRESSION_RELATIVE_THRESHOLD`'u aşan bir düşüş olursa
  eğitim OTOMATİK DURUR ve `eval_outputs/<mode>/regression_report.json` içine önerilen
  sonraki adımlar (learning rate düşürme, karışım oranı değiştirme) yazılır. Bu
  YARI-OTOMATİK bir mekanizmadır: hiperparametreyi siz değiştirip yeniden başlatırsınız.

## Pilot (T4) vs Tam (A100)

`configs/config.py` içindeki `PILOT_MODE` bayrağı (ortam değişkeni
`QWEN_OCR_PILOT_MODE` ile kontrol edilir) veri boyutunu, epoch sayısını, batch boyutunu,
4-bit yüklemeyi ve dtype/attention seçimini otomatik ayarlar. Pilot ve tam çalıştırmaların
çıktıları (`processed_data/`, `checkpoints/`, `eval_outputs/`) `MODE_TAG` alt klasörleriyle
birbirinden AYRILMIŞTIR — böylece pilotun küçük Test A/B setleriyle hesaplanan bir
baseline, yanlışlıkla tam-moddaki farklı Test A/B setleriyle karşılaştırılmaz.

## Lisans Notları

Kullanılan veri setlerinin kendi lisansları geçerlidir: `esengul3/turkish-word-ocr`
(CC-BY-SA-4.0), TS-TR (CC BY-NC-4.0, **ticari olmayan kullanım**), `emredeveloper/turkish-ocr`
(MIT), OmniDocBench (bkz. HuggingFace sayfası), SMHD (izinli dağıtım, akademik/araştırma
amaçlı). Bu projeyi ticari amaçla kullanmadan önce TS-TR'nin CC BY-NC-4.0 lisansını
kontrol edin.
