"""
prompts.py
==========
Eğitim örnekleri chat formatına (user-image-assistant) dönüştürülürken, kullanıcı
mesajındaki talimat metni bu dosyadaki havuzlardan RASTGELE seçilir.

Tasarım kuralı — ÇOK ÖNEMLİ:
    OCR_PROMPT_POOL içindeki hiçbir şablon, görselin kaynağını (sentetik mi gerçek mi,
    basılı mı el yazısı mı) İFŞA ETMEZ. Örneğin "bu el yazısını oku" gibi bir talimat
    YAZILMAZ; çünkü modelin, çıkarım (inference) zamanında karşılaşacağı gerçek kullanıcı
    girdilerinde böyle bir ön-bilgi olmayacaktır. Amaç, modelin talimatın ifadesinden
    değil, GÖRSELİN KENDİSİNDEN görev türünü çıkarmayı öğrenmesidir (göreve dayalı,
    kaynak-agnostik prompt tasarımı).

    GENERAL_PROMPT_POOL ise SADECE self-distillation replay verisi (replay_generation.py)
    için kullanılır ve OCR dışı genel görevleri (açıklama, soru-cevap, özetleme) kapsar;
    bu, modelin sadece OCR yapan dar bir modele dönüşmesini (katastrofik unutma) önlemeye
    yardımcı olur.
"""

import random

# ---------------------------------------------------------------------------
# OCR görev-prompt havuzu (basılı, sahne metni ve el yazısı örnekleri için ORTAK).
# Hepsi aynı görevi ("görseldeki metni birebir yaz") farklı doğal dil ifadeleriyle sorar.
# ---------------------------------------------------------------------------
OCR_PROMPT_POOL = [
    "Bu görseldeki metni oku ve birebir yaz.",
    "Görselde yazan metni aynen aktar.",
    "Görüntüdeki yazıyı transkribe et.",
    "Bu resimde görünen tüm metni yaz.",
    "Görseldeki yazıyı, hiçbir değişiklik yapmadan metne dök.",
    "Resimdeki yazıyı okuyup bana ilet.",
    "Görselde yer alan metni tanı ve yaz.",
    "Bu görüntüde ne yazıyor? Metni birebir yaz.",
    "Görseldeki karakterleri oku ve metin olarak ver.",
    "Görselde geçen yazıyı olduğu gibi kopyala.",
    "Bu görseli oku ve içindeki metni yaz.",
    "Görüntüde bulunan yazılı içeriği metne aktar.",
    "Resimdeki metni tam olarak transkribe et.",
    "Görseldeki yazının metin karşılığını yaz.",
]

# ---------------------------------------------------------------------------
# Genel-görev prompt havuzu — SADECE replay (self-distillation) verisi için.
# OCR dışı görevler: açıklama, soru-cevap, özetleme, sayma vb. Bu prompt'lara karşılık
# gelen "doğru cevap", replay_generation.py içinde MODELİN KENDİSİ (ince ayardan önceki
# baseline hali) tarafından üretilir (pseudo-label / self-distillation).
# ---------------------------------------------------------------------------
GENERAL_PROMPT_POOL = [
    "Bu görselde neler görüyorsun? Kısaca açıkla.",
    "Görseli detaylı bir şekilde tarif et.",
    "Bu belgenin genel yapısı hakkında bilgi ver.",
    "Görseldeki içeriği özetle.",
    "Bu görselin türü nedir (belge, tablo, fotoğraf vb.)? Açıkla.",
    "Görselde dikkatini çeken unsurları listele.",
    "Bu görüntüdeki düzeni (başlık, paragraf, tablo vb.) tarif et.",
    "Görselin genel içeriğini bir-iki cümleyle açıkla.",
    "Bu görselde hangi bilgiler öne çıkıyor?",
    "Görseli, hiç görmeyen birine anlatır gibi tarif et.",
]


def sample_ocr_prompt(rng: random.Random) -> str:
    """OCR havuzundan verilen rastgele üreteç (rng) ile bir talimat seçer.

    Deterministik/tekrarlanabilir veri hazırlığı için `rng`'nin çağıran taraftan
    (config.RANDOM_SEED ile başlatılmış) geçirilmesi beklenir; modül seviyesinde
    global `random` KULLANILMAZ ki farklı çalıştırmalarda aynı veri seti üretilebilsin.
    """
    return rng.choice(OCR_PROMPT_POOL)


def sample_general_prompt(rng: random.Random) -> str:
    """Genel-görev havuzundan (yalnızca replay verisi için) bir talimat seçer."""
    return rng.choice(GENERAL_PROMPT_POOL)
