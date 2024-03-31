import sys
import os

sys.path.insert(1, os.path.join(sys.path[0], ".."))
from filtering import LoadParameters

document = """Sabah Bakıda və Abşeron yarımadasında şimal-qərb küləyinin arabir güclənəcəyi gözlənilir - AZƏRTAC – Azərbaycan Dövlət İnformasiya Agentliyi
01.07.2018 [14:08]
Ölkə ərazisində iyulun 2-nə gözlənilən hava proqnozu açıqlanıb.
Ekologiya və Təbii Sərvətlər Nazirliyinin Milli Hidrometeorologiya Departamentindən AZƏRTAC-a verilən məlumata əsasən, Bakıda və Abşeron yarımadasında hava günəşli olacaq, mülayim şimal-qərb küləyi arabir güclənəcək. Havanın temperaturunun gecə 24-27° isti, gündüz 35-39° isti olacağı gözlənilir. Abşeron yarımadasının şimal çimərliklərində (Sumqayıt, Novxanı, Pirşağı, Nardaran, Bilgəh, Zuğulba) arabir güclənən şimal-qərb küləyi əsəcək və dəniz suyunun temperaturu 27-28° isti təşkil edəcək. Cənub çimərliklərində (Türkan, Hövsan, Sahil, Şıx) də arabir güclənən şimal-qərb küləyi əsəcək. Dəniz suyunun temperaturu 28-29° isti təşkil edəcək. Abşeron çimərliklərində davam edən anomal isti hava şəraiti ilə bağlı günorta saatlarında (11-dən 17-dək) günəş şüalarının birbaşa təsiri altında uzun müddət olmaq risklidir.
AZERTAG.AZ :Sabah Bakıda və Abşeron yarımadasında şimal-qərb küləyinin arabir güclənəcəyi gözlənilir"""

# document = "Hello my name is Eljan"

model_lang_id = LoadParameters.load_model_lang_id(
    "az",
    "/Users/eljan/Documents/azGPT/bigscience-dataprep/preprocessing/training/01b_oscar_cleaning_and_filtering/visualization/lid.176.bin",
)
document = document.lower().replace("\n", " ")
pred = model_lang_id.predict(document)
lang_pred_fasttext_id = pred[0][0].replace("__label__", "")
print(lang_pred_fasttext_id)
is_azerbaijani = True if lang_pred_fasttext_id == "az" else False
score_pred = pred[1][0] if is_azerbaijani else 0
print(is_azerbaijani, score_pred)

sys.exit()

path = "/Users/eljan/Documents/azGPT/bigscience-dataprep/preprocessing/training/01b_oscar_cleaning_and_filtering/visualization"
lang_dataset_id = "en"
path_fasttext_model = f"{path}/lid.176.bin"
path_sentencepiece_model = f"{path}/{lang_dataset_id}.sp.model"
path_kenlm_model = f"{path}/{lang_dataset_id}.arpa.bin"
path_save_stats = f"{path}/{lang_dataset_id}_examples_with_stats.json"
param = LoadParameters.load_parameters(lang_dataset_id)

sentencepiece_model = LoadParameters.load_sentencepiece_model(
    lang_dataset_id, path_sentencepiece_model
)
sentencepiece_model_tok = sentencepiece_model if param["tokenization"] else None

document = "This is a test sentence. This is another test sentence. What is the meaning of life? I don't know. No one knows it. So, !!! we have to!!! find it. Elon Musk is a good person."
document = """Attalus won an important victory, the Battle of the Caecus River, over the Galatians, a group of migratory Celtic tribes from Thrace, who had been plundering and exacting tribute throughout most of Asia Minor for more than a generation. The victory was celebrated with a triumphal monument at Pergamon (The Dying Gaul) and Attalus taking the name "Soter" and the title of king. He participated in the first and second Macedonian Wars against Philip V of Macedon as a loyal ally of the Roman Republic, although Pergamene participation was ultimately rather minor in these wars.[3] He conducted numerous naval operations throughout the Aegean, gained the island of Aegina for Pergamon during the first war and Andros during the second, twice narrowly escaping capture at the hands of Philip V. During his reign, Pergamon also repeatedly struggled with the neighboring Seleucid Empire to the east, resulting in both successes and setbacks."""
document_tokenized = sentencepiece_model.encode_as_pieces(document)
# print(document_tokenized)


from filtering import ModifyingDocuments

document = ModifyingDocuments.normalization(
    document=document,
    remove_non_printing_characters=True,
    strip=True,
    lower_case=False,
    uniform_whitespace=True,
    replace_digits_with_zeros=True,
    replace_unicode_punctuation=True,
)

print(document)
