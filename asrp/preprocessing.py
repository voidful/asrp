import re
import string
import unicodedata

import unidecode

langs = ['ab', 'ar', 'as', 'br', 'ca', 'cnh', 'cs', 'cv', 'cy', 'de', 'dv', 'el', 'en', 'eo', 'es', 'et', 'eu', 'fa',
         'fi', 'fr', 'fy-NL', 'ga-IE', 'hi', 'hsb', 'hu', 'ia', 'id', 'it', 'ja', 'ka', 'kab', 'ky', 'lg', 'lt', 'lv',
         'mn', 'mt', 'nl', 'or', 'pa-IN', 'pl', 'pt', 'rm-sursilv', 'rm-vallader', 'ro', 'ru', 'rw', 'sah', 'sl',
         'sv-SE', 'ta', 'th', 'tr', 'tt', 'uk', 'vi', 'vot', 'zh-CN', 'zh-HK', 'zh-TW']


def fun_ar(batch):
    chars_to_ignore_regex = '[\\\\\\\\\\\\\\\\؛\\\\\\\\\\\\\\\\—\\\\\\\\\\\\\\\\_get\\\\\\\\\\\\\\\\«\\\\\\\\\\\\\\\\»\\\\\\\\\\\\\\\\ـ\\\\\\\\\\\\\\\\ـ\\\\\\\\\\\\\\\\,\\\\\\\\\\\\\\\\?\\\\\\\\\\\\\\\\.\\\\\\\\\\\\\\\\!\\\\\\\\\\\\\\\\-\\\\\\\\\\\\\\\\;\\\\\\\\\\\\\\\\:\\\\\\\\\\\\\\\\"\\\\\\\\\\\\\\\\“\\\\\\\\\\\\\\\\%\\\\\\\\\\\\\\\\‘\\\\\\\\\\\\\\\\”\\\\\\\\\\\\\\\\�\\\\\\\\\\\\\\\\#\\\\\\\\\\\\\\\\،\\\\\\\\\\\\\\\\☭,\\\\\\\\\\\\\\\\؟]'
    batch["sentence"] = re.sub(chars_to_ignore_regex, '', batch["sentence"]).lower()
    return batch


def fun_as(batch):
    chars_to_ignore_regex = '[\\,\\?\\.\\!\\-\\;\\:\\"\\“\\%\\”\\়\\।]'
    batch["sentence"] = re.sub('’ ', ' ', batch["sentence"])
    batch["sentence"] = re.sub(' ‘', ' ', batch["sentence"])
    batch["sentence"] = re.sub('’|‘', '\'', batch["sentence"])
    batch["sentence"] = re.sub(chars_to_ignore_regex, '', batch["sentence"]).lower()
    return batch


def fun_br(batch):
    chars_to_ignore_regex = '[\\,\,\?\.\!\;\:\"\“\%\”\�\(\)\/\«\»\½\…]'
    batch["sentence"] = re.sub(chars_to_ignore_regex, '', batch["sentence"]).lower()
    batch["sentence"] = re.sub(chars_to_ignore_regex, '', batch["sentence"]).lower() + " "
    batch["sentence"] = re.sub("ʼ", "'", batch["sentence"])
    batch["sentence"] = re.sub("’", "'", batch["sentence"])
    batch["sentence"] = re.sub('‘', "'", batch["sentence"])
    return batch


def fun_ca(batch):
    chars_to_ignore_regex = '[\,\?\.\!\;\:\"\“]'
    batch["sentence"] = re.sub(chars_to_ignore_regex, '', batch["sentence"]).lower()
    return batch


def fun_cnh(batch):
    chars_to_ignore_regex = '[\,\?\.\!\-\;\:\"\“\%\‘\”\�\/]'
    batch["sentence"] = re.sub(chars_to_ignore_regex, '', batch["sentence"]).lower()
    return batch


def fun_cs(batch):
    chars_to_ignore = [",", "?", ".", "!", "-", ";", ":", '""', "%", "'", '"', "�", '«', '»', '—', '…', '(', ')', '*',
                       '”', '“']
    chars_to_ignore_regex = f'[{"".join(chars_to_ignore)}]'
    batch["sentence"] = re.sub(chars_to_ignore_regex, '', batch["sentence"]).lower().strip()
    batch["sentence"] = re.sub(re.compile('[äá]'), 'a', batch['sentence'])
    batch["sentence"] = re.sub(re.compile('[öó]'), 'o', batch['sentence'])
    batch["sentence"] = re.sub(re.compile('[èé]'), 'e', batch['sentence'])
    batch["sentence"] = re.sub(re.compile("[ïí]"), 'i', batch['sentence'])
    batch["sentence"] = re.sub(re.compile("[üů]"), 'u', batch['sentence'])
    batch['sentence'] = re.sub('  ', ' ', batch['sentence'])
    return batch


def fun_cv(batch):
    sent = batch["sentence"].lower()
    # replace non-alpha characters with space
    sent = "".join(ch if ch.isalpha() else " " for ch in sent)
    # remove repeated spaces
    sent = " ".join(sent.split())
    batch["sentence"] = sent
    return batch


def fun_cy(batch):
    chars_to_ignore_regex = '[\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\,\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\?\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\.\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\!\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\-\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\u2013\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\u2014\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\;\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\:\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\"\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\%\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\]'
    batch["sentence"] = re.sub(chars_to_ignore_regex, '', batch["sentence"]).lower()
    return batch


def fun_de(batch):
    chars_to_ignore_regex = '[\\,\\?\\.\\!\\-\\;\\:\\"\\“]'
    batch["sentence"] = re.sub(chars_to_ignore_regex, '', batch["sentence"]).lower()
    return batch


def fun_dv(batch):
    chars_to_ignore_regex = '[\,\?\.\!\-\;\:\"\“\%\‘\”\�\،\.\؟\!\'\"\–\’]'
    batch["sentence"] = re.sub(chars_to_ignore_regex, '', batch["sentence"]).lower()
    return batch


def fun_el(batch):
    chars_to_ignore_regex = '[\\\\\\\\,\\\\\\\\?\\\\\\\\.\\\\\\\\!\\\\\\\\-\\\\\\\\;\\\\\\\\:\\\\\\\\"\\\\\\\\“\\\\\\\\%\\\\\\\\‘\\\\\\\\”\\\\\\\\�]'
    batch["sentence"] = re.sub(chars_to_ignore_regex, '', batch["sentence"]).lower()
    return batch


def fun_en(batch):
    sent = batch["sentence"].lower()
    # normalize apostrophes
    sent = sent.replace("’", "'")
    # replace non-alpha characters with space
    sent = "".join(ch if ch.isalpha() or ch == "'" else " " for ch in sent)
    # remove repeated spaces
    sent = " ".join(sent.split())
    batch["sentence"] = sent
    return batch


def fun_eo(batch):
    chars_to_ignore_regex = """[\\\\\\\\,\\\\\\\\?\\\\\\\\.\\\\\\\\!\\\\\\\\-\\\\\\\\;\\\\\\\\:\\\\\\\\"\\\\\\\\“\\\\\\\\%\\\\\\\\‘\\\\\\\\”\\\\\\\\�\\\\\\\\„\\\\\\\\«\\\\\\\\(\\\\\\\\»\\\\\\\\)\\\\\\\\’\\\\\\\\']"""
    batch["sentence"] = re.sub(chars_to_ignore_regex, '', batch["sentence"]).lower().replace('—', ' ').replace('–', ' ')
    return batch


def fun_es(batch):
    # remove_special_characters
    chars_to_ignore_regex = '[\,\¿\?\.\¡\!\-\;\:\"\“\%\‘\”\￼\…\’\ː\'\‹\›\`\´\®\—\→]'
    chars_to_ignore_pattern = re.compile(chars_to_ignore_regex)
    batch["sentence"] = chars_to_ignore_pattern.sub('', batch["sentence"]).lower() + " "
    # replace_diacritics
    sentence = batch["sentence"]
    sentence = re.sub('ì', 'í', sentence)
    sentence = re.sub('ù', 'ú', sentence)
    sentence = re.sub('ò', 'ó', sentence)
    sentence = re.sub('à', 'á', sentence)
    batch["sentence"] = sentence
    # replace_additional
    sentence = batch["sentence"]
    sentence = re.sub('ã', 'a', sentence)  # Portuguese, as in São Paulo
    sentence = re.sub('ō', 'o', sentence)  # Japanese
    sentence = re.sub('ê', 'e', sentence)  # Português
    batch["sentence"] = sentence
    return batch


def fun_et(batch):
    sent = batch["sentence"].lower()
    # normalize apostrophes
    sent = sent.replace("’", "'")
    # replace non-alpha characters with space
    sent = "".join(ch if ch.isalpha() or ch == "'" else " " for ch in sent)
    # remove repeated spaces
    sent = " ".join(sent.split())
    batch["sentence"] = sent
    return batch


def fun_eu(batch):
    chars_to_ignore_regex = '[\,\¿\?\.\¡\!\-\;\:\"\“\%\‘\”\￼\…\’\ː\'\‹\›\`\´\®\—\→]'
    chars_to_ignore_pattern = re.compile(chars_to_ignore_regex)
    batch["sentence"] = chars_to_ignore_pattern.sub('', batch["sentence"]).lower() + " "
    return batch


def fun_fa(batch):
    import hazm
    _normalizer = hazm.Normalizer()
    chars_to_ignore = [
        ",", "?", ".", "!", "-", ";", ":", '""', "%", "'", '"', "�",
        "#", "!", "؟", "?", "«", "»", "،", "(", ")", "؛", "'ٔ", "٬", 'ٔ', ",", "?",
        ".", "!", "-", ";", ":", '"', "“", "%", "‘", "”", "�", "–", "…", "_", "”", '“', '„',
        'ā', 'š',
        # "ء",
    ]
    chars_to_ignore = chars_to_ignore + list(string.ascii_lowercase + string.digits)
    chars_to_mapping = {
        'ك': 'ک', 'دِ': 'د', 'بِ': 'ب', 'زِ': 'ز', 'ذِ': 'ذ', 'شِ': 'ش', 'سِ': 'س', 'ى': 'ی',
        'ي': 'ی', 'أ': 'ا', 'ؤ': 'و', "ے": "ی", "ۀ": "ه", "ﭘ": "پ", "ﮐ": "ک", "ﯽ": "ی",
        "ﺎ": "ا", "ﺑ": "ب", "ﺘ": "ت", "ﺧ": "خ", "ﺩ": "د", "ﺱ": "س", "ﻀ": "ض", "ﻌ": "ع",
        "ﻟ": "ل", "ﻡ": "م", "ﻢ": "م", "ﻪ": "ه", "ﻮ": "و", 'ﺍ': "ا", 'ة': "ه",
        'ﯾ': "ی", 'ﯿ': "ی", 'ﺒ': "ب", 'ﺖ': "ت", 'ﺪ': "د", 'ﺮ': "ر", 'ﺴ': "س", 'ﺷ': "ش",
        'ﺸ': "ش", 'ﻋ': "ع", 'ﻤ': "م", 'ﻥ': "ن", 'ﻧ': "ن", 'ﻭ': "و", 'ﺭ': "ر", "ﮔ": "گ",

        # "ها": "  ها", "ئ": "ی",

        "a": " ای ", "b": " بی ", "c": " سی ", "d": " دی ", "e": " ایی ", "f": " اف ",
        "g": " جی ", "h": " اچ ", "i": " آی ", "j": " جی ", "k": " کی ", "l": " ال ",
        "m": " ام ", "n": " ان ", "o": " او ", "p": " پی ", "q": " کیو ", "r": " آر ",
        "s": " اس ", "t": " تی ", "u": " یو ", "v": " وی ", "w": " دبلیو ", "x": " اکس ",
        "y": " وای ", "z": " زد ",
        "\u200c": " ", "\u200d": " ", "\u200e": " ", "\u200f": " ", "\ufeff": " ",
    }
    chars_to_ignore_regex = f"""[{"".join(chars_to_ignore)}]"""
    text = batch["sentence"].lower().strip()

    text = _normalizer.normalize(text)
    # multiple_replace
    pattern = "|".join(map(re.escape, chars_to_mapping.keys()))
    text = re.sub(pattern, lambda m: chars_to_mapping[m.group()], str(text))

    # remove_special_characters
    text = re.sub(chars_to_ignore_regex, '', text).lower() + " "

    text = re.sub(" +", " ", text)
    text = text.strip() + " "

    batch["sentence"] = text
    return batch


def fun_fi(batch):
    chars_to_ignore_regex = '[\,\?\.\!\-\;\:\"\“\%\‘\”\�\'\...\…\–\é]'
    batch["sentence"] = re.sub(chars_to_ignore_regex, '', batch["sentence"]).lower()
    return batch


def fun_fr(batch):
    chars_to_ignore_regex = '[\,\?\.\!\-\;\:\"\“\%\‘\”\�]'
    batch["sentence"] = re.sub(chars_to_ignore_regex, '', batch["sentence"]).lower()
    return batch


def fun_fy_NL(batch):
    chars_to_ignore_regex = '[\,\?\.\!\-\;\:\"\'\“\%\‘\”]'
    batch["sentence"] = re.sub(chars_to_ignore_regex, '', batch["sentence"]).lower()
    return batch


def fun_ga_IE(batch):
    chars_to_ignore_regex = '[\,\?\.\!\-\;\:\"\“\%\‘\”\�\’\–\(\)]'
    batch["sentence"] = re.sub(chars_to_ignore_regex, '', batch["sentence"]).lower()
    return batch


def fun_hi(batch):
    chars_to_ignore_regex = '[\,\?\.\!\-\;\:\"\“\%\‘\”\�\।]'
    batch["sentence"] = re.sub(chars_to_ignore_regex, '', batch["sentence"])
    return batch


def fun_hsb(batch):
    chars_to_ignore_regex = '[\\\\\\\\\\\\\\\\,\\\\\\\\\\\\\\\\?\\\\\\\\\\\\\\\\.\\\\\\\\\\\\\\\\!\\\\\\\\\\\\\\\\-\\\\\\\\\\\\\\\\;\\\\\\\\\\\\\\\\:\\\\\\\\\\\\\\\\"\\\\\\\\\\\\\\\\“\\\\\\\\\\\\\\\\%\\\\\\\\\\\\\\\\‘\\\\\\\\\\\\\\\\”\\\\\\\\\\\\\\\\�\\\\\\\\\\\\\\\\–\\\\\\\\\\\\\\\\—\\\\\\\\\\\\\\\\¬\\\\\\\\\\\\\\\\⅛]'
    batch["sentence"] = re.sub(chars_to_ignore_regex, '', batch["sentence"]).lower()
    return batch


def fun_hu(batch):
    CHARS_TO_IGNORE = [",", "?", "¿", ".", "!", "¡", ";", ":", '""', "%", '"', "�", "ʿ", "·", "჻", "~", "՞",
                       "؟", "،", "।", "॥", "«", "»", "„", "“", "”", "「", "」", "‘", "’", "《", "》", "(", ")", "[", "]",
                       "=", "`", "_", "+", "<", ">", "…", "–", "°", "´", "ʾ", "‹", "›", "©", "®", "—", "→", "。"]
    chars_to_ignore_regex = f"[{re.escape(''.join(CHARS_TO_IGNORE))}]"
    batch["sentence"] = re.sub(chars_to_ignore_regex, "", batch["sentence"]).upper()
    return batch


def fun_ia(batch):
    chars_to_ignore_regex = '[\.\,\!\?\-\"\:\;\'\“\”]'
    batch["sentence"] = re.sub(chars_to_ignore_regex, '', batch["sentence"]).lower()
    return batch


def fun_id(batch):
    chars_to_ignore_regex = '[\,\?\.\!\-\;\:\"\“\%\‘\'\”\�]'
    batch["sentence"] = re.sub(chars_to_ignore_regex, '', batch["sentence"]).lower()
    return batch


def fun_it(batch):
    chars_to_remove = [",", "?", ".", "!", "-", ";", ":", '""', "%", '"', "�", 'ʿ', '“', '”', '(', '=', '`', '_', '+',
                       '«', '<', '>', '~', '…', '«', '»', '–', '\[', '\]', '°', '̇', '´', 'ʾ', '„', '̇', '̇', '̇',
                       '¡']  # All extra characters

    chars_to_remove_regex = f'[{"".join(chars_to_remove)}]'
    batch["sentence"] = re.sub(chars_to_remove_regex, '', batch["sentence"]).lower().replace('‘', "'").replace('ʻ',
                                                                                                               "'").replace(
        'ʼ', "'").replace('’', "'").replace('ʹ', "''").replace('̇', '')

    allowed_characters = [
        " ",
        "'",
        'a',
        'b',
        'c',
        'd',
        'e',
        'f',
        'g',
        'h',
        'i',
        'j',
        'k',
        'l',
        'm',
        'n',
        'o',
        'p',
        'q',
        'r',
        's',
        't',
        'u',
        'v',
        'w',
        'x',
        'y',
        'z',
        'à',
        'á',
        'è',
        'é',
        'ì',
        'í',
        'ò',
        'ó',
        'ù',
        'ú',
    ]

    def remove_accents(input_str):
        if input_str in allowed_characters:
            return input_str
        if input_str == 'ø':
            return 'o'
        elif input_str == 'ß' or input_str == 'ß':
            return 'b'
        elif input_str == 'ё':
            return 'e'
        elif input_str == 'đ':
            return 'd'
        nfkd_form = unicodedata.normalize('NFKD', input_str)
        only_ascii = nfkd_form.encode('ASCII', 'ignore').decode()

        if only_ascii is None or only_ascii == '':
            return input_str
        else:
            return only_ascii

    def fix_accents(sentence):
        new_sentence = ''
        for char in sentence:
            new_sentence += remove_accents(char)
        return new_sentence

    batch["sentence"] = fix_accents(batch["sentence"])
    return batch


def fun_ja(batch):
    CHARS_TO_IGNORE = [",", "?", "¿", ".", "!", "¡", ";", ":", '""', "%", '"', "�", "ʿ", "·", "჻", "~", "՞",
                       "؟", "،", "।", "॥", "«", "»", "„", "“", "”", "「", "」", "‘", "’", "《", "》", "(", ")", "[", "]",
                       "=", "`", "_", "+", "<", ">", "…", "–", "°", "´", "ʾ", "‹", "›", "©", "®", "—", "→", "。"]
    chars_to_ignore_regex = f"[{re.escape(''.join(CHARS_TO_IGNORE))}]"
    batch["sentence"] = re.sub(chars_to_ignore_regex, "", batch["sentence"]).upper().strip()
    return batch


def fun_ka(batch):
    chars_to_ignore_regex = '[\\\\\\\\,\\\\\\\\?\\\\\\\\.\\\\\\\\!\\\\\\\\-\\\\\\\\;\\\\\\\\:\\\\\\\\"\\\\\\\\“]'
    batch["sentence"] = re.sub(chars_to_ignore_regex, '', batch["sentence"]).lower()
    return batch


def fun_ky(batch):
    chars_to_ignore = [",", "?", ".", "!", "-", ";", ":", "—", "–", "”"]
    chars_to_ignore_regex = f'[{"".join(chars_to_ignore)}]'
    batch["sentence"] = re.sub(chars_to_ignore_regex, '', batch["sentence"]).lower()
    return batch


def fun_lg(batch):
    chars_to_ignore_regex = '[\[\],?.!;:%"“”(){}‟ˮʺ″«»/…‽�–]'
    batch["sentence"] = re.sub(r'(\w)[‘’´`](\w)', r"\1'\2", batch["sentence"])
    batch["sentence"] = re.sub(chars_to_ignore_regex, "", batch["sentence"]).lower().strip()
    batch["sentence"] = re.sub(r"(-|' | '|  +)", " ", batch["sentence"])
    batch["sentence"] = unidecode.unidecode(batch["sentence"]).strip()
    return batch


def fun_lt(batch):
    chars_to_ignore_regex = '[\,\?\.\!\-\;\:\"\“\%\‘\”\�]'
    batch["sentence"] = re.sub(chars_to_ignore_regex, '', batch["sentence"]).lower()
    return batch


def fun_lv(batch):
    sent = batch["sentence"].lower()
    # replace non-alpha characters with space
    sent = "".join(ch if ch.isalpha() else " " for ch in sent)
    # remove repeated spaces
    sent = " ".join(sent.split())
    batch["sentence"] = sent
    return batch


def fun_mn(batch):
    sent = batch["sentence"].lower()
    # replace non-alpha characters with space
    sent = "".join(ch if ch.isalpha() else " " for ch in sent)
    # remove repeated spaces
    sent = " ".join(sent.split())
    batch["sentence"] = sent
    return batch


def fun_mt(batch):
    chars_to_ignore_regex = '[\\,\\?\\.\\!\\-\\;\\:\\"\\“\\%\\‘\\”\\�]'
    batch["sentence"] = re.sub(chars_to_ignore_regex, '', batch["sentence"]).lower() + " "
    return batch


def fun_nl(batch):
    chars_to_ignore_regex = '[\,\?\.\!\-\;\:\"\'\“\%\‘\”]'
    batch["sentence"] = re.sub(chars_to_ignore_regex, '', batch["sentence"]).lower()
    return batch


def fun_or(batch):
    chars_to_ignore_regex = '[\,\?\.\!\-\;\:\"\“\%\‘\”\�\–\…\'\_\’\।\|]'
    batch["sentence"] = re.sub(chars_to_ignore_regex, '', batch["sentence"]).lower()
    return batch


def fun_pa_IN(batch):
    chars_to_ignore_regex = '[\,\?\.\!\-\;\:\"\“\%\‘\”\�\’\–\(\)]'
    batch["sentence"] = re.sub(chars_to_ignore_regex, '', batch["sentence"]).lower()
    return batch


def fun_pl(batch):
    chars_to_ignore_regex = '[\—\…\,\?\.\!\-\;\:\"\“\„\%\‘\”\�\«\»\'\’]'
    batch["sentence"] = re.sub(chars_to_ignore_regex, '', batch["sentence"]).lower()
    return batch


def fun_pt(batch):
    CHARS_TO_IGNORE = [",", "?", "¿", ".", "!", "¡", ";", ":", '""', "%", '"', "�", "ʿ", "·", "჻", "~", "՞",
                       "؟", "،", "।", "॥", "«", "»", "„", "“", "”", "「", "」", "‘", "’", "《", "》", "(", ")", "[", "]",
                       "=", "`", "_", "+", "<", ">", "…", "–", "°", "´", "ʾ", "‹", "›", "©", "®", "—", "→", "。"]
    chars_to_ignore_regex = f"[{re.escape(''.join(CHARS_TO_IGNORE))}]"
    batch["sentence"] = re.sub(chars_to_ignore_regex, "", batch["sentence"]).upper()
    return batch


def fun_rm_sursilv(batch):
    chars_to_ignore_regex = '[\\,\\?\\.\\!\\-\\;\\:\\"\\“\\%\\‘\\”\\�\\…\\«\\»\\–]'
    batch["sentence"] = re.sub(chars_to_ignore_regex, '', batch["sentence"]).lower()
    return batch


def fun_rm_vallader(batch):
    chars_to_ignore_regex = '[\,\?\.\!\-\;\:\"\“\%\”\„\–\…\«\»]'
    batch["sentence"] = re.sub('’ ', ' ', batch["sentence"])
    batch["sentence"] = re.sub(' ‘', ' ', batch["sentence"])
    batch["sentence"] = re.sub('’|‘', '\'', batch["sentence"])
    batch["sentence"] = re.sub(chars_to_ignore_regex, '', batch["sentence"]).lower()
    return batch


def fun_ro(batch):
    sent = batch["sentence"].lower()
    # replace non-alpha characters with space
    sent = "".join(ch if ch.isalpha() else " " for ch in sent)
    # remove repeated spaces
    sent = " ".join(sent.split())
    batch["sentence"] = sent
    return batch


def fun_ru(batch):
    sent = batch["sentence"].lower()
    # these letters are considered equivalent in written Russian
    sent = sent.replace('ё', 'е')
    # replace non-alpha characters with space
    sent = "".join(ch if ch.isalpha() else " " for ch in sent)
    # remove repeated spaces
    sent = " ".join(sent.split())
    batch["sentence"] = sent
    return batch


def fun_sah(batch):
    sent = batch["sentence"].lower()
    # replace non-alpha characters with space
    sent = "".join(ch if ch.isalpha() else " " for ch in sent)
    # remove repeated spaces
    sent = " ".join(sent.split())
    batch["sentence"] = sent
    return batch


def fun_sl(batch):
    sent = batch["sentence"].lower()
    # replace non-alpha characters with space
    sent = "".join(ch if ch.isalpha() else " " for ch in sent)
    # remove repeated spaces
    sent = " ".join(sent.split())
    batch["sentence"] = sent
    return batch


def fun_sv_SE(batch):
    chars_to_ignore_regex = '[,?.!\\-;:"“]'
    batch["sentence"] = re.sub(chars_to_ignore_regex, '', batch["sentence"]).lower()
    return batch


def fun_ta(batch):
    chars_to_ignore_regex = '[\,\?\.\!\-\;\:\"\“\%\‘\”\�\’\–\(\)]'
    batch["sentence"] = re.sub(chars_to_ignore_regex, '', batch["sentence"]).lower()
    return batch


def fun_th(batch):
    chars_to_ignore_regex = '[\,\?\.\!\-\;\:\"\â€œ]'
    batch["sentence"] = re.sub(chars_to_ignore_regex, '', batch["sentence"]).lower()
    return batch


def fun_tr(batch):
    chars_to_ignore_regex = '[\,\?\.\!\-\;\'\:\"\“\%\‘\”\�]'
    batch["sentence"] = re.sub(chars_to_ignore_regex, '', batch["sentence"]).lower()
    return batch


def fun_tt(batch):
    sent = batch["sentence"].lower()
    # 'ё' is equivalent to 'е'
    sent = sent.replace('ё', 'е')
    # replace non-alpha characters with space
    sent = "".join(ch if ch.isalpha() else " " for ch in sent)
    # remove repeated spaces
    batch["sentence"] = " ".join(sent.split())
    return batch


def fun_uk(batch):
    chars_to_ignore = [",", "?", ".", "!", "-", ";", ":", '""', "%", "'", '"', "�", '«', '»', '—', '…', '(', ')', '*',
                       '”', '“']
    chars_to_ignore_regex = f'[{"".join(chars_to_ignore)}]'
    batch["sentence"] = re.sub(re.compile("['`]"), '’', batch['sentence'])
    batch["sentence"] = re.sub(re.compile(chars_to_ignore_regex), '', batch["sentence"]).lower().strip()
    batch["sentence"] = re.sub(re.compile('i'), 'і', batch['sentence'])
    batch["sentence"] = re.sub(re.compile('o'), 'о', batch['sentence'])
    batch["sentence"] = re.sub(re.compile('a'), 'а', batch['sentence'])
    batch["sentence"] = re.sub(re.compile('ы'), 'и', batch['sentence'])
    batch["sentence"] = re.sub(re.compile("–"), '', batch['sentence'])
    batch['sentence'] = re.sub('  ', ' ', batch['sentence'])
    return batch


def fun_vi(batch):
    chars_to_ignore_regex = '[\\\+\@\ǀ\,\?\.\!\-\;\:\"\“\%\‘\”\�]'
    batch["sentence"] = re.sub(chars_to_ignore_regex, '', batch["sentence"]).lower()
    return batch


def fun_zh_CN(batch):
    chars_to_ignore_regex = r"[¥•＂＃＄％＆＇（）＊＋，－／：；＜＝＞＠［＼］＾＿｀｛｜｝～｟｠｢｣､　、〃〈〉《》「」『』【】〔〕〖〗〘〙〚〛〜〝〞〟〰〾〿–—‘’‛“”„‟…‧﹏﹑﹔·'℃°•·．﹑︰〈〉─《﹖﹣﹂﹁﹔！？｡。＂＃＄％＆＇（）＊＋，﹐－／：；＜＝＞＠［＼］＾＿｀｛｜｝～｟｠｢｣､、〃》「」『』【】〔〕〖〗〘〙〚〛〜〝〞〟〰〾〿–—‘’‛“”„‟…‧﹏.．!\"#$%&()*+,\-.\:;<=>?@\[\]\\\/^_`{|}~]"
    batch["sentence"] = re.sub(chars_to_ignore_regex, '', batch["sentence"]).lower()
    return batch


def fun_zh_HK(batch):
    chars_to_ignore_regex = r"[¥•＂＃＄％＆＇（）＊＋，－／：；＜＝＞＠［＼］＾＿｀｛｜｝～｟｠｢｣､　、〃〈〉《》「」『』【】〔〕〖〗〘〙〚〛〜〝〞〟〰〾〿–—‘’‛“”„‟…‧﹏﹑﹔·'℃°•·．﹑︰〈〉─《﹖﹣﹂﹁﹔！？｡。＂＃＄％＆＇（）＊＋，﹐－／：；＜＝＞＠［＼］＾＿｀｛｜｝～｟｠｢｣､、〃》「」『』【】〔〕〖〗〘〙〚〛〜〝〞〟〰〾〿–—‘’‛“”„‟…‧﹏.．!\"#$%&()*+,\-.\:;<=>?@\[\]\\\/^_`{|}~]"
    batch["sentence"] = re.sub(chars_to_ignore_regex, '', batch["sentence"]).lower()
    return batch


def fun_zh_TW(batch):
    chars_to_ignore_regex = r"[¥•＂＃＄％＆＇（）＊＋，－／：；＜＝＞＠［＼］＾＿｀｛｜｝～｟｠｢｣､　、〃〈〉《》「」『』【】〔〕〖〗〘〙〚〛〜〝〞〟〰〾〿–—‘’‛“”„‟…‧﹏﹑﹔·'℃°•·．﹑︰〈〉─《﹖﹣﹂﹁﹔！？｡。＂＃＄％＆＇（）＊＋，﹐－／：；＜＝＞＠［＼］＾＿｀｛｜｝～｟｠｢｣､、〃》「」『』【】〔〕〖〗〘〙〚〛〜〝〞〟〰〾〿–—‘’‛“”„‟…‧﹏.．!\"#$%&()*+,\-.\:;<=>?@\[\]\\\/^_`{|}~]"
    batch["sentence"] = re.sub(chars_to_ignore_regex, '', batch["sentence"]).lower()
    return batch
