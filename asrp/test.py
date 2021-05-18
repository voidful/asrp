import unittest

import preprocessing
import eval


class TestPreprocssing(unittest.TestCase):

    def testfun(self):
        langs = ['ab', 'ar', 'as', 'br', 'ca', 'cnh', 'cs', 'cv', 'cy', 'de', 'dv', 'el', 'en', 'eo', 'es', 'et', 'eu',
                 'fa',
                 'fi', 'fr', 'fy-NL', 'ga-IE', 'hi', 'hsb', 'hu', 'ia', 'id', 'it', 'ja', 'ka', 'kab', 'ky', 'lg', 'lt',
                 'lv',
                 'mn', 'mt', 'nl', 'or', 'pa-IN', 'pl', 'pt', 'rm-sursilv', 'rm-vallader', 'ro', 'ru', 'rw', 'sah',
                 'sl',
                 'sv-SE', 'ta', 'th', 'tr', 'tt', 'uk', 'vi', 'vot', 'zh-CN', 'zh-HK', 'zh-TW']

        count = 0
        total = 0
        for lang in langs:
            total += 1
            try:
                batch = {}
                batch['sentence'] = "hello 你好嗎 図書館にいた事がバレた , , ? . !;: “%”�()«»½…"
                print("fun name", 'fun_' + lang.replace("-", "_"))
                preprocessing_sentence = getattr(preprocessing, 'fun_' + lang.replace("-", "_"))
                print(preprocessing_sentence(batch))
                count += 1
            except BaseException as e:
                print(e)
                pass
        print(count, total)

    def test_regex(self):
        langs = ['ab', 'ar', 'as', 'br', 'ca', 'cnh', 'cs', 'cv', 'cy', 'de', 'dv', 'el', 'en', 'eo', 'es', 'et', 'eu',
                 'fa',
                 'fi', 'fr', 'fy-NL', 'ga-IE', 'hi', 'hsb', 'hu', 'ia', 'id', 'it', 'ja', 'ka', 'kab', 'ky', 'lg', 'lt',
                 'lv',
                 'mn', 'mt', 'nl', 'or', 'pa-IN', 'pl', 'pt', 'rm-sursilv', 'rm-vallader', 'ro', 'ru', 'rw', 'sah',
                 'sl',
                 'sv-SE', 'ta', 'th', 'tr', 'tt', 'uk', 'vi', 'vot', 'zh-CN', 'zh-HK', 'zh-TW']

        count = 0
        total = 0
        re_list = []
        for lang in langs:
            total += 1
            try:
                batch = {}
                batch['sentence'] = "hello 你好嗎 図書館にいた事がバレた , , ? . !;: “%”�()«»½…"
                print("fun name", 'fun_' + lang.replace("-", "_"))
                preprocessing_sentence = getattr(preprocessing, 'fun_' + lang.replace("-", "_"))
                print(preprocessing_sentence(batch))
                if preprocessing_sentence.chars_to_ignore_regex not in re_list:
                    re_list.append(preprocessing_sentence.chars_to_ignore_regex)
                count += 1
            except BaseException as e:
                print(e)
                pass
        generic_re = '|'.join(re_list)
        print(generic_re)
        print(count, total)


class TestEval(unittest.TestCase):
    def testCER(self):
        targets = ['HuggingFace is great!', 'Love Transformers!', 'Let\'s wav2vec!']
        preds = ['HuggingFace is awesome!', 'Transformers is powerful.', 'Let\'s finetune wav2vec!']
        print("chunk size = None, CER: {:2f}".format(100 * eval.chunked_cer(targets, preds, chunk_size=None)))
        print("chunk size = 2, CER: {:2f}".format(100 * eval.chunked_cer(targets, preds, chunk_size=2)))
        print("chunk size = 3, CER: {:2f}".format(100 * eval.chunked_cer(targets, preds, chunk_size=3)))
        print("chunk size = 5, CER: {:2f}".format(100 * eval.chunked_cer(targets, preds, chunk_size=5)))
        print("chunk size = 7, CER: {:2f}".format(100 * eval.chunked_cer(targets, preds, chunk_size=6)))
        print("chunk size = 100, CER: {:2f}".format(100 * eval.chunked_cer(targets, preds, chunk_size=100)))

    def testBoth(self):
        targets = ["hello world", "你好啊"]
        preds = ["hello duck", "您好嗎"]
        print("chunk size = None, WER: {:2f}".format(100 * eval.chunked_wer(targets, preds, chunk_size=None)))
        print("chunk size = None, CER: {:2f}".format(100 * eval.chunked_cer(targets, preds, chunk_size=None)))
