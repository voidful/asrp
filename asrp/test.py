import unittest

import preprocessing


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
