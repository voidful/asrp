from collections import defaultdict
from pathlib import Path

import nlp2


class Seeak:
    def __init__(self, cin_source='jyutping.cin'):
        cin_table_path = nlp2.download_file('https://raw.githubusercontent.com/voidful/asrp/main/data/cin_table.json',
                                            f'{Path(__file__).parent}')
        self.char_to_cin_dict = nlp2.read_json(cin_table_path)
        self.cin_to_char_dict = defaultdict(list)
        self.cin_source = cin_source
        for k, v in self.char_to_cin_dict.items():
            if cin_source in v:
                for cin in v[cin_source]:
                    self.cin_to_char_dict[cin].append(k)

    def get(self, char):
        result_char = []
        if char in self.char_to_cin_dict:
            char_to_cin = self.char_to_cin_dict[char]
            if self.cin_source in char_to_cin:
                for cin in char_to_cin[self.cin_source]:
                    if cin in self.cin_to_char_dict:
                        result_char.extend(self.cin_to_char_dict[cin])
        return result_char

    def get_phone(self, char):
        if char is not None and char in self.char_to_cin_dict and self.cin_source in self.char_to_cin_dict[char]:
            return self.char_to_cin_dict[char][self.cin_source]
        else:
            return []
