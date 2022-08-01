from collections import defaultdict

import nlp2

from pathlib import Path


class Seeak:
    def __init__(self, cin_source='jyutping.cin'):
        cin_table_path = nlp2.download_file('https://raw.githubusercontent.com/voidful/asrp/main/data/cin_table.json',
                                            f'{Path(__file__).parent}')
        self.char_to_cin_dict = nlp2.read_json(cin_table_path)
        self.cin_to_char_dict = defaultdict(list)
        self.cin_source = cin_source
        for k, v in self.char_to_cin_dict.items():
            if cin_source in v:
                self.cin_to_char_dict[v[cin_source]].append(k)

    def get(self, char):
        if char in self.char_to_cin_dict:
            char_to_cin = self.char_to_cin_dict[char]
            if self.cin_source in char_to_cin and char_to_cin[self.cin_source] in self.cin_to_char_dict:
                return self.cin_to_char_dict[char_to_cin[self.cin_source]]
        return []
