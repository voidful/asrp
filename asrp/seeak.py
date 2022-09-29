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
        print("pre calculating homophone character")
        self.homochar = defaultdict(list)
        for char, cins in self.char_to_cin_dict.items():
            for cin in cins.get(cin_source, []):
                self.homochar[char].extend(self.cin_to_char_dict.get(cin, []))

    def get(self, char):
        return self.homochar[char]

    def get_phone(self, char):
        return self.char_to_cin_dict.get(char, {}).get(self.cin_source, [])
