# -*- coding: utf-8 -*-
"""
UTF-8 byte-level BPE 토크나이저 과제 템플릿.

외부 tokenizer 라이브러리 없이 BPE(Byte Pair Encoding)를 직접 구현합니다.
한국어 NSMC 리뷰를 다루므로 문자열을 글자/공백 단위로 먼저 자르지 말고,
항상 `text.encode("utf-8")`로 byte ID 시퀀스를 만든 뒤 merge를 적용하세요.
"""

from pathlib import Path
import re
import json



PAD_TOKEN = "<pad>"
UNK_TOKEN = "<unk>"
BOS_TOKEN = "<bos>"
EOS_TOKEN = "<eos>"

SPECIAL_TOKENS = [PAD_TOKEN, UNK_TOKEN, BOS_TOKEN, EOS_TOKEN]
SPECIAL_IDS = {token: idx for idx, token in enumerate(SPECIAL_TOKENS)}
BYTE_OFFSET = len(SPECIAL_TOKENS)
NUM_BYTES = 256

class BPETokenizer:
    """
    UTF-8 byte-level BPE 토크나이저.

    권장 ID 배치:
    - 0~3: <pad>, <unk>, <bos>, <eos>
    - 4~259: 원본 byte 0~255
    - 260 이상: BPE merge로 생성한 토큰
    """

    def __init__(self, vocab_size: int):
        self.vocab_size = vocab_size
        self.id_to_token = {}
        self.token_to_id = {}
        self.merges = []
        self._init_special_tokens() # 이렇게 해도 돼? 

    def _init_special_tokens(self):
        """
        TODO:
        1. 특수 토큰 4개를 고정 ID 0~3에 등록합니다.
        2. byte 0~255를 ID 4~259에 bytes([byte_value]) 형태로 등록합니다.
        """

        self.id_to_token = {}
        self.token_to_id = {}

        for token, token_id in SPECIAL_IDS.items():
            self.id_to_token[token_id] = token
            self.token_to_id[token] = token_id
        
        for byte_value in range(NUM_BYTES):
            token_id = BYTE_OFFSET + byte_value
            token = bytes([byte_value])
            self.id_to_token[token_id] = token
            self.token_to_id[token] = token_id

    def get_pad_id(self):
        """padding 토큰 ID."""
        return SPECIAL_IDS[PAD_TOKEN]

    def get_unk_id(self):
        """unknown 토큰 ID."""
        return SPECIAL_IDS[UNK_TOKEN]

    def get_bos_id(self):
        """문장 시작 토큰 ID."""
        return SPECIAL_IDS[BOS_TOKEN]

    def get_eos_id(self):
        """문장 끝 토큰 ID."""
        return SPECIAL_IDS[EOS_TOKEN]

    def train(self, corpus: str):
        """
        TODO: 코퍼스에서 BPE merge rule과 vocabulary를 학습합니다.

        구현 힌트:
        - `corpus.encode("utf-8")`로 byte ID 시퀀스를 만듭니다.
        - 가장 자주 등장하는 이웃 token pair를 찾습니다.
        - 새 token ID를 만들고, 시퀀스의 해당 pair를 새 ID로 치환합니다.
        - `self.merges`, `self.id_to_token`, `self.token_to_id`를 갱신합니다.
        """

        corpus_id =  [id + 4 for id in corpus.encode("utf-8")]

        while (len(self.id_to_token) < self.vocab_size and len(corpus_id) > 1): #  len(corpus_id) > 1???
            frequency = {}

            for i in range(len(corpus_id) - 1):
                pair = (corpus_id[i], corpus_id[i + 1])
                frequency[pair] = frequency.get(pair, 0) + 1 # 이 pair 가 나온 횟수를 1 늘림. pair 가 이미 딕셔너리에 있으면 그 값을 가져오고 없으면 기본값 0을 줌 

            best_pair = max(frequency, key = lambda pair : frequency[pair])

            
            # for 문 쓰면 점점 배열 길이는 줄어드는데 처음 길이만큼 돌려고 해서 오류뜸 
            i = 0
            while (i < len(corpus_id) - 1):
                if ((corpus_id[i], corpus_id[i+1]) == best_pair):
                    corpus_id[i] = len(self.id_to_token)
                    del corpus_id[i + 1]
                i += 1

            id = len(self.id_to_token)
            self.merges.append(best_pair)
            self.id_to_token[id] = best_pair
            self.token_to_id[best_pair] = id



    def save(self, path: str | Path):
        """
        TODO: vocabulary와 merge rule을 JSON 파일로 저장합니다.

        bytes와 tuple은 JSON에 바로 저장할 수 없으므로 type 정보를 함께 저장하세요.
        """
        data = {}
        merges_data = []
        
        for tup in self.merges:
            merges_data.append({
                "type" : "tuple",
                "value" : list(tup)
            })
        
            
        data["merges"] = merges_data
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)

    def load(self, path: str | Path):
        """
        TODO: save()로 저장한 JSON 파일을 읽어 vocabulary와 merge rule을 복원합니다.
        """
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)

        self._init_special_tokens()
        self.merges = []

        data_list = data["merges"]

        new_id = len(self.id_to_token)
        for dic in data_list:
            token = tuple(dic["value"])
            self.id_to_token[new_id] = token
            self.token_to_id[token] = new_id
            self.merges.append(token)
            new_id += 1

    def encode(self, text: str, add_bos_eos: bool = False) -> list[int]:
        """
        TODO: 문자열을 token ID 리스트로 변환합니다.

        구현 힌트:
        - 먼저 UTF-8 byte ID 리스트를 만듭니다.
        - train/load에서 얻은 merge rule을 학습 순서대로 적용합니다.
        - add_bos_eos=True이면 앞뒤에 bos/eos ID를 붙입니다. --> 어디 앞뒤? 
        """

        ids = [id + 4 for id in text.encode("utf-8")]


        # unk 나올 수 있지 않나???? 
        # 위에 잇는걸 이제 merge rule 써서 합쳐 
        for merge_id in range(len(self.merges)):
            i = 0
            while(i < len((ids))):
                if self.merges[merge_id] == (ids[i], ids[i+1]):
                    ids[i] = merge_id + 260
                    del ids[i + 1]
                i += 1

        if add_bos_eos:
            ids = [self.get_bos_id()] + ids + [self.get_eos_id()]

        return ids 

    def decode(self, ids: list[int], skip_special: bool = True) -> str:
        """
        TODO: token ID 리스트를 문자열로 복원합니다.

        주의:
        - merge token은 원본 byte token까지 재귀적으로 펼칩니다.
        - byte를 하나씩 decode하지 말고, 마지막에 `bytes(...).decode("utf-8")`를 한 번만 호출합니다.
        """
        
        IDs = []
        
        def divide(id):
            if 0 <= id and id <= 4:
                return 
            elif id < 260:
                IDs.append(id)
            else:
                (i1, i2) = self.id_to_token[id]
                divide(i1)
                divide(i2)

        for id in ids:
            divide(id)

        tokens = []
        for id in IDs:
            tokens.append(self.id_to_token[id])

        # return bytes(tokens).decode("utf-8")
        return b"".join(tokens).decode("utf-8") # b 떼고 숫자만 가져와라 바이트에서 아하아하 