# -*- coding: utf-8 -*-
"""GPT 사전 학습용 Dataset/DataLoader 과제 템플릿."""

import torch
from torch.utils.data import DataLoader, Dataset
import math
import tiktoken


class GPTDataset(Dataset):
    """
    token ID 리스트를 다음 토큰 예측용 input/target 쌍으로 자릅니다.

    예: token_ids=[10, 11, 12, 13], context_length=3
    - input:  [10, 11, 12]
    - target: [11, 12, 13]
    """

    def __init__(
        self,
        token_ids: list[int],
        context_length: int,
        stride: int | None = None,
    ):
        self.token_ids = token_ids
        self.context_length = context_length
        self.stride = stride if stride is not None else context_length
        # TODO: 만들 수 있는 학습 샘플 개수를 self._length에 저장하세요.
        # token_ids를 stride 칸 씩 이동하면서 context_length 개씩 묶어
        # self.length = math.ceil((len(self.token_ids) - self.context_length + 1) / self.stride)
        self.length = max((len(self.token_ids) - self.context_length - 1) // self.stride + 1, 0) # 왜이거야 -> 음수? 

    def __len__(self) -> int:
        """TODO: 전체 샘플 개수를 반환합니다."""
        return self.length

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        """
        TODO: idx번째 input_ids와 target_ids를 LongTensor로 반환합니다.

        Returns:
            input_ids: (context_length,)
            target_ids: (context_length,)
        """

        input_ids = self.token_ids[idx * self.stride : idx * self.stride + self.context_length] # 끝에 +1 해줘야되나? 
        target_ids = self.token_ids[idx * self.stride + 1 : idx * self.stride + self.context_length + 1]  

        # 이걸 Longtensor로 반환? 
        input_ids = torch.tensor(input_ids, dtype=torch.long)
        target_ids = torch.tensor(target_ids, dtype=torch.long)

        # return input_ids[idx], target_ids[idx]
        return input_ids, target_ids


def create_dataloader(
    token_ids: list[int],
    context_length: int,
    batch_size: int = 8,
    stride: int | None = None,
    drop_last: bool = False,
    shuffle: bool = True,
    num_workers: int = 0,
) -> DataLoader:
    """TODO: GPTDataset을 만들고 torch.utils.data.DataLoader로 감싸 반환합니다."""
    
    # batch size가 뭐냐 -> 데이터 8개 선택 
    # drop last가 뭐냐 -> 배치 사이즈보다 작을 경우 훈련 손실이 갑자기 높아지는 것을 피하기 위해 마지막 배치 삭제
    # shuffle 을 어떻게 하냐 -> 배치 데이터 생성할 때 에포크마다 랜덤하게 샘플을 섞음 
    # num_workers 가 뭐냐 -> 전처리에 사용할 CPU 프로세서 개수 

    # 토크나이저를 초기화합니다.
    tokenizer = tiktoken.get_encoding("gpt2")

    # 데이터셋을 만듭니다.
    dataset = GPTDataset(token_ids, context_length, stride)

    # 데이터 로더를 만듭니다.
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        drop_last=drop_last,
        num_workers=num_workers
    )

    return dataloader









    # if shuffle:
    #     shuffle을 해
    #     if 전체 가능한 샘플 개수인 self.length 가 batch_size 보다 작으면 drop last 적용을 해야겟지
    #     else 
    #     token_ids 안에서 이제 context_length 길이를 stride 씩 이동하면서 batch_size 개 만큼 골라

    # else:
    #     token_ids 에서 그대로 잘라

    # data_ids = []


    # if dataset.length > batch_size:
    #     if shuffle:
    #         shuffle = shuffle
    #     for bat_i in range(batch_size - 1):
    #         input_ids, target_ids = dataset[bat_i]
    #         data_ids[0].append(input_ids)
    #         data_ids[1].append(target_ids)

    # else:
    #     if shuffle:
    #         shuffle = shuffle
    #     for bat_i in range(batch_size):
    #         input_ids, target_ids = dataset[bat_i]
    #         data_ids[0].append(input_ids)
    #         data_ids[1].append(target_ids)


                
                



    
    



    
    


