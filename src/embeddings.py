# -*- coding: utf-8 -*-
"""토큰 임베딩 + 위치 임베딩 과제 템플릿."""

import torch
import torch.nn as nn


class InputEmbedding(nn.Module):
    """
    token ID를 Transformer 입력 벡터로 바꿉니다.

    구현할 구조:
    - token embedding: nn.Embedding(vocab_size, emb_dim)
    - position embedding: nn.Embedding(context_length, emb_dim)
    - token embedding + position embedding
    - dropout
    """

    def __init__(
        self,
        vocab_size: int,
        emb_dim: int,
        context_length: int,
        drop_rate: float = 0.1,
    ):
        super().__init__()
        self.emb_dim = emb_dim
        self.context_length = context_length
        # TODO: token_embedding, position_embedding, dropout을 정의하세요.
        self.token_embedding_layer = torch.nn.Embedding(vocab_size, emb_dim) # __init__에서 지역 변수로만 만들면 forward 아래 함수에서 가져다 쓸 수 없으므로 self. 형태로 저장해야 함 
        self.pos_embedding_layer = torch.nn.Embedding(context_length, emb_dim)
        self.drop_out_layer = torch.nn.Dropout(drop_rate)

        # nn.Dropout은 drop out layer 만드는 클래스임 



    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        TODO: token embedding과 position embedding을 더한 뒤 dropout을 적용합니다.

        Args:
            x: (batch_size, seq_len) token IDs

        Returns:
            (batch_size, seq_len, emb_dim)
        """
        # 입력값들 들어오면 위에서 구한 가중치들 적용해줌 
        # dropout은 뭐야? 차원을 dropout하나? -> 차원 자체를 없애는 건 아니고 일부를 차원의 값을 0으로 만드는 것. 나머지 값은 1 / (1 - p)로 스케일돼서 평균 크기는 유지됨 
        # 평가? evaluation 모드에서는 dropout이 꺼지고 아무 값도 0으로 만들지 않음 

        token_embeddings = self.token_embedding_layer(x)
        # pos_embeddings = self.pos_embedding_layer(torch.arange(self.context_length))
        pos_embeddings = self.pos_embedding_layer(torch.arange(x.shape[1]))
        # pos_embeddings = pos_embeddings.unsqueeze(0)


        # 두 값을 더한 뒤 nn.Dropout을 적용
        input_embeddings = token_embeddings + pos_embeddings
        result_embeddings = self.drop_out_layer(input_embeddings) # 이렇게 사용해야됨 

        return result_embeddings

        # batch_size = x.shape[0]
        # max_length = x.shape[1]
        # max_length = self.context_length
        # dataloader = create_dataloader(raw_text, batch_size, max_length=max_length, stride=max_length, shuffle=False)
        # data_iter = iter(dataloader)
        # inputs, targets = next(data_iter)