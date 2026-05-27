# -*- coding: utf-8 -*-
"""GPT 사전 학습용 Dataset/DataLoader 과제 템플릿."""

import torch
from torch.utils.data import DataLoader, Dataset


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
        # 만들 수 있는 학습 샘플 개수
        sample_count = (len(token_ids) - context_length - 1) // self.stride + 1
        self._length = max(0, sample_count) # 음수가 나오지 않게 보정

    def __len__(self) -> int:
        """전체 샘플 개수를 반환합니다."""
        return self._length

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        """
        idx번째 input_ids와 target_ids를 LongTensor로 반환합니다.

        Returns:
            input_ids: (context_length,)
            target_ids: (context_length,)
        """

        # 시작과 끝 인덱스 확보
        start = idx * self.stride
        end = start + self.context_length
        
        # 학습용 데이터 샘플링
        input_ids_list = self.token_ids[start:end]
        target_ids_list = self.token_ids[start + 1:end + 1]

        # PyTorch 모델에서 사용할 수 있도록 토큰 ID 리스트를 torch.long(int64) 텐서로 변환
        input_ids = torch.tensor(input_ids_list, dtype=torch.long)
        target_ids = torch.tensor(target_ids_list, dtype=torch.long)
        
        return (input_ids, target_ids)

def create_dataloader(
    token_ids: list[int],
    context_length: int,
    batch_size: int = 8,
    stride: int | None = None,
    drop_last: bool = False,
    shuffle: bool = True,
    num_workers: int = 0,
) -> DataLoader:
    """GPTDataset을 만들고 torch.utils.data.DataLoader로 감싸 반환합니다."""

    # GPTDataset은 전체 샘플 개수를 알려주고, idx번째 샘플로 tensor 쌍을 반환
    dataset = GPTDataset(token_ids, context_length, stride=stride)

    # DataLoader는 Dataset에서 여러 샘플을 꺼내 배치로 묶어 반환
    result = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        drop_last=drop_last,
        num_workers=num_workers,
    )

    return result
