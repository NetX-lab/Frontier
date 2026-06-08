from frontier.types.base_int_enum import BaseIntEnum


class NodeSKUType(BaseIntEnum):
    A40_PAIRWISE_NVLINK = 1
    A100_PAIRWISE_NVLINK = 2
    H100_PAIRWISE_NVLINK = 3
    A100_DGX = 4
    H100_DGX = 5
    H200_DGX = 6
    A800_DGX = 7
    H800_DGX = 8
    H20_DGX = 9
