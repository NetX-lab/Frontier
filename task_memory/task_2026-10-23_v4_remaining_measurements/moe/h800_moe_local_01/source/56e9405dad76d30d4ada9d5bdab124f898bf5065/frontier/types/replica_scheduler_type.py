from frontier.types.base_int_enum import BaseIntEnum


class ReplicaSchedulerType(BaseIntEnum):
    FASTER_TRANSFORMER = 1
    ORCA = 2
    SARATHI = 3
    VLLM = 4
    LIGHTLLM = 5
    VLLM_V1 = 6
    SGLANG = 7
    SJ2Q_FASTSERVE_LITE = 8
    SJ2Q_PENALTY_ONLY = 9
    SJ2Q_BOUNDED_CARRYOVER = 10
