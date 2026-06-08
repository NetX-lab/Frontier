from frontier.types.base_int_enum import BaseIntEnum


class ClusterType(BaseIntEnum):
    MONOLITHIC = 1      # Original monolithic architecture
    PREFILL = 2         # Prefill-dedicated cluster
    DECODE_ATTN = 3     # Decode Attention-dedicated cluster (PD+AF disaggregation)
    DECODE_FFN = 4      # Decode FFN-dedicated cluster (PD+AF disaggregation)
    DECODE = 5          # Unified Decode cluster (PD disaggregation, Attention + FFN)
    TRANS = 6           # Transfer-dedicated cluster (e.g., interconnect/transfer service)
