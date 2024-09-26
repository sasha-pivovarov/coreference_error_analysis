from enum import Enum


class ErrorType(Enum):
    MISALIGNED_SPAN = 0,
    EXTRA_SPAN = 1,
    MISSING_SPAN = 2,
    ENTITY_MERGED = 3,
    ENTITY_SPLIT = 4,
    MISSING_ENTITY = 5,
    EXTRA_ENTITY = 6,