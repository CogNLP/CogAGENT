from dataclasses import dataclass
SPECIAL_TOKENS = [
    "<bos>",
    "<eos>",
    "<pad>",
    "<speaker1>",
    "<speaker2>",
    "<subject>",
    "<predicate>",
    "<object>",
    "<triple>",
    "<sep>",
]

ATTR_TO_SPECIAL_TOKEN = {
    "bos_token": "<bos>",
    "eos_token": "<eos>",
    "pad_token": "<pad>",
    "sep_token": "<sep>",
    "additional_special_tokens": [
        "<speaker1>",
        "<speaker2>",
        "<subject>",
        "<predicate>",
        "<object>",
        "<triple>",
    ],
}

@dataclass
class SpecialTokens:
    bos: int
    eos: int
    pad: int
    speaker1: int
    speaker2: int
    subject: int
    predicate: int
    object: int
    triple: int
    sep: int
