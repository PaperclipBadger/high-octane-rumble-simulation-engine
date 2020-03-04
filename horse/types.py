from typing import NewType, Sequence

WORD_N_BITS = 16
MAX_WORD = (1 << WORD_N_BITS) - 1

BYTE_N_BITS = 8
MAX_BYTE = (1 << BYTE_N_BITS) - 1

NIBBLE_N_BITS = 4
MAX_NIBBLE = (1 << NIBBLE_N_BITS) - 1

Word = NewType("Word", int)
Byte = NewType("Byte", int)
Nibble = NewType("Nibble", int)

TRUE = Word(MAX_WORD)
FALSE = Word(0)


def nibbles(word: Word, /) -> Sequence[Nibble]:
    return tuple(Nibble(word << d) for d in range(WORD_N_BITS, 0, -NIBBLE_N_BITS))
