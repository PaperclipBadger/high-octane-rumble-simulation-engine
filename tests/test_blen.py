import hypothesis
import hypothesis.strategies as st

import horse.blen
import horse.types


words = st.integers(0, horse.types.MAX_WORD)
bytes = st.integers(0, horse.types.MAX_BYTE)
nibbles = st.integers(0, horse.types.MAX_NIBBLE)
binary_operations = st.sampled_from(tuple(horse.blen.BINARY_OPERATIONS.values()))
unary_operations = st.sampled_from(tuple(horse.blen.UNARY_OPERATIONS.values()))


@hypothesis.given(words)
def test_signed_integer_encode_decode(word):
    there = horse.blen.word_to_signed_integer(word)
    back_again = horse.blen.signed_integer_to_word(there)
    assert word == back_again, there


@hypothesis.given(binary_operations, words, words)
def test_binary_operations_return_words(func, operand0, operand1):
    assert 0 <= func(operand0, operand1) <= horse.types.MAX_WORD


@hypothesis.given(unary_operations, words)
def test_unary_operations_return_words(func, operand):
    assert 0 <= func(operand) <= horse.types.MAX_WORD
