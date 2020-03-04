import hypothesis
import hypothesis.strategies as st

import horse.types


words = st.integers(0, horse.types.MAX_WORD)


@hypothesis.given(words)
def test_nibbles_returns_nibbles(word):
    nibbles = horse.types.word_to_nibbles(word)
    for nibble in nibbles:
        assert 0 <= nibble <= horse.types.MAX_NIBBLE, nibbles


@hypothesis.given(words)
def test_nibbles_there_and_back_again(word):
    there = horse.types.word_to_nibbles(word)
    back_again = horse.types.nibbles_to_word(there)
    assert word == back_again, there
