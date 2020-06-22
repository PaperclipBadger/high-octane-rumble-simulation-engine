import unittest
import hypothesis
import hypothesis.strategies as st

# Add '.' to path so running this file by itself also works
import os, sys
sys.path.append(os.path.realpath('.'))
import horse.types

words = st.integers(0, horse.types.MAX_WORD)

class WordsToNibbles(unittest.TestCase):

    @hypothesis.given(words)
    def test_word_to_nibbles(self, word):
        ''' Nibbles return nibbles '''
        nibbles = horse.types.word_to_nibbles(word)
        for nibble in nibbles:
            assert 0 <= nibble <= horse.types.MAX_NIBBLE, nibbles

    @hypothesis.given(words)
    def test_there_and_back(self, word):
        ''' Nibbles to words and back '''
        there = horse.types.word_to_nibbles(word)
        back_again = horse.types.nibbles_to_word(there)
        assert word == back_again, there

if __name__ == '__main__':
    unittest.main()