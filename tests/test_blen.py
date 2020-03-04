import collections

import hypothesis
import hypothesis.strategies as st

import horse.blen
import horse.types


words = st.integers(0, horse.types.MAX_WORD)
bytes = st.integers(0, horse.types.MAX_BYTE)
nibbles = st.integers(0, horse.types.MAX_NIBBLE)

MIN_SIGNED_INT = -(1 << horse.types.WORD_N_BITS - 1)
MAX_SIGNED_INT = (1 << horse.types.WORD_N_BITS - 1) - 1
in_range_signed_integers = st.integers(MIN_SIGNED_INT, MAX_SIGNED_INT)

binary_operations = st.sampled_from(tuple(horse.blen.BINARY_OPERATIONS.values()))
unary_operations = st.sampled_from(tuple(horse.blen.UNARY_OPERATIONS.values()))

machines = st.builds(
    horse.blen.Machine,
    registers=st.fixed_dictionaries(
        {register: words for register in horse.blen.Register}
    ),
)
memories = st.dictionaries(words, words).map(lambda d: collections.defaultdict(int, d))


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


@hypothesis.given(
    st.sampled_from((horse.blen.test_equal, horse.blen.test_greater_than)),
    words,
    words,
)
def test_comparison_functions_returns_bool_values(func, operand0, operand1):
    result = func(operand0, operand1)
    assert result == horse.types.TRUE or result == horse.types.FALSE


@hypothesis.given(words)
def test_convert_to_bool_returns_bool_values(operand):
    result = horse.blen.convert_to_bool(operand)
    assert result == horse.types.TRUE or result == horse.types.FALSE


@hypothesis.given(in_range_signed_integers)
def test_increment_non_overflow(operand):
    result = horse.blen.increment(horse.blen.signed_integer_to_word(operand))
    true_result = operand + 1
    if MIN_SIGNED_INT <= true_result <= MAX_SIGNED_INT:
        assert result == horse.blen.signed_integer_to_word(true_result)


@hypothesis.given(in_range_signed_integers)
def test_decrement_non_overflow(operand):
    result = horse.blen.decrement(horse.blen.signed_integer_to_word(operand))
    true_result = operand - 1
    if MIN_SIGNED_INT <= true_result <= MAX_SIGNED_INT:
        assert result == horse.blen.signed_integer_to_word(true_result)


@hypothesis.given(words)
def test_parse_defined_for_all_words(word):
    assert isinstance(horse.blen.parse(word), horse.blen.Instruction)


@hypothesis.given(machines, memories)
def test_register_zero_always_zero(machine, memory):
    machine.tick(memory)
    assert machine.registers[horse.blen.Register.ZERO_REGISTER] == 0
