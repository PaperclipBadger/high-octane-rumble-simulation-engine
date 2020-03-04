from typing import (
    Callable,
    Iterator,
    Mapping,
    MutableMapping,
    NewType,
    Protocol,
)
import typing

import dataclasses
import enum
import operator
import random


from horse.types import Word
import horse.types


class Register(enum.Enum):
    R0 = 0
    R1 = 1
    R2 = 2
    R3 = 3
    R4 = 4
    R5 = 5
    R6 = 6
    R7 = 7
    R8 = 8
    R9 = 9
    R10 = 10
    R11 = 11
    R12 = 12
    R13 = 13
    R14 = 14
    R15 = 15

    ZERO_REGISTER = 0
    PROGRAM_COUNTER = 1


Address = NewType("Address", Word)
Memory = MutableMapping[Address, Word]

SignedInteger = NewType("SignedInteger", int)


def word_to_signed_integer(word: Word, /) -> SignedInteger:
    if word >= (1 << (horse.types.WORD_N_BITS - 1)):
        return SignedInteger(word - (1 << horse.types.WORD_N_BITS))
    else:
        return SignedInteger(word)


def signed_integer_to_word(signed_integer: SignedInteger, /) -> Word:
    flowed = signed_integer % (1 << horse.types.WORD_N_BITS)
    return Word(flowed)


# TODO: these don't typecheck and also I don't use them yet
# either uncomment or delete
#
# @dataclasses.dataclass
# class Nibbles(Sequence[Nibble]):
#     """Wrapper for accessing ``bytearray``s by the 4-bit nibble."""
#
#     bytearray: bytearray
#
#     def __getitem__(self, index: Union[int, slice], /) -> Nibble:
#         byte = self.bytearray[index // 2]
#
#         if index % 2 == 0:
#             return Nibble(byte >> 4)
#         else:
#             return Nibble(byte % 16)
#
#     def __setitem__(self, index: int, item: Nibble, /) -> None:
#         assert 0 <= item < (1 << 4)
#
#         byte = self.bytearray[index // 2]
#
#         if index % 2 == 0:
#             self.bytearray[index // 2] = ((byte >> 4) << 4) + item
#         else:
#             self.bytearray[index // 2] = (item << 4) + (byte % 16)
#
#     def __len__(self) -> int:
#         return len(self.bytearray) * 2
#
#
# @dataclasses.dataclass
# class Words(Sequence[Word]):
#     """Wrapper for accessing ``bytearray``s by the 16-bit word."""
#
#     bytearray: bytearray
#
#     def __post_init__(self):
#         assert len(self.bytearray) % 2 == 0
#
#     def __getindex__(self, index: int, /) -> Word:
#         top, bottom = self.bytearray[index * 2 : index * 2 + 1]
#         return Word((top << 8) + bottom)
#
#     def __setitem__(self, index: int, item: Word, /) -> None:
#         assert 0 <= item < (1 << 16)
#         self.bytearray[index] = item // (1 << 8)
#         self.bytearray[index + 1] = item % (1 << 8)
#
#     def __len__(self) -> int:
#         return len(self.bytearray) // 2


@dataclasses.dataclass
class RegisterMappingWrapper(MutableMapping[Register, Word]):
    wrapped_mapping: MutableMapping[Register, Word]

    def __post_init__(self):
        self.wrapped_mapping[Register.ZERO_REGISTER] = 0

    def __getitem__(self, key: Register) -> Word:
        return self.wrapped_mapping[key]

    def __setitem__(self, key: Register, value: Word) -> None:
        if key != Register.ZERO_REGISTER:
            self.wrapped_mapping[key] = value

    def __delitem__(self, key: Register) -> None:
        if key != Register.ZERO_REGISTER:
            del self.wrapped_mapping[key]

    def __len__(self) -> int:
        return len(self.wrapped_mapping)

    def __iter__(self) -> Iterator[Register]:
        return iter(self.wrapped_mapping)


@dataclasses.dataclass
class Machine:
    name: str
    memory: MutableMapping[Address, Word]
    registers: MutableMapping[Register, Word] = dataclasses.field(
        default_factory=lambda: {register: Word(0) for register in Register}
    )
    halted: bool = False

    def __post_init__(self) -> None:
        self.registers = RegisterMappingWrapper(self.registers)

    def tick(self) -> None:
        instruction_address = Address(self.registers[Register.PROGRAM_COUNTER])
        instruction_as_word = self.memory[instruction_address]
        instruction = parse(instruction_as_word)

        instruction(self)

        if not self.halted:
            UnaryOperation(
                increment, Register.PROGRAM_COUNTER, Register.PROGRAM_COUNTER
            )(self)


@typing.runtime_checkable
class Instruction(Protocol):
    def __call__(self, machine: Machine) -> None:
        raise NotImplementedError


class NoOp(Instruction):
    def __call__(self, machine: Machine) -> None:
        pass


class Halt(Instruction):
    def __call__(self, machine: Machine) -> None:
        machine.halted = True


@dataclasses.dataclass
class Load(Instruction):
    address: Register
    target: Register

    def __call__(self, machine: Machine) -> None:
        address = Address(machine.registers[self.address])
        machine.registers[self.target] = machine.memory[address]


@dataclasses.dataclass
class Store(Instruction):
    address: Register
    source: Register

    def __call__(self, machine: Machine) -> None:
        address = Address(machine.registers[self.address])
        machine.memory[address] = machine.registers[self.source]


@dataclasses.dataclass
class CopyIf(Instruction):
    register_to_test: Register
    source: Register
    target: Register

    def __call__(self, machine: Machine) -> None:
        if machine.registers[self.register_to_test]:
            machine.registers[self.source] = machine.registers[self.target]


@dataclasses.dataclass
class BinaryOperation(Instruction):
    class _WrappedFunction(Protocol):
        def __call__(self, operand0: Word, operand1: Word, /) -> Word:
            ...

    func: _WrappedFunction
    operand0: Register
    operand1: Register
    result: Register

    def __call__(self, machine: Machine) -> None:
        machine.registers[self.result] = self.func(
            machine.registers[self.operand0], machine.registers[self.operand1],
        )


@dataclasses.dataclass
class UnaryOperation(Instruction):
    class _WrappedFunction(Protocol):
        def __call__(self, operand: Word, /) -> Word:
            ...

    func: _WrappedFunction
    operand: Register
    result: Register

    def __call__(self, machine: Machine) -> None:
        machine.registers[self.result] = self.func(machine.registers[self.operand])


class BinaryOpCode(enum.Enum):
    NON_BINARY_OPERATION = 0
    COPY_IF = 1
    # 2
    # 3

    TEST_EQUAL = 4
    TEST_GREATER_THAN = 5

    BITWISE_AND = 6
    BITWISE_OR = 7
    BITWISE_XOR = 8

    ADD = 9
    SUBTRACT = 10
    MULTIPLY = 11

    FLOOR_DIVIDE = 12
    MODULUS = 13
    LEFT_SHIFT = 14
    RIGHT_SHIFT = 15


def test_equal(operand0: Word, operand1: Word, /) -> Word:
    return convert_to_bool(Word(operand0 == operand1))


def test_greater_than(operand0: Word, operand1: Word, /) -> Word:
    test_result = word_to_signed_integer(operand0) < word_to_signed_integer(operand1)
    return convert_to_bool(Word(test_result))


def bitwise_and(operand0: Word, operand1: Word, /) -> Word:
    return Word(operand0 & operand1)


def bitwise_or(operand0: Word, operand1: Word, /) -> Word:
    return Word(operand0 | operand1)


def bitwise_xor(operand0: Word, operand1: Word, /) -> Word:
    return Word(operand0 ^ operand1)


def _signed_binop(
    func: Callable[[SignedInteger, SignedInteger], SignedInteger],
) -> Callable[[Word, Word], Word]:
    def signed_binop(operand0: Word, operand1: Word, /, func=func):
        return signed_integer_to_word(
            func(word_to_signed_integer(operand0), word_to_signed_integer(operand1))
        )

    return signed_binop


add = _signed_binop(operator.add)
subtract = _signed_binop(operator.sub)
multiply = _signed_binop(operator.mul)


def floor_divide(operand0: Word, operand1: Word, /) -> Word:
    try:
        return _signed_binop(operator.floordiv)(operand0, operand1)
    except ZeroDivisionError:
        return Word(0)


def modulus(operand0: Word, operand1: Word, /) -> Word:
    try:
        return _signed_binop(operator.mod)(operand0, operand1)
    except ZeroDivisionError:
        return Word(0)


def left_shift(operand0: Word, operand1: Word, /) -> Word:
    return Word((operand0 << operand1) % (1 << horse.types.WORD_N_BITS))


def right_shift(operand0: Word, operand1: Word, /) -> Word:
    return Word((operand0 >> operand1) % (1 << horse.types.WORD_N_BITS))


BINARY_OPERATIONS = {
    BinaryOpCode.TEST_EQUAL: test_equal,
    BinaryOpCode.TEST_GREATER_THAN: test_greater_than,
    BinaryOpCode.BITWISE_AND: bitwise_and,
    BinaryOpCode.BITWISE_OR: bitwise_or,
    BinaryOpCode.BITWISE_XOR: bitwise_xor,
    BinaryOpCode.ADD: add,
    BinaryOpCode.SUBTRACT: subtract,
    BinaryOpCode.MULTIPLY: multiply,
    BinaryOpCode.FLOOR_DIVIDE: floor_divide,
    BinaryOpCode.MODULUS: modulus,
    BinaryOpCode.LEFT_SHIFT: left_shift,
    BinaryOpCode.RIGHT_SHIFT: right_shift,
}


class NonBinaryOpCode(enum.Enum):
    NOOP = 0
    HALT = 1
    # 2
    # 3

    LOAD = 4
    STORE = 5

    INCREMENT = 6
    DECREMENT = 7

    CONVERT_TO_BOOL = 8
    BITWISE_NOT = 9
    NEGATE = 10
    POSIT = 11

    # 12
    # 13
    # 14
    # 15


def increment(operand: Word, /) -> Word:
    return add(operand, Word(1))


def decrement(operand: Word, /) -> Word:
    return subtract(operand, Word(1))


def convert_to_bool(operand: Word, /) -> Word:
    return Word(horse.types.TRUE if operand else horse.types.FALSE)


def _signed_unop(
    func: Callable[[SignedInteger], SignedInteger]
) -> Callable[[Word], Word]:
    def signed_unop(operand: Word, /, func=func) -> Word:
        return signed_integer_to_word(func(word_to_signed_integer(operand)))

    return signed_unop


bitwise_not = _signed_unop(operator.invert)
negate = _signed_unop(operator.neg)
posit = _signed_unop(operator.pos)


UNARY_OPERATIONS = {
    NonBinaryOpCode.INCREMENT: increment,
    NonBinaryOpCode.DECREMENT: decrement,
    NonBinaryOpCode.CONVERT_TO_BOOL: convert_to_bool,
    NonBinaryOpCode.BITWISE_NOT: bitwise_not,
    NonBinaryOpCode.NEGATE: negate,
    NonBinaryOpCode.POSIT: posit,
}


def parse(word: Word) -> Instruction:
    nibbles = horse.types.word_to_nibbles(word)

    try:
        opcode = BinaryOpCode(nibbles[0])
    except ValueError:
        # undefined opcode
        return Halt()

    if opcode == BinaryOpCode.NON_BINARY_OPERATION:
        return parse_non_binary_operation(word)
    elif opcode == BinaryOpCode.COPY_IF:
        register_to_test = Register(nibbles[1])
        source = Register(nibbles[2])
        target = Register(nibbles[3])
        return CopyIf(register_to_test, source, target)
    elif opcode in BINARY_OPERATIONS:
        return parse_binary_operation(word)
    else:
        assert False, "This should never happen."


def parse_binary_operation(word: Word) -> Instruction:
    nibbles = horse.types.word_to_nibbles(word)
    opcode = BinaryOpCode(nibbles[0])

    func = BINARY_OPERATIONS[opcode]
    operand0 = Register(nibbles[1])
    operand1 = Register(nibbles[2])
    result = Register(nibbles[3])

    return BinaryOperation(func, operand0, operand1, result)


def parse_non_binary_operation(word: Word) -> Instruction:
    nibbles = horse.types.word_to_nibbles(word)

    try:
        opcode = NonBinaryOpCode(nibbles[1])
    except ValueError:
        # undefined opcode
        return Halt()

    if opcode == NonBinaryOpCode.NOOP:
        return NoOp()
    elif opcode == NonBinaryOpCode.HALT:
        return Halt()
    elif opcode == NonBinaryOpCode.LOAD:
        address = Register(nibbles[2])
        target = Register(nibbles[3])
        return Load(address, target)
    elif opcode == NonBinaryOpCode.STORE:
        address = Register(nibbles[2])
        source = Register(nibbles[3])
        return Store(address, source)
    elif opcode in UNARY_OPERATIONS:
        func = UNARY_OPERATIONS[opcode]
        operand = Register(nibbles[2])
        result = Register(nibbles[3])
        return UnaryOperation(func, operand, result)
    else:
        assert False, "This should never happen."


@dataclasses.dataclass
class VirtualMemory(MutableMapping[Address, Word]):
    offset: int
    real_memory: MutableMapping[Address, Word]

    def __repr__(self):
        return f"{self.__class__.__name__}(offset={self.offset}, real_memory=...)"

    def real_address(self, virtual_address: Address) -> Address:
        return Address(
            Word((virtual_address + self.offset) % (1 << horse.types.WORD_N_BITS))
        )

    def __getitem__(self, virtual_address: Address) -> Word:
        return self.real_memory[self.real_address(virtual_address)]

    def __setitem__(self, virtual_address: Address, value: Word) -> None:
        self.real_memory[self.real_address(virtual_address)] = value

    def __delitem__(self, virtual_address: Address) -> None:
        del self.real_memory[self.real_address(virtual_address)]

    def __len__(self) -> int:
        return len(self.real_memory)

    def __iter__(self) -> Iterator[Address]:
        return iter(self.real_memory)


def tournament(programs: Mapping[str, bytes], seed: int) -> str:
    """Runs a tournament and determines the winner."""
    memory = {Address(Word(i)): Word(0) for i in range(1 << horse.types.WORD_N_BITS)}

    random.seed(seed)

    offsets = [0 for program in programs]
    for program in programs:
        pass

    # create machines
    machines = [  # noqa: F841
        Machine(name, memory=VirtualMemory(offset, memory))
        for name, offset in zip(programs, offsets)
    ]

    return "it was a draw"
