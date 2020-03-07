====================================
High-Octane Rumble Simulation Engine
====================================

It's a fight to the death!

.. contents::

----------
How to use
----------

Start by installing poetry: https://python-poetry.org/

Install the project dependencies:

.. code::

    poetry install

Run the unit tests:

.. code::

    poetry run pytest

Compile a program:

.. code::

    poetry run python -m horse.compiler --input program.blen --output program.blenc


---------------------
The ``blen`` language
---------------------

``blen`` is an assembly language for the imaginary Blen machine.

What is the Blen machine?
=========================

The Blen machine is composed of _registers_ and _memory_.

A _register_ is a box that contains a number. When a number is in a register, 
the machine can use it to do maths. The Blen machine only has 16 registers,
labelled ``R0`` to ``R15``.

The _memory_ is a mapping from addresses to numbers. The machine can't do maths
on numbers when they're in the memory, but it can load numbers from the memory
into registers and store numbers from registers into the memory to use them
later. The memory also contains the program instructions.

The Blen machine runs using the following loop:

1. read a number from the memory at the address in a special register called the
   ``PROGRAM_COUNTER``.
2. parse that number as an instruction.
3. execute that instruction by reading some numbers from the registers, doing
   some maths on them and writing the result back to a register.
4. add one to the ``PROGRAM_COUNTER``.

How does the Blen machine parse instructions?
=============================================

The Blen machine represents numbers using 16 bits. When it parses isntructions,
it groups those 16 bits into four 4-bit numbers (nibbles). For example, this is
the way that the Blen machine parses the number 38611 (1001011011010011):

+--------+----------+----------+--------+--------------------------------+
| opcode | operand0 | operand1 | result | meaning                        |
+--------+----------+----------+--------+--------------------------------+
| 1001   | 0110     | 1101     | 0011   | add ``R6`` to ``R13``          |
+--------+----------+----------+--------+ and store the result in ``R3`` |
| 9      | 6        | 13       | 3      |                                |
+--------+----------+----------+--------+--------------------------------+

All binary mathematical operations are parsed this way: the first nibble is an
_opcode_ that tells the machine which function to call, the next two nibbles
tell the machine which two registers to use as arguments for the function, and
the last nibble tells the machine which register to store the result in.

There are some instructions that require fewer than two arguments, which are
parsed slightly differently. For example, here is how the number 
1233 (0000010011010001) is parsed:

+------+--------+------+------+-----------------------------+
| zero | opcode | a    | b    | meaning                     |
+------+--------+------+------+-----------------------------+
| 0000 | 0100   | 1101 | 0001 | load a number from memory   |
+------+--------+------+------+ using the address in ``R6`` |
| 0    | 4      | 6    | 1    | and store in in ``R1``      |
+------+--------+------+------+-----------------------------+

These are called "non-binary operations". They all have opcode 0 when parsed as
binary operations.

Special registers
=================

The Blen machine gives some registers special meaning:

- ``R0`` is the ``ZERO_REGISTER``. The ``ZERO_REgISTER`` always contains 0.
  Writes to the ``ZERO_REGISTER`` do nothing.

- ``R1`` is the ``PROGRAM_COUNTER``. 
  At the start of each iteration of the core program loop, the machine reads an 
  address from the ``PROGRAM_COUNTER`` and loads from memory the instruction 
  with that address. 
  At the end of each iteration, the ``PROGRAM_COUNTER`` is increased by 1.
  You can read and write from the ``PROGRAM_COUNTER`` as normal, and you can use
  this to jump between sections of the program.

Boolean values
==============

Any number that is non-zero is treated as ``TRUE`` by instructions that expect
a boolean value. Instructions that return a boolean value return 
``TRUE`` or ``FALSE``.

.. code::

    TRUE = 65536  (1111111111111111 in binary)
    FALSE = 0     (0000000000000000 in binary)

Signed integers
===============

Most binary operations treat their arguments as `two's complement signed integers`__.

__ https://en.wikipedia.org/wiki/Two%27s_complement


Instructions for the Blen machine
=================================

Binary operations
-----------------

The numbers in this list indicate the opcode of the relevant instruction:

1. ``copy_if test source target``: If the ``test`` register is non-zero, copy
   the number in the ``source`` register to the ``target`` register.

4. ``test_equal a b result``: If the number in register ``a`` is equal to the
   number in register ``b``, write ``TRUE`` to the ``result`` register.
   Otherwise, write ``FALSE`` to the ``result`` register.
5. ``test_greater_than a b result``: If the number in regsiter ``a`` is greater
   than the number in register ``b``, write ``TRUE`` to the ``result`` register.
   Otherwise, write ``FALSE`` to the ``result`` register.
6. ``bitwise_and a b result``: compute the bitwise and of ``a`` and ``b`` and
   write the result to the ``result`` register.
7. ``bitwise_or a b result``: compute the bitwise or of ``a`` and ``b`` and
   write the result to the ``result`` register.
8. ``bitwise_xor a b result``: compute the bitwise exclusive or of ``a`` and ``b`` and
   write the result to the ``result`` register.
9. ``add a b result``: add ``a`` to ``b`` and write the result to ``result``.
10. ``subtract a b result``: subtract ``b`` from ``a`` and write the result to ``result``.
11. ``multiply a b result``: mulitply ``a`` and ``b`` and write the result to ``result``.
12. ``floor_divide a b result``: divide ``a`` by ``b``, discard the remainder 
    and write the result to ``result``.
13. ``modulus a b result``: divide ``a`` by ``b`` and write the remainder to ``result``.
14. ``left_shift a b result``: move each bit of ``a`` to the left by ``b`` bits 
    and write the remainder to ``result``.
15. ``left_shift a b result``: move each bit of ``a`` to the right by ``b`` bits 
    and write the remainder to ``result``.

Non-binary operations
---------------------

The numbers in this list indicate the opcode of the relevant instruction:

0. ``no_op``: Does nothing.
1. ``halt``: Stops the program.

4. ``load address target``: Load a number from the memory using the address in
   the ``address`` register and write it to the ``target`` register.
5. ``store address source``: Store the number in the ``source`` register in
   the memory using the address in the ``target`` register.
6. ``increment a result``: Add one to the number in register ``a`` and store the
   result in the ``result`` register.
7. ``decrement a result``: Subtract one from the number in register ``a`` and store the
   result in the ``result`` register.
8. ``convert_to_bool a result``: If the number in register ``a`` is not zero,
   write ``TRUE`` in the ``result`` register. Otherwise, write ``FALSE`` to the
   ``result`` register.
9. ``bitwise_not a result``: Compute the bitwise not of ``a`` and store the
   result in the ``result`` register.
10. ``negate a result``: Treating the number in ``a`` as a signed integer,
    write ``-a`` to the ``result`` register.
11. ``posit a result``: Treating the number in ``a`` as a signed integer,
    write the absolute value of ``a`` to the ``result`` register.


FAQ: How do I do X?
===================

Move the value in ``RX`` to ``RY``: ``add R0 RX RY``

Jump to the instruction at the address in ``RX``: ``add R0 RX R1``
