====================================
High-Octane Rumble Simulation Engine
====================================

It's a fight to the death!

.. contents::

-------------
What is this?
-------------

Once my dad told me a story from his time as a young software engineer.
To pass the time, he and his friends would have programming deathmatches.
One of them would write a compiler, then the others would write programs.
All the programs would be put in the same block of memory,
and the last program still running wins.

It sounded like fun, so for a birthday party (yes, I am very cool)
I wrote this compiler and challenged my friends to a tournament.

This programming language is not intended to be practical;
it explicitly lacks a number of convenience features,
like instructions for loading small constants directly into registers,
to make useful programs harder (but more fun) to write.

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

Run a tournament!

.. code::

    poetry run python -m horse.tournament jess.blenc blaine.blenc


---------------------
The ``blen`` language
---------------------

``blen`` is an assembly language for the imaginary Blen machine.

What is the Blen machine?
=========================

The Blen machine is composed of *registers* and *memory*.

A *register* is a box that contains a number. When a number is in a register, 
the machine can use it to do maths. The Blen machine only has 16 registers,
labelled ``R0`` to ``R15``.

The *memory* is a mapping from addresses to numbers. The machine can't do maths
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

The Blen machine represents numbers using 16 bits. When it parses instructions,
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
*opcode* that tells the machine which function to call, the next two nibbles
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

- ``R0`` is the ``ZERO_REGISTER``. The ``ZERO_REGISTER`` always contains 0.
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

    TRUE = 65535  (1111111111111111 in binary)
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

Miscellaneous
-------------

``constant i``: insert the constant ``i`` into the source code of the program.
``; comment`` is a comment; anything from ``;`` to end-of-line will be ignored by the compiler. 

Examples
--------

A program that slides down some no-ops before looping back to the top:

   ; no_op does nothing
   no_op
   no_op
   no_op
   no_op
   no_op
   no_op
   no_op
   no_op
   ; decrement R0 gives us -1, which we write to the program counter
   ; after executing this instruction we increment the program counter
   ; hence next loop will begin executing R0
   decrement R0 PROGRAM_COUNTER

A program fills the address space with halt:

   ; we will use R2 as the register containing the halt instruction
   ; any integer that begins 00000001 will be treated as a halt
   ; i.e. anything between 512 and 1023
   increment R2       ; R2 =           1 = 1
   increment R2       ; R2 =          10 = 2
   multiply R2 R2 R2  ; R2 =         100 = 4
   multiply R2 R2 R2  ; R2 =      1 0000 = 16
   multiply R2 R2 R2  ; R2 = 1 0000 0000 = 512

   ; we will use R3 as the address to write to
   ; needs to be after our program
   increment R3 R3      ; R3 = 1
   increment R3 R3      ; R3 = 2
   left_shift R3 R3 R3  ; R3 = 8
   multiply R3 R3 R3    ; R3 = 64

   ; we will use R4 as the jump-back register
   ; tells us how long our loop is so we can reset the program counter
   increment R4 R4      ; R4 = 1
   increment R4 R4      ; R4 = 2
   increment R4 R4      ; R4 = 3

   ; now we do a tight loop to minimize our exposure
   store R3 R2      ; put the value in R2 (halt) at memory address R3
   increment R3 R3  ; increment the address to which we write
   subtract PROGRAM_COUNTER R4 PROGRAM_COUNTER  ; loop!!!


FAQ: How do I do X?
===================

Move the value in ``RX`` to ``RY``: ``add R0 RX RY``

Jump to the instruction at the address in ``RX``: ``add R0 RX PROGRAM_COUNTER``
