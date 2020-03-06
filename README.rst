====================================
High-Octane Rumble Simulation Engine
====================================

It's a fight to the death!

.. contents::

---------------------
The ``blen`` language
---------------------

The ``blen`` language

thing.blen

   +---+---+---+---+---+---+---+---+---------+
   | opcode        | other data    | meaning |
   +---+---+---+---+---+---+---+---+---------+
   | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | nop     |
   +---+---+---+---+---+---+---+---+---------+

``blen`` semantics
===================

The ``blen`` machine has 16 registers.

r0 - always 0
r1 - program counter
r2-r15 - general purpose int registers

Instructions:
=============

1. ``pass``: does nothing
2. ``halt``: ends the program (avoid if possible)

1. This is a 'unary instruction', inspect second 4-bit chunk
2. ``copyif a b c`` if a nonzero copy b to c
3. ``eq a b c``
4. ``gt a b c``

5. ``and a b c``
6. ``or a b c``
7. ``xor a b c``
8.

9. ``add a b c``
10. ``sub a b c``
11. ``mul a b c``
12. ``div a b c``

13. ``mod a b c``
14. ``lshift a b c``
15. ``rshift a b c``
16.

Unary instructions
------------------

1. ``halt``
2. ``nop``
4.

5. ``load a b`` - load data from address in register a into register b
6. ``store a b`` - store data from register b into address in register a

7. ``inc a b``
8. ``dec a b``

9. ``bool a b`` - if a nonzero, set b to True (-1)
10. ``not a b`` - do a bitwise not of a and put it in b
11. ``neg a b``
12. ``pos a b``

13.
14.
15.
16. ``show a`` - prints the contents of a to the chat


FAQ: How do I do X?
===================

``mv a b``: ``add r0 a b``
``jmp c``: ``add r0 c r1``
``a != b --> c``: ``eq a b c; zeroif c c``





