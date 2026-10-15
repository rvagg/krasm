import unittest

from check_dispatch import check, functions, stray_calls

JUMPING = """\
   1000:\tmov    -0x8(%rcx),%rax
   1004:\tadd    %rax,-0x10(%rcx)
   1008:\tmov    0x58(%rax,%rsi,1),%rax
   100c:\tjmp    *%rax
"""

LIBM = """\
   1000:\tmov    0xc0730(%rip),%rbx        # 19e0a8 <_DYNAMIC+0x320>
   1007:\tcall   *%rbx
   1009:\tcall   *%rbx
   100b:\tmov    0x58(%rax,%rsi,1),%rax
   100f:\tjmp    *%rax
"""

CALLING = """\
   1000:\tinc    %rdx
   1003:\tcall   *0x28(%rax)
   1006:\tadd    $0x48,%rsp
   100a:\tret
"""

CALLING_THROUGH_REGISTER = """\
   1000:\tmov    0x58(%rax,%rsi,1),%r11
   1004:\tcall   *%r11
   1007:\tret
"""


def disassembly(*bodies):
    return "".join(
        f"{n:016x} <krasm::runtime::flat_executor::handler_for::{{closure#{n}}} as FnOnce>::call_once>:\n{text}"
        for n, text in enumerate(bodies)
    )


def body(text):
    (instructions,) = functions(disassembly(text)).values()
    return instructions


class DispatchTests(unittest.TestCase):
    def test_external_calls_through_the_got_are_allowed(self):
        self.assertEqual(stray_calls(body(LIBM)), [])

    def test_calling_a_successor_is_reported(self):
        self.assertEqual(stray_calls(body(CALLING)), [("1003", "call *0x28(%rax)")])
        self.assertEqual(stray_calls(body(CALLING_THROUGH_REGISTER)), [("1004", "call *%r11")])

    def test_jumping_handlers_pass(self):
        self.assertEqual(check(disassembly(*[JUMPING] * 100)), [])

    def test_one_calling_handler_fails(self):
        problems = check(disassembly(*[JUMPING] * 100, CALLING))
        self.assertEqual(len(problems), 1)
        self.assertIn("{closure#100}", problems[0])

    def test_too_few_handlers_fail(self):
        self.assertIn("found 99 handlers", check(disassembly(*[JUMPING] * 99))[0])

    def test_a_trampoline_build_fails(self):
        returning = "   1000:\tret\n"
        problems = check(disassembly(*[JUMPING] * 80, *[returning] * 20))
        self.assertIn("80 of 100 handlers dispatch by jump", problems[0])


if __name__ == "__main__":
    unittest.main()
