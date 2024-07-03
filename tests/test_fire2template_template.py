#!python3
"""👋🌎
testing fire2template.template.py
"""
import pytest


def test_calc():
    from fire2template.template import calc
    from numpy.random import rand

    # OJO: testing with random is not a good practice
    numbers = rand(10)

    summ = numbers[0]
    subs = numbers[0]
    mult = numbers[0]
    div = numbers[0]
    for num in numbers[1:]:
        summ += num
        subs -= num
        mult *= num
        div /= num if num != 0 else 1
    assert calc("+", numbers) == summ
    assert calc("-", numbers) == subs
    assert calc("*", numbers) == mult
    assert calc("/", numbers) == div


def test_help():
    """test help by running the cli command --help and not failing"""
    import io
    from contextlib import redirect_stderr, redirect_stdout

    from fire2template.template import main

    with io.StringIO() as buf, redirect_stdout(buf), redirect_stderr(buf):
        with pytest.raises(SystemExit) as e:
            main(["--help"])
            output = buf.getvalue()
            assert "usage:" in output
        # Assert that the exit code is 0
        assert e.value.code == 0

        # clean the buffer
        buf.truncate(0)

        with pytest.raises(SystemExit) as e:
            main(["-h"])
            output = buf.getvalue()
            assert "usage:" in output
        # Assert that the exit code is 0
        assert e.value.code == 0
