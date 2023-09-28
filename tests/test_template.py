#!python3
import pytest

def test_get_root(request):
    rootdir = request.config.rootdir
    assert (rootdir / 'pyproject.toml').isfile()

# test Raise
def f():
    raise SystemExit(1)


def test_mytest():
    with pytest.raises(SystemExit):
        f()


# test class
class TestClassDemoInstance:
    value = 0

    def test_one(self):
        self.value = 1
        assert self.value == 1

    def test_two(self):
        assert self.value == 0
