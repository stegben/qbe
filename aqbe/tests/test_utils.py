import pytest

from aqbe.utils import RangeLookup


@pytest.fixture
def container():
    rl = RangeLookup()
    rl.add(10, 'a')
    rl.add(20, 'b')
    rl.add(30, 'c')
    rl.add(40, 'd')
    return rl


def test_range_lookup_get_sth(container):
    assert container[0] == ('a', 0)
    assert container[3] == ('a', 3)
    assert container[10] == ('a', 10)
    assert container[11] == ('b', 0)
    assert container[14] == ('b', 3)
    assert container[20] == ('b', 9)
    assert container[27] == ('c', 6)
    assert container[31] == ('d', 0)
    assert container[33] == ('d', 2)
    assert container[40] == ('d', 9)


def test_range_lookup_slicing(container):
    # a
    assert container[0:10] == [('a', 0), ('a', 10)]
    assert container[0:8] == [('a', 0), ('a', 8)]
    assert container[3:10] == [('a', 3), ('a', 10)]
    assert container[4:7] == [('a', 4), ('a', 7)]

    # b, c
    assert container[11:20] == [('b', 0), ('b', 9)]
    assert container[23:30] == [('c', 2), ('c', 9)]
    assert container[12:18] == [('b', 1), ('b', 7)]

    # d
    assert container[31:40] == [('d', 0), ('d', 9)]
    assert container[31:38] == [('d', 0), ('d', 7)]
    assert container[35:40] == [('d', 4), ('d', 9)]
    assert container[32:39] == [('d', 1), ('d', 8)]

    # a, ...
    assert container[0:22] == [('a', 0), ('a', 10), ('b', 0), ('b', 9), ('c', 0), ('c', 1)]
    assert container[7:12] == [('a', 7), ('a', 10), ('b', 0), ('b', 1)]

    # ...
    assert container[12: 31] == [('b', 1), ('b', 9), ('c', 0), ('c', 9), ('d', 0)]
    assert container[21: 33] == [('c', 0), ('c', 9), ('d', 0), ('d', 2)]

    # TODO: handle this case
    # assert container[20: 33] == [('b', 9), ('c', 0), ('c', 9), ('d', 0), ('d', 2)]


def test_range_lookup_raise_if_add_smaller():
    rl = RangeLookup()
    rl.add(100, 'a')
    with pytest.raises(Exception):
        rl.add(99, 'b')
