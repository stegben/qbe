import pytest

from aqbe.eval import overlap_ratio


def test_overlap_ratio():
    assert overlap_ratio(0, 1, 0, 1) == pytest.approx(1.)
    assert overlap_ratio(0, 1, 1, 2) == pytest.approx(0.)
    assert overlap_ratio(0, 1, 0, 1.1) == pytest.approx(1./1.1)
    assert overlap_ratio(0, 1, -1, 1) == pytest.approx(1./2.)
    assert overlap_ratio(0, 1, 0, 0.5) == pytest.approx(0.5/1.)
    assert overlap_ratio(0, 1, 0.5, 1) == pytest.approx(0.5/1.)
    assert overlap_ratio(0, 1, -1, 2) == pytest.approx(1./3.)
