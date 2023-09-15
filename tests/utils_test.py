from oemer.utils import slope_to_degree


def test_slope_to_degree():
    assert slope_to_degree(0.3, 0.3) == 45.0