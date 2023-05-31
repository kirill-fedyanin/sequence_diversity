from true.calculate_ue import calculate_ue
from true.default_values import DEFAULT_UE_METHODS


def test_empty():
    assert calculate_ue([], {}) == {}


# def test_just_works():

#     model =

#     pass


# def test_equal_results():
#     man = UEManager()
#     pass
