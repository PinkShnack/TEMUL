
import periodictable as pt
import pytest

import temul.element_tools as tml_et


def test_get_and_return_element():
    moly = tml_et.get_and_return_element(element_symbol='Mo')
    assert isinstance(moly, pt.core.Element)
    assert moly.symbol == 'Mo'
    assert moly.covalent_radius == 1.54
    assert moly.number == 42


def test_atomic_radii_in_pixels():
    radius = tml_et.atomic_radii_in_pixels(0.01666666667, 'Mo')
    assert radius == pytest.approx(4.62, rel=1e-2)


def test_split_and_sort_element_for_stacked_configuration():
    info = tml_et.split_and_sort_element('O_6.Mo_3.Ti_5')

    assert info == [
        [['O', '6'], 'O', 6, 8],
        [['Mo', '3'], 'Mo', 3, 42],
        [['Ti', '5'], 'Ti', 5, 22],
    ]


def test_split_and_sort_element_rejects_wrong_split_symbol():
    with pytest.raises(ValueError, match="split a stacked element"):
        tml_et.split_and_sort_element('O_6.Mo_3', split_symbol=['_', '-'])


def test_get_individual_elements_from_element_list_handles_nested_lists():
    elements = [['Ti_7_0', 'Ti_9.Re_3', 'Ge_2'], ['B_9', 'B_2.Fe_8']]

    assert tml_et.get_individual_elements_from_element_list(elements) == [
        'B', 'Fe', 'Ge', 'Re', 'Ti']


def test_get_individual_elements_from_element_list_rejects_empty():
    with pytest.raises(ValueError, match="greater than 0"):
        tml_et.get_individual_elements_from_element_list([])


def test_combine_element_lists():
    combined = tml_et.combine_element_lists(
        [['Mo_0', 'Ti_3'], ['Ti_3', 'Ge_2']])

    assert combined == ['Ge_2', 'Mo_0', 'Ti_3']
