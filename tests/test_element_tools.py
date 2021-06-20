
import periodictable as pt

import temul.element_tools as tml_et


def test_get_and_return_element():
    moly = tml_et.get_and_return_element(element_symbol='Mo')
    assert isinstance(moly, pt.core.Element)
    assert moly.symbol == 'Mo'
    assert moly.covalent_radius == 1.54
    assert moly.number == 42
