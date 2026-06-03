import pytest
from hypothesis import given, settings
from hypothesis import strategies as st


@pytest.fixture
def fs_dir(tmp_path):
    return tmp_path


@given(data=st.lists(st.integers(), max_size=10))
@settings(max_examples=50)
def test_hypothesis_flatten(data):
    from sac.utils import UtilsSDK

    result = UtilsSDK.flatten([data])
    assert result == data
