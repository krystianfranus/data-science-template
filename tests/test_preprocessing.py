import pytest

from src.preprocessing import split_data


@pytest.mark.parametrize("split_ratio", [0.12, 0.5])
def test_split_data(data, split_ratio):
    data_size = len(data)
    train_data, test_data = split_data(data, split_ratio)
    assert len(train_data) == int(data_size * split_ratio)
    assert len(test_data) == data_size - int(data_size * split_ratio)


@pytest.mark.parametrize("split_ratio", [-1, 1.1])
def test_split_data_with_incorrect_split_ratio(data, split_ratio):
    with pytest.raises(ValueError):
        split_data(data, split_ratio)
