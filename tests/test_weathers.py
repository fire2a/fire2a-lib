from pathlib import Path
from shutil import rmtree

import pytest
from pandas import DataFrame

from fire2a.weathers import cut_weather_scenarios, random_weather_scenario_generator, re_size_durations


@pytest.fixture
def weather_records():
    # Create a sample DataFrame of weather records for testing
    return DataFrame(
        {
            "WS": [20, 22, 25, 18, 19, 23, 20],
            "WD": [50, 60, 55, 58, 62, 48, 52],
            "TMP": [1010, 1015, 1005, 1008, 1012, 1003, 1010],
        }
    )


def test_input_validation(weather_records):
    # Test input with non-DataFrame weather_records
    with pytest.raises(ValueError):
        cut_weather_scenarios("not a DataFrame", [3, 5, 8])

    # Test input with non-integer scenario_lengths
    with pytest.raises(ValueError):
        cut_weather_scenarios(weather_records, [3, "five", 8])

    # Test input with n_output_files as a non-integer
    with pytest.raises(ValueError):
        cut_weather_scenarios(weather_records, [3, 5, 8], n_output_files="ten")

    # Test scenario length greater than total length of weather_records
    with pytest.raises(ValueError):
        cut_weather_scenarios(weather_records, [6, 10])


def test_cut_weather_scenarios(weather_records, tmp_path):
    # Define scenario lengths
    scenario_lengths = [2, 3, 2, 3, 7, 5, 6, 4, 3, 2]

    # Test scenario cutting and file creation
    cut_weather_scenarios(weather_records, scenario_lengths, output_folder=tmp_path)

    # Verify if files are created in the temporary directory
    assert all((Path(tmp_path) / f"Weather{i}.csv").exists() for i in range(1, 101))


def test_random_weather_scenario_generator(tmp_path):
    n_scenarios = 10
    # Generate random weather scenarios
    random_weather_scenario_generator(n_scenarios, output_folder=tmp_path)
    # Verify if files are created in the temporary directory
    assert all((tmp_path / f"Weather{i}.csv").exists() for i in range(1, n_scenarios + 1))
