from pandas import DataFrame
from shutil import rmtree
from fire2a.weathers import cut_weather_scenarios, random_weather_scenario_generator, re_size_durations
from pathlib import Path 
import unittest

class TestCutWeatherScenarios(unittest.TestCase):
    def setUp(self):
        # Create a sample DataFrame of weather records for testing
        self.weather_records = DataFrame({
            'WS': [20, 22, 25, 18, 19, 23, 20],
            'WD': [50, 60, 55, 58, 62, 48, 52],
            'TMP': [1010, 1015, 1005, 1008, 1012, 1003, 1010]
        })

    def test_input_validation(self):
        # Test input with non-DataFrame weather_records
        with self.assertRaises(ValueError):
            cut_weather_scenarios("not a DataFrame", [3, 5, 8])

        # Test input with non-integer scenario_lengths
        with self.assertRaises(ValueError):
            cut_weather_scenarios(self.weather_records, [3, 'five', 8])

        # Test input with n_output_files as a non-integer
        with self.assertRaises(ValueError):
            cut_weather_scenarios(self.weather_records, [3, 5, 8], n_output_files='ten')

        # Test scenario length greater than total length of weather_records
        with self.assertRaises(ValueError):
            cut_weather_scenarios(self.weather_records, [6, 10])

    def test_cut_weather_scenarios(self):

        # Define scenario lengths
        scenario_lengths = [2, 3, 2, 3, 7, 5, 6, 4, 3, 2]

        # Test scenario cutting and file creation
        cut_weather_scenarios(self.weather_records, scenario_lengths, output_folder='Weathers_test')

        # Verify if files are created in the 'Weathers' directory
        output_folder = Path('Weathers_test')
        assert all((output_folder / f'weather{i}.csv').exists() for i in range(1, 100))
        rmtree(output_folder)

def test_random_weather_scenario_generator():
    n_scenarios = 10
    output_folder = Path("TestOutput")
    # Generate random weather scenarios
    random_weather_scenario_generator(n_scenarios, output_folder=output_folder)

    # Verify if files are created in the output directory
    assert all((output_folder / f'weather{i}.csv').exists() for i in range(1, n_scenarios + 1))
    rmtree(output_folder)

class TestResizeDurations(unittest.TestCase):
    def test_input_validation(self):
        # Test input with non-integer values
        with self.assertRaises(ValueError):
            re_size_durations([1, 2, 'three', 4, 5])

        # Test input with n_samples as a non-integer
        with self.assertRaises(ValueError):
            re_size_durations([1, 2, 3, 4, 5], 'ten')

    def test_output_length(self):
        # Test output length matches n_samples when n_samples is provided
        result = re_size_durations([1, 2, 3, 4, 5], n_samples=50)
        self.assertEqual(len(result), 50)

        # Test output length defaults to 100 if n_samples is not provided
        result_default = re_size_durations([1, 2, 3, 4, 5])
        self.assertEqual(len(result_default), 100)

if __name__ == '__main__':
    unittest.main()
    test_random_weather_scenario_generator()
    print("All tests passed!") 