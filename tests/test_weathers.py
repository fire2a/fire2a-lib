from pandas import DataFrame
from shutil import rmtree
from fire2a.weathers import cut_weather_scenarios, random_weather_scenario_generator
from pathlib import Path 

# Test module
def test_cut_weather_scenarios():
    # Creating sample weather data (assuming columns: 'Temperature', 'Humidity', 'Pressure')
    weather_data = DataFrame({
        'WS': [20, 22, 25, 18, 19, 23, 20],
        'WD': [50, 60, 55, 58, 62, 48, 52],
        'TMP': [1010, 1015, 1005, 1008, 1012, 1003, 1010]
    })

    # Define scenario lengths
    scenario_lengths = [2, 3, 2, 3, 7, 5, 6, 4, 3, 2]

    # Test scenario cutting and file creation
    cut_weather_scenarios(weather_data, scenario_lengths, output_folder='Weathers_test')

    # Verify if files are created in the 'Weathers' directory
    output_folder = Path('Weathers_test')
    assert all((output_folder / f'weather{i:02d}.csv').exists() for i in range(1, len(scenario_lengths) + 1))
    rmtree(output_folder)

# Test module
def test_random_weather_scenario_generator():
    n_scenarios = 10
    output_folder = Path("TestOutput")
    # Generate random weather scenarios
    random_weather_scenario_generator(n_scenarios, output_folder=output_folder)

    # Verify if files are created in the output directory
    assert all((output_folder / f'weather{i:02d}.csv').exists() for i in range(1, n_scenarios + 1))
    rmtree(output_folder)

if __name__ == "__main__":
    test_random_weather_scenario_generator()
    test_cut_weather_scenarios()
    print("All tests passed!") 