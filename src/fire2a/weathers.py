#!python3
"""👋🌎  
Some functions related to weather scenario creation. 
"""
__author__ = "Rodrigo Mahaluf-Recasens"
__version__ = 'v0.0.1-38-gd1276ea-dirty'
__revision__ = "$Format:%H$"

from pandas import DataFrame
from random import randint
from numpy import vstack
from numpy.random import normal
from typing import List, Union, Optional
from pathlib import Path
from datetime import datetime, timedelta

def cut_weather_scenarios(weather_records: DataFrame, scenario_lengths: List[int], output_folder: Union[Path,str] = None) -> None:
    """
    Split weather records into smaller scenarios following specified scenario lengths.

    Parameters:
    - weather_records : pd.DataFrame
        A Pandas DataFrame containing weather records where each row represents an hour of data.
    - scenario_lengths : List[int]
        A list of integers representing desired lengths (in hours) for each weather scenario.
    - output_folder : Union[Path,str], optional
        A Path object or a string representing the folder path where the output will be stored.
        If not provided, 'Weathers' directory will be used.

    Output:
    - write as many file as weather scenarios generated based on specified lengths.

    Raises:
    - ValueError
        If input 'weather_records' is not a Pandas DataFrame.
        If input 'scenario_lengths' is not a List of integers.
        If any scenario length is greater than the total length of weather_records.
    """

    # Check if input is a Pandas DataFrame
    if not isinstance(weather_records, DataFrame):
        raise ValueError("Input 'weather_records' must be a Pandas DataFrame.")
    
    # Check if input is a list of integers
    if not all(isinstance(length, int) for length in scenario_lengths):
        raise ValueError("Input 'scenario_lengths' must be a list of integers.")
    
    # Defining the output folder
    output_folder = output_folder if output_folder else Path('Weathers')
    output_folder = Path(output_folder) # Ensure output_folder is a Path object
    output_folder.mkdir(parents=True, exist_ok=True)  # Create the output directory if it doesn't exist
    
    total_data_length = len(weather_records)

    # Check if any scenario length is greater than the total data length
    if any(length > total_data_length for length in scenario_lengths):
        raise ValueError("Scenario length cannot be greater than the total length of weather records")

    scenarios = []  # List to store weather scenarios
    total_scenarios = len(str(len(scenario_lengths))) #this is set just to preserve output format

    # Generate scenarios based on specified lengths
    for index, length in enumerate(scenario_lengths, start = 1):

        # Randomly select a start index for the scenario
        start_index = randint(0, total_data_length - length)

        # Extract the scenario based on the start index and length
        scenario = weather_records.iloc[start_index:start_index + length]

        # Save the weather scenario
        output_path = output_folder / f'weather{str(index).zfill(total_scenarios)}.csv'
        scenario.to_csv(output_path)

    return scenarios

# Example usage:
# Assuming 'weather_data' is your DataFrame and 'scenario_lengths' is a list of desired scenario lengths
# weather_data = pd.read_csv('your_weather_data.csv')
# scenario_lengths = [24, 48, 72]  # Example lengths
# weather_scenarios = cut_weather_scenarios(weather_data, scenario_lengths)

def random_weather_scenario_generator(n_scenarios: int, 
                                     hr_limit: Optional[int] = None, 
                                     lambda_ws: Optional[float] = None,
                                     lambda_wd: Optional[float] = None, 
                                     output_folder: Optional[str] = None):
    """
    Generates random weather scenarios and saves them as CSV files.

    Parameters:
    - n_scenarios : int
        Number of weather scenarios to generate.
    - hr_limit : int, optional
        Limit for the number of hours for each scenario (default is 72).
    - lambda_ws : float, optional
        Lambda parameter for wind speed variation (default is 0.5). If set to 0, all rows will have the same wind speed. 
    - lambda_wd : float, optional
        Lambda parameter for wind direction variation (default is 0.5). If set to 0, all rows will have the same wind direction.
    - output_folder : str, optional
        Path to the folder where output files will be saved (default is 'Weathers').

    Output:
    - Saves generated weather scenarios as CSV files in the specified output folder.
    """
    hr_limit = hr_limit if hr_limit else 72
    lambda_ws = lambda_ws if lambda_ws else 0.5
    lambda_wd = lambda_wd if lambda_wd else 0.5
    output_folder = Path(output_folder) if output_folder else Path('Weathers')
    output_folder.mkdir(parents=True, exist_ok=True)  # Create the output directory if it doesn't exist

    for index, scenario in enumerate(range(n_scenarios), start=1):
        n_rows = randint(5, hr_limit)

        instance = ['NA'] * n_rows
        fire_scenario = [2] * n_rows

        wd_0 = randint(0, 359)
        ws_0 = randint(1, 100)

        wd_1 = abs(wd_0 + normal(loc=0.0, scale=30.0, size=None))
        ws_1 = abs(ws_0 + normal(loc=0.0, scale=8.0, size=None))

        ws = [ws_0, ws_1]
        wd = [wd_0, wd_1]

        dt = [(datetime.now() + timedelta(hours=i)).isoformat(timespec='minutes') for i in range(n_rows)]
        for row in range(2, n_rows):
            wd_i = wd[row - 1] * lambda_wd + wd[row - 2] * (1 - lambda_wd)
            ws_i = ws[row - 1] * lambda_wd + ws[row - 2] * (1 - lambda_wd)

            wd.append(wd_i)
            ws.append(ws_i)

        df = DataFrame(vstack((instance, dt, wd, ws, fire_scenario)).T,
                          columns=['Instance', 'datetime', 'WD', 'WS', 'FireScenario'])
        output_path = output_folder / f'weather{str(index).zfill(len(str(n_scenarios)))}.csv'
        df.to_csv(output_path, index=False)