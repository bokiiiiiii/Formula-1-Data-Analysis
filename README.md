# Formula 1 Data Analysis

## Project Description
This project aims to analyze and visualize Formula 1 race data. Users can select specific events and sessions (e.g., Practice, Qualifying, Race), and the program will automatically download relevant data, generate various analytical plots, and optionally auto-post them to a specified Instagram account.

*Project Start Date: January 2024*

## Instagram
f1.data.analysis [Link](https://www.instagram.com/f1.data.analysis/)

## Key Features
- **Event Selection**: Provides a Graphical User Interface (GUI) for users to select specific Grand Prix events from an F1 season.
- **Data Acquisition**: Automatically downloads telemetry data, lap times, session information, etc., for the selected event using the FastF1 API.
- **Data Visualization**: Generates a variety of analytical plots, including:
    - Track map with annotated corners
    - Qualifying/Sprint Qualifying flying lap analysis (with telemetry)
    - Race fastest lap analysis (with telemetry)
    - Driver lap time distribution plots
    - Driver lap time scatter plots
    - Team pace ranking
    - Driver fuel-corrected lap time scatter plots
- **Image Saving**: Saves the generated plots as PNG image files.
- **Instagram Auto-Posting**: Automatically generates captions for the plots and can optionally post them to Instagram.

## How to Use
1. **Environment Setup**:
   - Ensure you have a Python environment installed.
   - Install the necessary libraries:
     ```bash
     pip install -r requirements.txt
     ```
   - (If auto-posting to Instagram is needed) Configure the Instagram login credentials and relevant APIs required by `auto_ig_post.py`.

2. **Run the Project**:
   - Execute the `main.py` file:
     ```bash
     python main.py
     ```
   - After starting, a window will pop up allowing you to select the F1 event to analyze.
   - Once an event is selected, the program will begin downloading data and generating plots.

3. **Configuration**:
   - `YEAR`: Set the year to analyze in `main.py`.
   - `SESSION_NAME`: Set the session(s) to analyze in `main.py` (e.g., "FP1", "Q", "R", "FP1+Q+R").
   - `ENABLE_ALL`: Set in `main.py` to enable all plotting functions.
   - `FUNC_PARAMS`: Fine-tune the enabled status and corresponding session for each plotting function in `main.py`.
   - `FOLDER_PATH`: Set the save path for images and text files in `main.py` (default is `../Pic`).

4. **Output**:
   - Generated plots will be saved in the folder specified by `FOLDER_PATH`.
   - Instagram post captions will be saved in `.txt` files corresponding to the image names.

## Project Structure
```
Formula-1-Data-Analysis/
│
├── .gitignore
├── auto_ig_post.py         # Instagram auto-posting module
├── main.py                 # Main program entry point
├── README.md               # Project documentation
│
└── plot_functions/         # Modules for various plotting functions
    ├── __init__.py
    ├── annotated_qualifying_flying_lap.py
    ├── annotated_race_fatest_lap.py
    ├── annotated_sprint_qualifying_flying_lap.py
    ├── driver_fuel_corrected_laptimes_scatterplot.py
    ├── driver_laptimes_distribution.py
    ├── driver_laptimes_scatterplot.py
    ├── plot_track_with_annotated_corners.py
    ├── race_fatest_lap_telemetry_data.py
    └── team_pace_ranking.py
```
