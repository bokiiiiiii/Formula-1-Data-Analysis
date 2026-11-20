# Formula 1 Data Analysis

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://www.python.org/)
[![OpenF1](https://img.shields.io/badge/Data-OpenF1-red.svg)](https://openf1.org/docs/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

## Project Description
A comprehensive Formula 1 data analysis and visualization tool that automatically downloads telemetry data, generates insightful analytical plots, and optionally posts them to Instagram. The app uses the public OpenF1 archive for completed sessions from 2023 onwards. Race data downloads require no account, subscription or API key.

*Project Start Date: January 2024*

## Instagram
📸 Follow our analysis: [@f1.data.analysis](https://www.instagram.com/f1.data.analysis/)

## Key Features

### 🖥️ Modern GUI Interface
- Intuitive event selection interface built with CustomTkinter
- Dark mode design for comfortable viewing
- Real-time progress tracking and logging

### 📊 Comprehensive Data Analysis
- **Event Selection**: Interactive GUI for selecting specific Grand Prix events from the 2023 season onwards
- **Data Acquisition**: Automatic download of telemetry data, lap times, and session information from OpenF1
- **Advanced Caching**: Compressed local cache for historical race data and per-driver telemetry
- **Retry Logic**: Network error handling with automatic retry mechanism

### 📈 Visualization Features
The project generates a wide variety of analytical plots:

#### Track & Lap Analysis
- 🗺️ **Track map with annotated corners** - Visual representation of circuit layouts
- 🏎️ **Qualifying flying lap analysis** - Detailed telemetry breakdown
- 🏁 **Sprint Qualifying flying lap analysis** - Sprint session insights
- 🚀 **Race fastest lap analysis** - Complete telemetry data visualization

#### Performance Metrics
- ⏱️ **Driver lap time distribution** - Statistical lap time analysis
- 📉 **Driver lap time scatter plots** - Lap-by-lap performance visualization
- ⛽ **Fuel-corrected lap times (Scatterplot)** - Performance adjusted for fuel load
- 🧮 **Fuel-corrected lap times (Gaussian Processes)** - Advanced ML-based analysis
- 🌡️ **Driver race evolution heatmap** - Race pace visualization over time

#### Strategic Analysis
- 🏆 **Team pace ranking** - Comparative team performance
- 🎲 **Monte Carlo race strategy simulation** - Probabilistic strategy analysis

#### Vehicle Dynamics
- 🎯 **G-G diagram (friction circle)** - Longitudinal vs lateral acceleration with speed-mapped colours

### 💾 Output Management
- High-quality PNG image export
- Automatic caption generation for social media
- Organized file structure with timestamped outputs

### 📱 Social Media Integration
- Automatic Instagram posting capability
- Custom caption generation for each plot type
- Configurable posting options

## Installation

### Prerequisites
- Python 3.10 or higher
- pip package manager
- Git (for cloning the repository)

### Setup Instructions

1. **Clone the Repository**
   ```bash
   git clone https://github.com/bokiiiiiii/Formula-1-Data-Analysis.git
   cd Formula-1-Data-Analysis
   ```

2. **Create Virtual Environment** (Recommended)
   ```bash
   python -m venv venv
   
   # On Windows
   venv\Scripts\activate
   
   # On macOS/Linux
   source venv/bin/activate
   ```

3. **Install Dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Configure Instagram (Optional)**
   - Install Playwright's Chromium browser in the activated virtual environment:
     ```bash
     python -m playwright install chromium
     ```
   - Run this command again after upgrading Playwright if it reports a missing browser executable.
   - Copy `.env.example` to `.env` next to `main.py` and fill in `INSTAGRAM_USERNAME` and `INSTAGRAM_PASSWORD`. You can also set these as environment variables.
   - For accounts using authenticator-based two-factor authentication, set `INSTAGRAM_2FA_SEED` to the authenticator setup key (not a temporary six-digit code).
   - Credentials are read from `.env`; `config.json` holds analysis settings. An existing authenticated `ig_session.json` can be reused without entering credentials until Instagram requests login again.

## Usage

### Quick Start

1. **Launch the Application**
   ```bash
   python main.py
   ```

2. **Select Your Event**
   - Choose a season to see events with completed sessions
   - Choose the Grand Prix you want to analyze
   - Select one or more completed sessions that have configured charts, then click "START"

3. **View Results**
   - Plots will be automatically generated and saved
   - Check the output folder (default: `../Pic`) for images and captions

### Configuration Options

The GUI controls the season, event, session and optional Instagram posting. The app creates `config.json` on first use. Set `folder_path` for generated PNGs and captions, and `cache_path` for downloaded data. OpenF1 responses are stored under `<cache_path>/openf1/`. The current season's schedule refreshes every five minutes.

Each entry in `plot_functions` assigns a chart to a session code. The archive supports `FP1`, `FP2`, `FP3`, `Q` (qualifying), `SQ` (sprint qualifying), `S` (sprint) and `R` (race). The default chart assignments use FP1, Q, SQ and R.

### Historical telemetry

Lap times and qualifying classification come from OpenF1. The charts integrate speed over time to estimate distance, and align car and position samples by UTC timestamp. Final qualifying comparisons use the classified Q3 or SQ3 time. Circuit outlines use OpenF1 position samples. Corner numbers use the optional MultiViewer circuit metadata referenced by OpenF1. Charts omit those labels when the metadata is unavailable. [OpenF1 data reference](https://openf1.org/docs/)

### Tests

Run `python -m unittest discover -s tests -v` for the offline regression suite. The existing Instagram tests use local Playwright fixtures and require the Chromium installation above. To also render and verify all 13 charts using real archived sessions, set `OPENF1_TEST_CACHE` to a writable cache directory before running the suite.

## Project Structure
```
Formula-1-Data-Analysis/
│
├── main.py                          # Main application entry point
├── config.py                        # Configuration management
├── config.json                      # User configuration file (create manually)
├── logger_config.py                 # Logging setup
├── performance_monitor.py           # Performance tracking utilities
├── retry_utils.py                   # Network retry logic
├── auto_ig_post.py                  # Instagram posting module
├── requirements.txt                 # Python dependencies
├── README.md                        # Project documentation
│
├── plot_functions/                  # Plotting modules
│   ├── __init__.py
│   ├── plot_runner.py              # Plot execution coordinator
│   ├── utils.py                    # Shared utilities
│   │
│   ├── annotated_qualifying_flying_lap.py
│   ├── annotated_race_fatest_lap.py
│   ├── annotated_sprint_qualifying_flying_lap.py
│   ├── driver_fuel_corrected_laptimes_gaussian_processes.py
│   ├── driver_fuel_corrected_laptimes_scatterplot.py
│   ├── driver_laptimes_distribution.py
│   ├── driver_laptimes_scatterplot.py
│   ├── driver_g_g_diagram.py
│   ├── driver_race_evolution_heatmap.py
│   ├── monte_carlo_race_strategy.py
│   ├── plot_track_with_annotated_corners.py
│   ├── race_fatest_lap_telemetry_data.py
│   └── team_pace_ranking.py
│
└── logs/                            # Application logs (auto-generated)
```

## Dependencies

### Core Libraries
- **Requests** for OpenF1 history and optional circuit metadata
- **Matplotlib** (≥3.8): Plotting and visualization
- **Pandas** (≥2.0): Data manipulation
- **NumPy** (≥2.0): Numerical computing
- **SciPy** (≥1.10): Scientific computing

### UI & Automation
- **CustomTkinter** (≥5.2): Modern GUI framework
- **Playwright** (≥1.40): Browser automation for Instagram

### Machine Learning
- **Scikit-learn** (≥1.3): ML algorithms (Gaussian Processes)
- **Seaborn** (≥0.13): Statistical data visualization

### Utilities
- **Pillow** (≥10.0): Image processing
- **python-dotenv** (≥1.0): Environment variable management
- **pydantic** (≥2.0): Data validation

See [requirements.txt](requirements.txt) for complete dependency list.

## License

This project is licensed under the MIT License - see the LICENSE file for details.
