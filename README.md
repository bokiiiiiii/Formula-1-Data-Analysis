# Formula 1 Data Analysis

[![Python](https://img.shields.io/badge/Python-3.8%2B-blue.svg)](https://www.python.org/)
[![FastF1](https://img.shields.io/badge/FastF1-3.5%2B-red.svg)](https://github.com/theOehrly/Fast-F1)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

## Project Description
A comprehensive Formula 1 data analysis and visualization tool that automatically downloads telemetry data, generates insightful analytical plots, and optionally posts them to Instagram. This project leverages the FastF1 API to provide in-depth analysis of F1 races, qualifying sessions, and practice sessions.

*Project Start Date: January 2024*

## Instagram
📸 Follow our analysis: [@f1.data.analysis](https://www.instagram.com/f1.data.analysis/)

## Key Features

### 🖥️ Modern GUI Interface
- Intuitive event selection interface built with CustomTkinter
- Dark mode design for comfortable viewing
- Real-time progress tracking and logging

### 📊 Comprehensive Data Analysis
- **Event Selection**: Interactive GUI for selecting specific Grand Prix events from any F1 season
- **Data Acquisition**: Automatic download of telemetry data, lap times, and session information via FastF1 API
- **Advanced Caching**: Built-in FastF1 cache management for faster subsequent loads
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
- Python 3.8 or higher
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
   - If you want to use the auto-posting feature, create a `config.json` file
   - Add your Instagram credentials (refer to `config.py` for structure)

5. **Run FastF1 Cache Setup**
   ```bash
   # The first run will create a cache directory for faster subsequent loads
   python main.py
   ```

## Usage

### Quick Start

1. **Launch the Application**
   ```bash
   python main.py
   ```

2. **Select Your Event**
   - A GUI window will appear showing available F1 events
   - Choose the Grand Prix you want to analyze
   - Click "Start Analysis" to begin

3. **View Results**
   - Plots will be automatically generated and saved
   - Check the output folder (default: `../Pic`) for images and captions

### Configuration Options

The project uses a flexible configuration system. Key settings in [main.py](main.py):

#### Basic Settings
```python
YEAR = 2024                    # F1 season year
SESSION_NAME = "Q"             # Session(s): "FP1", "FP2", "FP3", "Q", "S", "R", or "FP1+Q+R"
FOLDER_PATH = "../Pic"         # Output directory for plots
```

#### Plot Function Control
```python
ENABLE_ALL = True              # Enable all plotting functions
FUNC_PARAMS = {
    "plot_annotated_qualifying_flying_lap": {"enable": True, "session": "Q"},
    "plot_annotated_race_fastest_lap": {"enable": True, "session": "R"},
    "plot_team_pace_ranking": {"enable": True, "session": "Q"},
    # ... more plot functions
}
```

### Available Sessions
- `FP1`, `FP2`, `FP3` - Free Practice sessions
- `Q` - Qualifying
- `S` - Sprint Qualifying
- `R` - Race
- Combined: `FP1+Q+R` (processes multiple sessions)

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
- **FastF1** (≥3.5): F1 data acquisition and analysis
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

## Features in Detail

### Telemetry Analysis
The project provides detailed telemetry analysis including:
- Speed traces throughout the lap
- Throttle, brake, and gear selection
- DRS activation zones
- Corner-by-corner performance breakdown

### Fuel-Corrected Analysis
Advanced fuel correction algorithms adjust lap times for fuel load, providing:
- Fair performance comparison across the race
- Identification of true pace vs. fuel-affected pace
- Gaussian Process regression for smooth trend analysis

### Monte Carlo Simulation
Probabilistic race strategy simulation considering:
- Pit stop timing variations
- Tire degradation models
- Safety car probabilities
- Strategic decision outcomes

## Troubleshooting

### Common Issues

1. **FastF1 API Errors**
   - The project includes automatic retry logic
   - Check your internet connection
   - Verify the event name and year are correct

2. **Missing Data**
   - Some sessions may not have complete telemetry data
   - Sprint races have different data availability
   - Check FastF1 cache directory

3. **Instagram Posting Fails**
   - Verify credentials in `config.json`
   - Ensure Playwright is properly installed: `playwright install`
   - Check Instagram API rate limits

4. **Memory Issues with Large Datasets**
   - Close other applications
   - Process one session at a time
   - Clear FastF1 cache if needed

## Performance

The project includes performance monitoring:
- Execution time tracking for each plot function
- Memory usage monitoring
- Detailed logging for debugging
- FastF1 cache optimization

## Contributing

Contributions are welcome! Please feel free to submit pull requests or open issues for:
- New plot types
- Performance improvements
- Bug fixes
- Documentation enhancements

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- **FastF1**: For providing the excellent F1 data API
- **F1 Community**: For inspiration and feedback
- **Contributors**: Everyone who has contributed to this project

## Contact

For questions or collaboration:
- Instagram: [@f1.data.analysis](https://www.instagram.com/f1.data.analysis/)
- GitHub: [bokiiiiiii/Formula-1-Data-Analysis](https://github.com/bokiiiiiii/Formula-1-Data-Analysis)

## Changelog

### Recent Updates
- ✨ Added Monte Carlo race strategy simulation
- ✨ Implemented Gaussian Process fuel-corrected analysis
- ✨ Added driver race evolution heatmap
- 🎨 Improved GUI with modern CustomTkinter interface
- ⚡ Enhanced performance monitoring and logging
- 🔄 Added retry logic for network errors
- 📦 Updated dependencies to latest stable versions

---

**Made with ❤️ for F1 fans and data enthusiasts**
