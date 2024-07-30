import seaborn as sns
from matplotlib import pyplot as plt
from matplotlib.lines import Line2D
from sklearn.linear_model import LinearRegression
import fastf1
import fastf1.plotting

# Constants
QUICKLAP_THRESHOLD = 1.05
MARKERS = [".", "*"]
LINES = ["--", ":"]

def load_race_data(race):
    """Load race data."""
    try:
        race.load()
    except Exception as e:
        raise RuntimeError(f"Error loading race data: {e}")

def get_podium_finishers(race):
    """Get the top 2 finishers."""
    return race.drivers[:2]

def get_driver_laps(race, driver):
    """Get the laps for a driver."""
    return race.laps.pick_drivers(driver).pick_quicklaps(QUICKLAP_THRESHOLD)

def get_stints_laps(race):
    """Get the stints laps data."""
    stints_laps = race.laps
    stints = stints_laps[["Driver", "Stint", "LapNumber"]]
    stints = stints.groupby(["Driver", "Stint"]).count().reset_index()
    return stints

def plot_driver_laps(ax, driver_laps, driver_index, driver_color):
    """Plot the driver laps."""
    sns.scatterplot(
        data=driver_laps,
        x="LapNumber",
        y="LapTime(s)",
        ax=ax,
        hue="Compound",
        palette=fastf1.plotting.COMPOUND_COLORS,
        marker=MARKERS[driver_index],
        s=80,
        linewidth=0,
    )

def plot_pit_lap_lines(ax, pit_lap_lines, driver_color):
    """Plot the pit lap lines."""
    for pit_lap_line in pit_lap_lines:
        ax.axvline(x=pit_lap_line, color=driver_color, linestyle="-", linewidth=1.5)

def plot_stint_trendlines(ax, stint_laps, driver_color, driver_index, lines):
    """Plot the stint trendlines."""
    slope_str_array = []
    for stint in stint_laps["Stint"].unique():
        stint_data = stint_laps[stint_laps["Stint"] == stint]
        tire_type = stint_data["Compound"].iloc[0]

        X = stint_data["LapNumber"].values.reshape(-1, 1)
        Y = stint_data["LapTime(s)"].values.reshape(-1, 1)
        reg = LinearRegression().fit(X, Y)
        slope = reg.coef_[0][0]

        sns.regplot(
            x="LapNumber",
            y="LapTime(s)",
            data=stint_data,
            ax=ax,
            scatter=False,
            color=driver_color,
            line_kws={"linestyle": lines[driver_index], "linewidth": 1.4},
        )

        midpoint = (X.min() + X.max()) / 2 - 1 if driver_index == 0 else (X.min() + X.max()) / 2 + 1
        text_y_position = reg.predict([[midpoint]])[0][0]
        slope_str = f"+{slope:.3f} s/lap" if slope > 0 else f"{slope:.3f} s/lap"
        slope_str_array.append(slope_str.replace(" s/lap", ""))
        ax.text(
            midpoint,
            text_y_position,
            slope_str,
            color=driver_color,
            fontsize=10,
            fontweight="bold",
            verticalalignment="bottom",
        )
    return slope_str_array

def plot_annotations(ax, pit_lap_array, driver_laps):
    """Plot pit lap annotations."""
    for pit_lap in pit_lap_array:
        ax.text(
            pit_lap,
            driver_laps["LapTime(s)"].max() - 0.2,
            "Pit Lap",
            rotation=90,
            color="grey",
            fontsize=10,
            fontweight="bold",
            verticalalignment="top",
            horizontalalignment="right",
        )
        ax.axvspan(pit_lap - 0.5, pit_lap + 0.5, color="grey", alpha=0.3)

def set_plot_labels(ax, race):
    """Set plot labels."""
    ax.set_xlabel("Lap Number", fontweight="bold", fontsize=14)
    ax.set_ylabel("Lap Time (s)", fontweight="bold", fontsize=14)
    ax.set_xlim(1, race.laps["LapNumber"].max())

def add_plot_titles(fig, ax, year, event_name, drivers_abbr):
    """Add plot titles and subtitles."""
    suptitle = f"{year} {event_name} Grand Prix Driver Lap Time Variation"
    plt.suptitle(suptitle, fontweight="bold", fontsize=16)

    subtitle_upper = "with Lap Time Variation Rate and Pit Lap Annotated"
    subtitle_lower = f"{drivers_abbr[0]} vs {drivers_abbr[1]}"
    bg_color = ax.get_facecolor()
    plt.figtext(
        0.5,
        0.935,
        subtitle_upper,
        ha="center",
        fontsize=14,
        bbox=dict(facecolor=bg_color, alpha=0.5, edgecolor="none"),
    )
    plt.figtext(
        0.5,
        0.912,
        subtitle_lower,
        ha="center",
        fontsize=12,
        fontweight="bold",
        bbox=dict(facecolor=bg_color, alpha=0.5, edgecolor="none"),
    )

def save_plot(fig, year, event_name):
    """Save the plot to a file."""
    suptitle = f"{year} {event_name} Grand Prix Driver Lap Time Variation"
    filename = f"../pic/{suptitle.replace(' ', '_')}.png"
    plt.savefig(filename)
    return filename

def create_caption(year, event_name, drivers_abbr, pit_lap_caption_arrays, tire_type_arrays, slope_str_arrays):
    """Create a caption for the plot."""
    titles_str = f"{event_name} Grand Prix"
    caption = f"""\
🏎️
« {year} {event_name} Grand Prix »

• {titles_str} {drivers_abbr[0]} vs {drivers_abbr[1]}

‣ Pit Lap
\t◦ {drivers_abbr[0]}
\t1 → {' → '.join(pit_lap_caption_arrays[0])}
\t◦ {drivers_abbr[1]}
\t1 → {' → '.join(pit_lap_caption_arrays[1])}    

‣ Compound
\t◦ {drivers_abbr[0]}
\t{' → '.join(tire_type_arrays[0])}
\t◦ {drivers_abbr[1]}
\t{' → '.join(tire_type_arrays[1])} 
            
‣ Lap Time Variation Rate (s/lap)
\t◦ {drivers_abbr[0]}
\t{' → '.join(slope_str_arrays[0])}
\t◦ {drivers_abbr[1]}
\t{' → '.join(slope_str_arrays[1])}
    
#F1 #Formula1 #{event_name.replace(" ", "")}GP"""
    return caption

def initialize_driver_data():
    """Initialize the driver data dictionary."""
    return {
        "pit_lap_array": [],
        "legend_elements": [],
        "drivers_abbr": [],
        "slope_str_arrays": [],
        "tire_type_arrays": [],
        "pit_lap_caption_arrays": []
    }

def process_driver_data(ax, race, stints, driver, driver_index, driver_data):
    """Process data for a single driver."""
    driver_laps = get_driver_laps(race, driver)
    driver_laps["LapTime(s)"] = driver_laps["LapTime"].dt.total_seconds()
    driver_laps = driver_laps.reset_index()
    driver_abbr = race.get_driver(driver)["Abbreviation"]
    driver_data["drivers_abbr"].append(driver_abbr)
    driver_name = fastf1.plotting.DRIVER_TRANSLATE[driver_abbr]
    driver_color = fastf1.plotting.DRIVER_COLORS[driver_name]
    driver_data["legend_elements"].append(
        Line2D(
            [0], [0], marker=MARKERS[driver_index], color=driver_color, markerfacecolor=driver_color, label=driver_abbr, markersize=10, linestyle=""
        )
    )
    
    stints_stints = stints.loc[stints["Driver"] == driver_abbr]
    pit_lap_lines = []
    pit_lap_caption_array = []
    pit_lap = 0
    for idx, row in stints_stints.iterrows():
        pit_lap += row["LapNumber"]
        if pit_lap not in driver_data["pit_lap_array"]:
            driver_data["pit_lap_array"].append(pit_lap)
        pit_lap_lines.append(pit_lap + 0.25 * driver_index)
        pit_lap_caption_array.append(f"{pit_lap}")
    if driver_data["pit_lap_array"]:
        driver_data["pit_lap_array"].pop()
        pit_lap_lines.pop()

    slope_str_array = plot_stint_trendlines(ax, driver_laps, driver_color, driver_index, LINES)
    driver_data["slope_str_arrays"].append(slope_str_array)
    driver_data["tire_type_arrays"].append([driver_laps[driver_laps["Stint"] == stint]["Compound"].iloc[0] for stint in driver_laps["Stint"].unique()])
    driver_data["pit_lap_caption_arrays"].append(pit_lap_caption_array)

    return driver_laps, pit_lap_lines

def driver_laptimes_scatterplot(year: int, event_name: str, session_name: str, race, post: bool) -> dict:
    """Plot driver lap times variation with pit lap annotations."""
    load_race_data(race)
    podium_finishers = get_podium_finishers(race)
    stints = get_stints_laps(race)
    
    fig, ax = plt.subplots(figsize=(10.8, 10.8), dpi=100)
    driver_data = initialize_driver_data()

    for i, driver in enumerate(podium_finishers):
        driver_laps, pit_lap_lines = process_driver_data(ax, race, stints, driver, i, driver_data)
        plot_driver_laps(ax, driver_laps, i, fastf1.plotting.DRIVER_COLORS[fastf1.plotting.DRIVER_TRANSLATE[race.get_driver(driver)["Abbreviation"]]])
        plot_pit_lap_lines(ax, pit_lap_lines, fastf1.plotting.DRIVER_COLORS[fastf1.plotting.DRIVER_TRANSLATE[race.get_driver(driver)["Abbreviation"]]])

    plot_annotations(ax, driver_data["pit_lap_array"], driver_laps)
    set_plot_labels(ax, race)
    add_plot_titles(fig, ax, year, event_name, driver_data["drivers_abbr"])
    
    sns.despine(left=True, bottom=True)
    plt.tight_layout()
    
    ax.legend(title="Compound", loc="upper right")
    fig.legend(
        title="Drivers",
        handles=driver_data["legend_elements"],
        loc="upper left",
        bbox_to_anchor=(0.08, 0.95),
    )
    
    filename = save_plot(fig, year, event_name)
    caption = create_caption(year, event_name, driver_data["drivers_abbr"], driver_data["pit_lap_caption_arrays"], driver_data["tire_type_arrays"], driver_data["slope_str_arrays"])
    
    return {"filename": filename, "caption": caption, "post": post}
