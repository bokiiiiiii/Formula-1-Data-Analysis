import seaborn as sns
from matplotlib import pyplot as plt
from matplotlib.lines import Line2D
from sklearn.linear_model import LinearRegression
import fastf1
import fastf1.plotting

# Parameters
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
    """Get the laps for a driver, with standardized Compound names."""
    laps_orig = race.laps.pick_drivers(driver).pick_quicklaps(QUICKLAP_THRESHOLD)
    laps = laps_orig.copy()

    if "LapTime(s)" not in laps.columns:
        laps.loc[:, "LapTime(s)"] = laps["LapTime"].dt.total_seconds()

    if "Compound" in laps.columns:

        laps.loc[:, "Compound"] = laps["Compound"].fillna("TEMP_NAN_FOR_UNKNOWN")
        laps.loc[:, "Compound"] = laps["Compound"].astype(str)

        replace_map = {
            "UNKOWN": "UNKNOWN",
            "TEMP_NAN_FOR_UNKNOWN": "UNKNOWN",
            "nan": "UNKNOWN",
            "None": "UNKNOWN",
            "": "UNKNOWN",
        }
        laps.loc[:, "Compound"] = laps["Compound"].replace(replace_map)

        plot_handled_compounds = ["SOFT", "MEDIUM", "HARD", "INTERMEDIATE"]
        compounds_to_keep_as_is = plot_handled_compounds + ["UNKNOWN"]
        mask_to_change_to_unknown = ~laps["Compound"].isin(compounds_to_keep_as_is)
        laps.loc[mask_to_change_to_unknown, "Compound"] = "UNKNOWN"
    else:
        laps.loc[:, "Compound"] = "UNKNOWN"

    return laps


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
        palette={**fastf1.plotting.COMPOUND_COLORS, "UNKNOWN": "grey"},
        hue_order=["SOFT", "MEDIUM", "HARD", "INTERMEDIATE", "UNKNOWN"],
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
        if stint_data.empty:
            continue
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

        midpoint = (
            (X.min() + X.max()) / 2 - 1
            if driver_index == 0
            else (X.min() + X.max()) / 2 + 1
        )
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


def set_plot_labels(ax, race, min_time=None, max_time=None):
    """Set plot labels and axis limits."""
    ax.set_xlabel("Lap Number", fontweight="bold", fontsize=14)
    ax.set_ylabel("Lap Time (s)", fontweight="bold", fontsize=14)
    ax.set_xlim(0.5, race.laps["LapNumber"].max() + 0.5)
    if min_time is not None and max_time is not None:
        padding = (max_time - min_time) * 0.05
        ax.set_ylim(min_time - padding, max_time + padding)


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


def create_caption(
    year,
    event_name,
    drivers_abbr,
    pit_lap_caption_arrays,
    tire_type_arrays,
    slope_str_arrays,
):
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
        "pit_lap_caption_arrays": [],
    }


def process_driver_data(
    ax, race, stints, driver, driver_index, driver_data, driver_laps
):
    """Process data for a single driver."""
    driver_laps = driver_laps.copy().reset_index()
    driver_abbr = race.get_driver(driver)["Abbreviation"]
    driver_data["drivers_abbr"].append(driver_abbr)
    driver_name = fastf1.plotting.DRIVER_TRANSLATE[driver_abbr]
    driver_color = fastf1.plotting.get_driver_style(
        driver_name, style="color", session=race
    )["color"]

    driver_data["legend_elements"].append(
        Line2D(
            [0],
            [0],
            marker=MARKERS[driver_index],
            color=driver_color,
            markerfacecolor=driver_color,
            label=driver_abbr,
            markersize=10,
            linestyle="",
        )
    )

    stints_stints = stints.loc[stints["Driver"] == driver_abbr]

    all_stint_end_laps_for_driver = []
    current_pit_lap = 0
    for idx, row in stints_stints.iterrows():
        current_pit_lap += row["LapNumber"]
        all_stint_end_laps_for_driver.append(current_pit_lap)

    pit_lap_lines = []
    actual_pit_stop_laps = []
    if len(all_stint_end_laps_for_driver) > 0:
        actual_pit_stop_laps = all_stint_end_laps_for_driver[:-1]
        for p_lap in actual_pit_stop_laps:
            pit_lap_lines.append(p_lap + 0.25 * driver_index)
            if p_lap not in driver_data["pit_lap_array"]:
                driver_data["pit_lap_array"].append(p_lap)
    elif stints_stints.empty:
        pass
    else:
        pass

    pit_lap_caption_array = [f"{lap}" for lap in all_stint_end_laps_for_driver]
    driver_data["pit_lap_caption_arrays"].append(pit_lap_caption_array)

    slope_str_array = plot_stint_trendlines(
        ax, driver_laps, driver_color, driver_index, LINES
    )
    driver_data["slope_str_arrays"].append(slope_str_array)
    tire_type_array = []
    for stint_num in driver_laps["Stint"].unique():
        stint_data = driver_laps[driver_laps["Stint"] == stint_num]
        if not stint_data.empty:
            tire_type_array.append(stint_data["Compound"].iloc[0])
        else:
            tire_type_array.append("UNKNOWN")
    driver_data["tire_type_arrays"].append(tire_type_array)

    return driver_laps, pit_lap_lines


def driver_laptimes_scatterplot(
    year: int, event_name: str, session_name: str, race, post: bool
) -> dict:
    """Plot driver lap times variation with pit lap annotations."""
    load_race_data(race)
    podium_finishers = get_podium_finishers(race)
    stints = get_stints_laps(race)

    all_driver_laps_data = {}
    all_times = []
    for driver in podium_finishers:
        laps_data = get_driver_laps(race, driver)
        all_driver_laps_data[driver] = laps_data
        all_times.extend(laps_data["LapTime(s)"].dropna().tolist())

    min_lap_time = min(all_times) if all_times else 0
    max_lap_time = max(all_times) if all_times else 1

    fig, ax = plt.subplots(figsize=(10.8, 10.8), dpi=100)
    driver_data = initialize_driver_data()
    last_driver_laps = None

    for i, driver in enumerate(podium_finishers):
        current_driver_laps = all_driver_laps_data[driver]

        _, pit_lap_lines = process_driver_data(
            ax, race, stints, driver, i, driver_data, current_driver_laps
        )

        driver_abbr = race.get_driver(driver)["Abbreviation"]
        driver_color = fastf1.plotting.get_driver_style(
            driver_abbr, style="color", session=race
        )["color"]

        plot_driver_laps(ax, current_driver_laps, i, driver_color)
        plot_pit_lap_lines(ax, pit_lap_lines, driver_color)
        last_driver_laps = current_driver_laps

    if last_driver_laps is not None:
        plot_annotations(ax, driver_data["pit_lap_array"], last_driver_laps)

    set_plot_labels(ax, race, min_lap_time, max_lap_time)
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
    caption = create_caption(
        year,
        event_name,
        driver_data["drivers_abbr"],
        driver_data["pit_lap_caption_arrays"],
        driver_data["tire_type_arrays"],
        driver_data["slope_str_arrays"],
    )

    return {"filename": filename, "caption": caption, "post": post}
