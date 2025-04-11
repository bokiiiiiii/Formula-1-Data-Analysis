import seaborn as sns
from matplotlib import pyplot as plt
from matplotlib.lines import Line2D
from sklearn.linear_model import LinearRegression
import fastf1
import fastf1.plotting
import numpy as np
import pandas as pd
from collections import OrderedDict

# Parameters
QUICKLAP_THRESHOLD = 1.03
CORRECTED_LAPTIME = 0.05
MARKERS = [".", "*"]
LINES = ["--", ":"]


def load_race_data(race):
    try:
        race.load()
    except Exception as e:
        raise RuntimeError(f"Error loading race data: {e}")


def get_podium_finishers(race):
    return race.drivers[:2]


def get_driver_laps(race, driver):
    driver_laps = (
        race.laps.pick_drivers(driver).pick_quicklaps(QUICKLAP_THRESHOLD).copy()
    )
    driver_laps["LapTime(s)"] = driver_laps["LapTime"].dt.total_seconds()
    driver_laps["FuelCorrectedLapTime(s)"] = (
        driver_laps["LapTime(s)"] + CORRECTED_LAPTIME * driver_laps["LapNumber"]
    )
    driver_laps["StintLapNumber"] = driver_laps.groupby("Stint").cumcount() + 1
    return driver_laps


def get_stints_laps(race):
    stints_laps = race.laps
    stints = stints_laps[["Driver", "Stint", "LapNumber"]]
    stints = stints.groupby(["Driver", "Stint"]).count().reset_index()
    return stints


def plot_driver_laps(ax, driver_laps, driver_index, driver_color):
    sns.scatterplot(
        data=driver_laps,
        x="StintLapNumber",
        y="FuelCorrectedLapTime(s)",
        ax=ax,
        hue="Compound",
        palette=fastf1.plotting.COMPOUND_COLORS,
        marker=MARKERS[driver_index],
        s=80,
        linewidth=0,
    )


def plot_stint_trendlines(ax, driver_laps, driver_color, driver_index, lines):
    slope_dict = OrderedDict()
    compound_groups = driver_laps.groupby("Compound")

    for compound, group in compound_groups:
        X = group["StintLapNumber"].values.reshape(-1, 1)
        Y = group["FuelCorrectedLapTime(s)"].values.reshape(-1, 1)

        reg = LinearRegression().fit(X, Y)
        slope = reg.coef_[0][0]
        slope_str = f"+{slope:.3f}" if slope > 0 else f"{slope:.3f}"
        slope_dict[compound] = slope_str

        x_min = group["StintLapNumber"].min()
        x_max = group["StintLapNumber"].max()
        x_vals = np.linspace(x_min, x_max, 100).reshape(-1, 1)
        y_vals = reg.predict(x_vals)

        y_true = Y
        y_pred = reg.predict(X)
        residuals = y_true - y_pred
        mse = np.mean(residuals**2)
        se = np.sqrt(mse)
        ci = 0.2 * se
        y_lower = y_vals.flatten() - ci
        y_upper = y_vals.flatten() + ci

        compound_color = fastf1.plotting.COMPOUND_COLORS.get(compound, "black")
        ax.plot(
            x_vals,
            y_vals,
            linestyle=lines[driver_index],
            linewidth=1.4,
            color=compound_color,
        )
        ax.fill_between(
            x_vals.flatten(),
            y_lower,
            y_upper,
            color=driver_color,
            alpha=0.2,
            linewidth=0,
        )

        midpoint = (x_min + x_max) / 2
        y_mid = reg.predict([[midpoint]])[0][0]
        ax.text(
            midpoint,
            y_mid,
            f"{slope_str} s/lap",
            color=driver_color,
            fontsize=10,
            fontweight="bold",
            verticalalignment="bottom",
        )

    return slope_dict


def set_plot_labels(ax, race):
    ax.set_xlabel("Stint Lap Number", fontweight="bold", fontsize=14)
    ax.set_ylabel("Fuel Corrected Lap Time (s)", fontweight="bold", fontsize=14)


def add_plot_titles(fig, ax, year, event_name, drivers_abbr):
    suptitle = f"{year} {event_name} Grand Prix Driver Lap Time Variation"
    plt.suptitle(suptitle, fontweight="bold", fontsize=16)

    subtitle_upper = "with Fuel-Corrected Lap Time Variation Rate"
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
    suptitle = f"{year} {event_name} Grand Prix Driver Lap Time Variation"
    filename = f"../pic/{suptitle.replace(' ', '_')}.png"
    plt.savefig(filename)
    return filename


def create_caption(year, event_name, drivers_abbr, tire_type_arrays, slope_str_arrays):
    titles_str = f"{event_name} Grand Prix"

    compound_slope_str = ""
    for i in range(2):
        compound_slope_str += f"\t◦ {drivers_abbr[i]}\n"
        compound_slope_str += (
            "\t"
            + " | ".join(
                f"{compound} : {slope}"
                for compound, slope in zip(tire_type_arrays[i], slope_str_arrays[i])
            )
            + "\n"
        )

    caption = f"""

🏎️
« {year} {event_name} Grand Prix »

• {titles_str} {drivers_abbr[0]} vs {drivers_abbr[1]}

‣ Compound : Fuel-Corrected Lap Time Variation Rate (s/lap)
{compound_slope_str}#F1 #Formula1 #{event_name.replace(" ", "")}GP"""

    return caption


def initialize_driver_data():
    return {
        "legend_elements": [],
        "drivers_abbr": [],
        "slope_str_arrays": [],
        "tire_type_arrays": [],
    }


def process_driver_data(ax, race, stints, driver, driver_index, driver_data):
    driver_laps = get_driver_laps(race, driver).reset_index()
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

    slope_dict = plot_stint_trendlines(
        ax, driver_laps, driver_color, driver_index, LINES
    )
    driver_data["slope_str_arrays"].append(list(slope_dict.values()))
    driver_data["tire_type_arrays"].append(list(slope_dict.keys()))

    return driver_laps


def driver_fuel_corrected_laptimes_scatterplot(
    year: int, event_name: str, session_name: str, race, post: bool
) -> dict:
    load_race_data(race)
    podium_finishers = get_podium_finishers(race)
    stints = get_stints_laps(race)

    fig, ax = plt.subplots(figsize=(10.8, 10.8), dpi=100)
    driver_data = initialize_driver_data()

    for i, driver in enumerate(podium_finishers):
        driver_laps = process_driver_data(ax, race, stints, driver, i, driver_data)
        driver_abbr = race.get_driver(driver)["Abbreviation"]
        driver_color = fastf1.plotting.get_driver_style(
            driver_abbr, style="color", session=race
        )["color"]
        plot_driver_laps(ax, driver_laps, i, driver_color)

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
    caption = create_caption(
        year,
        event_name,
        driver_data["drivers_abbr"],
        driver_data["tire_type_arrays"],
        driver_data["slope_str_arrays"],
    )

    return {"filename": filename, "caption": caption, "post": post}
