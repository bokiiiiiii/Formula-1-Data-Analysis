import seaborn as sns
from matplotlib import pyplot as plt
from matplotlib.lines import Line2D
from sklearn.linear_model import LinearRegression

import fastf1
import fastf1.plotting


# @brief driver_laptimes_scatterplot: Plot driver lap times variation with pit lap annotations
def driver_laptimes_scatterplot(Year: int, EventName: str, SessionName: str, race):

    race.load()

    podium_finishers = race.drivers[:2]

    fig, ax = plt.subplots(figsize=(10.8, 10.8), dpi=100)

    markers = [".", "*"]
    lines = ["--", ":"]

    stints_laps = race.laps
    stints = stints_laps[["Driver", "Stint", "LapNumber"]]
    stints = stints.groupby(["Driver", "Stint"])
    stints = stints.count().reset_index()
    print(stints)
    pit_lap_array = []
    legend_elements = []
    for i, driver in enumerate(podium_finishers):
        print(i)
        driver_laps = race.laps.pick_drivers(driver).pick_quicklaps(
            1.03
        )  # Need adjustment
        driver_laps["LapTime(s)"] = driver_laps["LapTime"].dt.total_seconds()

        driver_laps = driver_laps.reset_index()
        driver_lap = driver_laps.pick_fastest()

        sns.scatterplot(
            data=driver_laps,
            x="LapNumber",
            y="LapTime(s)",
            ax=ax,
            hue="Compound",
            palette=fastf1.plotting.COMPOUND_COLORS,
            marker=markers[i],
            s=80,
            linewidth=0,
        )

        driver_abbr = race.get_driver(driver)["Abbreviation"]
        driver_name = fastf1.plotting.DRIVER_TRANSLATE[driver_abbr]
        driver_color = fastf1.plotting.DRIVER_COLORS[driver_name]
        legend_elements.append(
            Line2D(
                [0],
                [0],
                marker=markers[i],
                color=driver_color,
                markerfacecolor=driver_color,
                label=driver_abbr,
                markersize=10,
                linestyle="",
            )
        )

        stints_stints = stints.loc[stints["Driver"] == driver_abbr]

        pit_lap = 0
        pit_lap_line_array = []
        for idx, row in stints_stints.iterrows():
            pit_lap += row["LapNumber"]
            if pit_lap not in pit_lap_array:
                pit_lap_array.append(pit_lap)
            pit_lap_line_array.append(pit_lap + 0.25 * i)
        if pit_lap_array:
            pit_lap_array.pop()
            pit_lap_line_array.pop()

        for pit_lap_line in pit_lap_line_array:
            ax.axvline(x=pit_lap_line, color=driver_color, linestyle="-", linewidth=1.5)

        for stint in driver_laps["Stint"].unique():
            stint_laps = driver_laps[driver_laps["Stint"] == stint]

            X = stint_laps["LapNumber"].values.reshape(-1, 1)
            Y = stint_laps["LapTime(s)"].values.reshape(-1, 1)
            reg = LinearRegression().fit(X, Y)
            slope = reg.coef_[0][0]

            sns.regplot(
                x="LapNumber",
                y="LapTime(s)",
                data=stint_laps,
                ax=ax,
                scatter=False,
                color=driver_color,
                line_kws={"linestyle": lines[i], "linewidth": 1.4},
            )

            if i == 0:
                midpoint = (X.min() + X.max()) / 2 - 1
            else:  # i == 1
                midpoint = (X.min() + X.max()) / 2 + 1
            text_y_position = reg.predict([[midpoint]])[0][0]
            slope_str = f"+{slope:.3f} s/lap" if slope > 0 else f"{slope:.3f} s/lap"
            ax.text(
                midpoint,
                text_y_position,
                slope_str,
                color=driver_color,
                fontsize=10,
                fontweight="bold",
                verticalalignment="bottom",
            )

    for pit_lap in pit_lap_array:
        ax.text(
            pit_lap,
            driver_laps["LapTime(s)"].max() - 0.1,
            "Pit Lap",
            rotation=90,
            color="grey",
            fontsize=10,
            fontweight="bold",
            verticalalignment="top",
            horizontalalignment="right",
        )

        ax.axvspan(pit_lap - 0.5, pit_lap + 0.5, color="grey", alpha=0.3)

    ax.set_xlabel("Lap Number", fontweight="bold", fontsize=14)
    ax.set_ylabel("Lap Time (s)", fontweight="bold", fontsize=14)
    ax.set_xlim(1, race.laps["LapNumber"].max())

    suptitle = f"{Year} {EventName} Grand Prix Driver Lap Time Variation"

    plt.suptitle(
        suptitle,
        fontweight="bold",
        fontsize=16,
    )

    subtitle = "with Lap Time Variation Rate and Pit Lap Annotated"
    plt.figtext(0.5, 0.94, subtitle, ha="center", fontsize=14)

    sns.despine(left=True, bottom=True)

    plt.tight_layout()

    ax.legend(title="Compound", loc="upper right")
    fig.legend(
        title="Drivers",
        handles=legend_elements,
        loc="upper left",
        bbox_to_anchor=(0.08, 0.95),
    )

    filename = "../pic/" + suptitle.replace(" ", "_") + ".png"
    plt.savefig(filename)
