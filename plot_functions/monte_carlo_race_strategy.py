import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import fastf1
import fastf1.plotting
import pandas as pd
import textwrap
import os
import warnings
import itertools
import scienceplots
from scipy import stats
from matplotlib.ticker import MaxNLocator
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

# Advanced Fitting
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, WhiteKernel, ConstantKernel as C
from sklearn.preprocessing import StandardScaler

# Filter warnings
warnings.filterwarnings("ignore")

# Global Constants
DPI = 125
FIG_SIZE = (1080 / DPI, 1350 / DPI)
DEFAULT_CORRECTION = 0.06

# Default Data Fallbacks
DEFAULT_STINT_STATS = {
    "S": {"mean": 15, "max": 20},
    "M": {"mean": 25, "max": 35},
    "H": {"mean": 40, "max": 55},
    "I": {"mean": 25, "max": 40},
    "W": {"mean": 25, "max": 40},
}

# Default Limits for a Q3 Qualifier (starting P1-P10)
Q3_DEFAULT_LIMITS = {"S": 1, "M": 2, "H": 2}

DEFAULT_PHYSICS = {
    "S": {"base_pace": 90.0, "deg_per_lap": 0.15},
    "M": {"base_pace": 90.6, "deg_per_lap": 0.10},
    "H": {"base_pace": 91.2, "deg_per_lap": 0.05},
    "I": {"base_pace": 100.0, "deg_per_lap": 0.05},
    "W": {"base_pace": 105.0, "deg_per_lap": 0.05},
}


def generate_linear_curve(base_pace, deg_per_lap, max_laps=100):
    return np.array([base_pace + deg_per_lap * i for i in range(max_laps)])


DEFAULT_CURVES = {
    k: generate_linear_curve(v["base_pace"], v["deg_per_lap"])
    for k, v in DEFAULT_PHYSICS.items()
}

# ==========================================
# PART 1: Helper Functions (Data & IO)
# ==========================================


def load_race_data(race):
    try:
        race.load()
    except Exception as e:
        raise RuntimeError(f"Error loading race data: {e}")


def get_winner_abbr(race):
    try:
        return race.results.iloc[0]["Abbreviation"]
    except:
        return None


def save_plot_and_get_filename(fig, suptitle_text_param, dpi_val):
    if not os.path.exists("../pic"):
        os.makedirs("../pic", exist_ok=True)
    filename_safe_title = (
        suptitle_text_param.replace(" ", "_").replace(":", "").replace("/", "_")
    )
    filename = f"../pic/{filename_safe_title}.png"
    fig.savefig(filename, dpi=dpi_val)
    return filename


def create_styled_caption_monte_carlo(
    year_val,
    event_name_val,
    n_sims,
    best_strategy_name,
    deg_source,
    driver,
    fuel_k,
    pit_loss,
):
    caption_parts = [
        "🏎️",
        f"« {year_val} {event_name_val} Grand Prix »",
        "",
        f"• {driver} Race Analysis: Monte Carlo Simulations Top 5 Options vs Actual",
        "",
        f"‣ Simulation Details:",
        f"\t◦ Driver: {driver}",
        f"\t◦ Iterations: {n_sims} runs per strategy",
        f"\t◦ Recommended: {best_strategy_name}",
        "",
        "‣ Advanced Models:",
        f"\t◦ Fuel Correction Factor: {fuel_k:.3f}s/lap",
        f"\t◦ Avg Pit Loss: {pit_loss:.1f}s",
        "",
        f"#F1 #Formula1 #{event_name_val.replace(' ', '')}GP #MonteCarlo #StrategyAnalysis",
    ]
    return textwrap.dedent("\n".join(caption_parts))


def get_driver_actual_strategy(race, driver_abbr):
    if race.laps is None or race.laps.empty:
        return "No data"
    laps = race.laps.pick_driver(driver_abbr).copy()
    laps["Compound"] = laps["Compound"].fillna("UNKNOWN")
    compound_map = {
        "SOFT": "S",
        "MEDIUM": "M",
        "HARD": "H",
        "INTERMEDIATE": "I",
        "WET": "W",
        "UNKNOWN": "?",
    }
    stints = laps.groupby("Stint")["Compound"].first()
    codes = [compound_map.get(c.upper(), "?") for c in stints]
    if codes:
        return f"{driver_abbr} Actual Choice: {'-'.join(codes)}"
    return f"{driver_abbr} Actual Choice: N/A"


def get_actual_tyre_counts(race, driver_abbr):
    if race.laps is None or race.laps.empty:
        return {"S": 0, "M": 0, "H": 0}
    laps = race.laps.pick_driver(driver_abbr)
    stints = laps.groupby("Stint")["Compound"].first()
    counts = {"S": 0, "M": 0, "H": 0}
    compound_map = {"SOFT": "S", "MEDIUM": "M", "HARD": "H"}
    for comp in stints:
        c_code = compound_map.get(comp.upper())
        if c_code in counts:
            counts[c_code] += 1
    return counts


# ==========================================
# PART 2: Advanced Physics (Optimized Logic)
# ==========================================


def prepare_race_data(race):
    """Preprocesses laps: filtering, time conversion, and stint indexing."""
    laps = race.laps.pick_accurate().pick_track_status("1").copy()
    if laps.empty:
        return laps

    laps["LapTime(s)"] = laps["LapTime"].dt.total_seconds()
    laps["StintLapNumber"] = laps.groupby("Stint")["LapNumber"].rank(
        method="first", ascending=True
    )

    if "TyreLife" not in laps.columns:
        laps["TyreLife"] = laps["StintLapNumber"]
    else:
        laps["TyreLife"] = laps["TyreLife"].fillna(laps["StintLapNumber"])
    return laps


def calculate_actual_stint_stats(race):
    """Optimized Robust Mean/Max stint length calculation."""
    laps = race.laps.pick_accurate().pick_track_status("1")
    stats_data = {}

    for comp in ["SOFT", "MEDIUM", "HARD"]:
        code = comp[0]
        comp_laps = laps[laps["Compound"] == comp]

        if comp_laps.empty:
            stats_data[code] = DEFAULT_STINT_STATS[code]
            continue

        counts = comp_laps.groupby(["Driver", "Stint"])["LapNumber"].count()
        valid = counts[counts > 3]

        if not valid.empty:
            # IQR Filtering
            q1, q3 = valid.quantile(0.25), valid.quantile(0.75)
            valid = valid[(valid >= q1 - 1.5 * (q3 - q1))]

        if valid.empty:
            valid = (
                counts[counts > 3]
                if not counts.empty
                else pd.Series([DEFAULT_STINT_STATS[code]["mean"]])
            )

        stats_data[code] = {"mean": valid.mean(), "max": valid.max()}

    return stats_data


def calculate_avg_pit_loss(race):
    laps = race.laps.pick_accurate().pick_track_status("1")
    if laps.empty:
        return 22.0

    base_pace = (
        laps[laps["PitInTime"].isna() & laps["PitOutTime"].isna()]["LapTime"]
        .dt.total_seconds()
        .median()
    )
    if np.isnan(base_pace):
        return 22.0

    in_loss = (
        laps[~laps["PitInTime"].isna()]["LapTime"].dt.total_seconds() - base_pace
    ).median()
    out_loss = (
        laps[~laps["PitOutTime"].isna()]["LapTime"].dt.total_seconds() - base_pace
    ).median()

    total = (in_loss if not np.isnan(in_loss) else 4.0) + (
        out_loss if not np.isnan(out_loss) else 18.0
    )

    if np.isnan(total) or total < 15 or total > 40:
        return 22.0
    return total


def optimize_fuel_correction(laps_df):
    """Optimized fuel correction calculation."""
    if laps_df.empty:
        return DEFAULT_CORRECTION

    counts = laps_df.groupby("Compound")["Stint"].nunique()
    if not any(counts > 1):
        return DEFAULT_CORRECTION

    best_k, min_score = DEFAULT_CORRECTION, float("inf")

    # Vectorized trial
    for k in np.linspace(0.01, 0.08, 36):
        laps_df["_tmp"] = laps_df["LapTime(s)"] + k * (laps_df["LapNumber"] - 1)
        rmsd_sum, n = 0, 0

        repeat_compounds = counts[counts > 1].index
        for comp in repeat_compounds:
            c_data = laps_df[laps_df["Compound"] == comp]
            stints = c_data["Stint"].unique()
            for i in range(len(stints)):
                s1 = c_data[c_data["Stint"] == stints[i]].set_index("StintLapNumber")[
                    "_tmp"
                ]
                for j in range(i + 1, len(stints)):
                    s2 = c_data[c_data["Stint"] == stints[j]].set_index(
                        "StintLapNumber"
                    )["_tmp"]
                    common = s1.index.intersection(s2.index)
                    if len(common) >= 3:
                        rmsd_sum += np.sqrt(
                            np.mean((s1.loc[common] - s2.loc[common]) ** 2)
                        )
                        n += 1

        if n > 0:
            score = rmsd_sum / n
            if score < min_score:
                min_score, best_k = score, k

    return best_k


def calculate_race_degradation_curves(race, stint_stats):
    """Optimized GPR Curve Generation."""
    laps = prepare_race_data(race)
    if laps.empty:
        return DEFAULT_CURVES, "Default (No Clean Laps)", DEFAULT_CORRECTION

    fuel_k = optimize_fuel_correction(laps)
    laps["FuelCorrected"] = laps["LapTime(s)"] + fuel_k * (laps["LapNumber"] - 1)

    curves = {}
    MAX_LAPS = 100

    for comp in ["SOFT", "MEDIUM", "HARD"]:
        code = comp[0]
        c_laps = laps[laps["Compound"] == comp]

        if len(c_laps) < 8:
            curves[code] = None
            continue

        try:
            X = c_laps["TyreLife"].values.reshape(-1, 1)
            y = c_laps["FuelCorrected"].values.reshape(-1, 1)

            scaler_X, scaler_y = StandardScaler(), StandardScaler()
            gpr = GaussianProcessRegressor(
                kernel=C(1.0) * RBF(1.0) + WhiteKernel(0.1), n_restarts_optimizer=5
            )
            gpr.fit(scaler_X.fit_transform(X), scaler_y.fit_transform(y))

            pred = scaler_y.inverse_transform(
                gpr.predict(
                    scaler_X.transform(np.arange(MAX_LAPS).reshape(-1, 1))
                ).reshape(-1, 1)
            ).flatten()

            if code == "S":  # Cliff logic
                limit = stint_stats["S"]["max"]
                over = np.arange(MAX_LAPS) > limit
                pred[over] += 0.05 * np.power(np.arange(MAX_LAPS)[over] - limit, 1.8)

            curves[code] = pred
        except:
            curves[code] = None

    # Fill missing curves using relative scaling
    avail = [k for k, v in curves.items() if v is not None]
    if not avail:
        return DEFAULT_CURVES, "Default (GPR Failed)", fuel_k

    ref = avail[0]
    ref_curve = curves[ref]

    for code in ["S", "M", "H", "I", "W"]:
        if curves.get(code) is None:
            ratio = (
                DEFAULT_PHYSICS.get(code, DEFAULT_PHYSICS["M"])["deg_per_lap"]
                / DEFAULT_PHYSICS.get(ref, DEFAULT_PHYSICS["M"])["deg_per_lap"]
            )
            base_diff = (
                DEFAULT_PHYSICS.get(code, DEFAULT_PHYSICS["M"])["base_pace"]
                - DEFAULT_PHYSICS.get(ref, DEFAULT_PHYSICS["M"])["base_pace"]
            )
            curves[code] = ref_curve[0] + base_diff + (ref_curve - ref_curve[0]) * ratio

    return curves, "Real Race (GPR+Cliff)", fuel_k


# ==========================================
# PART 3: Dynamic Strategy Generation (Optimized)
# ==========================================


def validate_strategy_with_inventory(strategy_compounds, actual_used_counts=None):
    counts = {"H": 0, "M": 0, "S": 0}
    for c in strategy_compounds:
        if c in counts:
            counts[c] += 1

    limit_H, limit_M, limit_S = (
        Q3_DEFAULT_LIMITS["H"],
        Q3_DEFAULT_LIMITS["M"],
        Q3_DEFAULT_LIMITS["S"],
    )

    if actual_used_counts:
        limit_H = max(limit_H, actual_used_counts.get("H", 0))
        limit_M = max(limit_M, actual_used_counts.get("M", 0))
        limit_S = max(limit_S, actual_used_counts.get("S", 0))

    return not (counts["H"] > limit_H or counts["M"] > limit_M or counts["S"] > limit_S)


def generate_all_possible_strategies(total_laps, stint_stats):
    compounds = ["S", "M", "H"]
    strategies = []

    for r in [2, 3]:
        for comps in itertools.product(compounds, repeat=r):
            # Calculate pit laps based on mean stint stats
            means = [stint_stats[c]["mean"] for c in comps]
            total_mean = sum(means)
            ratio = total_laps / total_mean

            stint_lens = [max(5, int(m * ratio)) for m in means]

            # Validation: Stint limit
            if any(
                l > stint_stats[comps[i]]["max"] * 1.10
                for i, l in enumerate(stint_lens)
            ):
                continue

            pit_laps = np.cumsum(stint_lens[:-1]).tolist()
            if any(p >= total_laps or p <= 0 for p in pit_laps):
                continue

            strategies.append(
                {
                    "name": f"{len(comps)-1}-Stop ({'-'.join(comps)})",
                    "compounds": list(comps),
                    "pit_laps": pit_laps,
                }
            )
    return strategies


# ==========================================
# PART 4: Simulation Engine (Optimized)
# ==========================================


class RaceStrategySimulator:
    def __init__(
        self,
        deg_curves,
        total_laps,
        pit_loss,
        lap_time_std=0.3,
        fuel_correction_factor=0.06,
    ):
        self.total_laps = int(total_laps)
        self.pit_loss = pit_loss
        self.deg_curves = deg_curves
        self.fuel_k = fuel_correction_factor
        self.lap_time_std = lap_time_std

    def simulate_strategy(self, strategy_name, compounds, pit_laps, n_sims=1000):
        total_race_times = np.zeros(n_sims) + 4.5  # Standing start loss

        # Pre-calculate SC params
        has_sc = np.random.rand(n_sims) < 0.4
        sc_start = np.random.randint(2, self.total_laps - 5, n_sims)
        sc_dur = np.random.randint(3, 6, n_sims)

        stint_idx = 0
        current_tyre = compounds[0]
        age = np.zeros(n_sims, dtype=int)
        curve = self.deg_curves.get(current_tyre, self.deg_curves.get("M"))
        traffic_penalty = np.zeros(n_sims)

        for lap_number in range(1, self.total_laps + 1):
            is_sc = has_sc & (lap_number >= sc_start) & (lap_number < sc_start + sc_dur)

            if lap_number in pit_laps:
                loss = np.where(is_sc, self.pit_loss * 0.6, self.pit_loss)
                total_race_times += loss + np.random.exponential(0.6, n_sims)

                # Traffic injection
                caught = np.random.rand(n_sims) < 0.6
                traffic_penalty[caught] += np.random.randint(3, 8, np.sum(caught))

                stint_idx += 1
                current_tyre = (
                    compounds[stint_idx]
                    if stint_idx < len(compounds)
                    else compounds[-1]
                )
                age[:] = 0
                curve = self.deg_curves.get(current_tyre, self.deg_curves.get("M"))
            else:
                raw_lap = curve[np.clip(age, 0, len(curve) - 1)]

                # Traffic handling
                delay = np.zeros(n_sims)
                in_traffic = traffic_penalty > 0
                delay[in_traffic] = np.random.uniform(0.5, 1.2, np.sum(in_traffic))
                traffic_penalty[in_traffic] -= 1

                lap_val = (
                    raw_lap
                    - (self.fuel_k * lap_number)
                    + delay
                    + np.random.normal(0, self.lap_time_std, n_sims)
                )
                total_race_times += lap_val

                age += np.where(is_sc, 0, 1)  # SC saves tyres

        return total_race_times


# ==========================================
# PART 5: Visualization
# ==========================================


def plot_strategy_distribution_styled(
    ax, results_dict, total_laps, driver_actual_text, driver_abbr
):
    # Use SciencePlots compatible colors
    prop_cycle = plt.rcParams["axes.prop_cycle"]
    colors = prop_cycle.by_key()["color"]
    # Ensure enough colors
    while len(colors) < len(results_dict):
        colors += colors

    sorted_items = sorted(results_dict.items(), key=lambda x: np.percentile(x[1], 50))

    for idx, (name, data) in enumerate(sorted_items):
        sns.histplot(
            data=data,
            ax=ax,
            kde=True,
            element="step",
            stat="percent",
            color=colors[idx],
            alpha=0.25,
            linewidth=0,
            label=name,
        )
        median_val = np.median(data)
        ax.axvline(
            median_val, color=colors[idx], linestyle="--", linewidth=1.5, alpha=0.9
        )

    ax.set_xlabel("Total Race Time (s)", fontsize=14, color="black")
    ax.set_ylabel(r"Probability (\%)", fontsize=14, color="black")
    ax.tick_params(axis="both", which="major", colors="black")

    # Matched Grid Style
    ax.grid(True, which="major", linestyle="--", linewidth=0.5, alpha=0.5, zorder=0)

    # Matched Legend Style
    leg = ax.legend(
        title=f"{driver_abbr} Top 5 Options",
        loc="upper right",
        fontsize=10,
        title_fontsize=12,
        frameon=True,
        facecolor=ax.get_facecolor(),
        edgecolor=ax.get_facecolor(),
        framealpha=0.5,
        labelcolor="black",
    )
    leg.get_title().set_color("black")

    if driver_actual_text:
        ax.text(
            0.02,
            0.98,
            driver_actual_text,
            transform=ax.transAxes,
            fontsize=12,
            verticalalignment="top",
            fontfamily="monospace",
            color="black",
            fontweight="bold",
        )


# ==========================================
# PART 6: Main Execution
# ==========================================


def monte_carlo_race_strategy(
    year: int, event_name: str, session_name: str, race, post: bool
) -> dict:
    # Matched Context Setup
    fastf1.plotting.setup_mpl(
        mpl_timedelta_support=False, color_scheme=None, misc_mpl_mods=False
    )

    load_race_data(race)

    # 1. Data & Physics
    winner_abbr = get_winner_abbr(race)
    if not winner_abbr:
        return {"filename": None, "caption": "No winner found.", "post": False}

    stint_stats = calculate_actual_stint_stats(race)
    deg_curves, deg_source, fuel_k = calculate_race_degradation_curves(
        race, stint_stats
    )

    actual_strategy_str = get_driver_actual_strategy(race, winner_abbr)
    actual_used_counts = get_actual_tyre_counts(race, winner_abbr)

    total_laps_real = race.total_laps or race.laps["LapNumber"].max()
    pit_loss_real = calculate_avg_pit_loss(race)

    # 2. Strategy Generation
    all_possible_strategies = generate_all_possible_strategies(
        total_laps_real, stint_stats
    )

    # 3. Filter
    valid_strategies = [
        s
        for s in all_possible_strategies
        if validate_strategy_with_inventory(s["compounds"], actual_used_counts)
    ]

    # 4. Simulation
    N_SIMS = 1000
    simulator = RaceStrategySimulator(
        deg_curves=deg_curves,
        total_laps=total_laps_real,
        pit_loss=pit_loss_real,
        lap_time_std=0.4,
        fuel_correction_factor=fuel_k,
    )

    all_results = {}
    for strat in valid_strategies:
        all_results[strat["name"]] = simulator.simulate_strategy(
            strat["name"], strat["compounds"], strat["pit_laps"], N_SIMS
        )

    # 5. Selection
    ranking_metric = {
        name: np.percentile(data, 70) for name, data in all_results.items()
    }
    sorted_strategies = sorted(ranking_metric, key=ranking_metric.get)
    top_5_results = {name: all_results[name] for name in sorted_strategies[:5]}
    best_strategy_name = sorted_strategies[0]

    # 6. Plotting with Matched Style
    with plt.style.context(["science", "bright"]):
        plt.rcParams["figure.dpi"] = DPI
        plt.rcParams["savefig.dpi"] = DPI
        plt.rcParams["figure.autolayout"] = False
        plt.rcParams["figure.constrained_layout.use"] = False
        plt.rcParams["savefig.bbox"] = None

        fig, ax = plt.subplots(figsize=FIG_SIZE, dpi=DPI)
        fig.patch.set_facecolor("white")
        ax.set_facecolor("white")

        plot_strategy_distribution_styled(
            ax, top_5_results, total_laps_real, actual_strategy_str, winner_abbr
        )

        # Titles Matched to Code 2 Format
        suptitle_text_global = (
            f"{year} {event_name} Grand Prix: {winner_abbr} Race Strategy Analysis"
        )
        subtitle_upper = f"Monte Carlo Simulations Top 5 Options vs Actual Choice"

        plt.suptitle(suptitle_text_global, fontsize=18, color="black")
        plt.figtext(0.5, 0.94, subtitle_upper, ha="center", fontsize=15, color="black")

        filename = save_plot_and_get_filename(fig, suptitle_text_global, DPI)
        plt.close(fig)

    caption = create_styled_caption_monte_carlo(
        year,
        event_name,
        N_SIMS,
        best_strategy_name,
        deg_source,
        winner_abbr,
        fuel_k,
        pit_loss_real,
    )
    return {"filename": filename, "caption": caption, "post": post}
