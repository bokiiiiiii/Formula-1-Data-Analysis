"""Smart plot selector for session-based recommendations."""

from typing import List, Dict, Set, Optional
from logger_config import get_logger

logger = get_logger(__name__)


class SmartPlotSelector:
    """Intelligent plot selection based on session characteristics."""

    # Session-specific plot mappings
    SESSION_PLOTS = {
        "FP1": {
            "recommended": [
                "plot_track_with_annotated_corners",
                "driver_laptimes_scatterplot",
                "team_pace_ranking",
            ],
            "optional": [
                "driver_laptimes_distribution",
            ],
            "not_applicable": [
                "annotated_qualifying_flying_lap",
                "annotated_sprint_qualifying_flying_lap",
                "monte_carlo_race_strategy",
            ],
        },
        "FP2": {
            "recommended": [
                "driver_laptimes_scatterplot",
                "driver_laptimes_distribution",
                "team_pace_ranking",
            ],
            "optional": [
                "driver_fuel_corrected_laptimes_scatterplot",
            ],
            "not_applicable": [
                "annotated_qualifying_flying_lap",
                "annotated_sprint_qualifying_flying_lap",
                "monte_carlo_race_strategy",
            ],
        },
        "FP3": {
            "recommended": [
                "driver_laptimes_scatterplot",
                "driver_laptimes_distribution",
                "team_pace_ranking",
            ],
            "optional": [
                "driver_fuel_corrected_laptimes_scatterplot",
            ],
            "not_applicable": [
                "annotated_qualifying_flying_lap",
                "annotated_sprint_qualifying_flying_lap",
                "monte_carlo_race_strategy",
            ],
        },
        "Q": {
            "recommended": [
                "annotated_qualifying_flying_lap",
                "driver_laptimes_distribution",
                "team_pace_ranking",
            ],
            "optional": [
                "driver_laptimes_scatterplot",
                "race_fatest_lap_telemetry_data",
            ],
            "not_applicable": [
                "annotated_race_fatest_lap",
                "monte_carlo_race_strategy",
                "driver_race_evolution_heatmap",
                "driver_fuel_corrected_laptimes_gaussian_processes",
                "driver_fuel_corrected_laptimes_scatterplot",
            ],
        },
        "SS": {
            "recommended": [
                "annotated_sprint_qualifying_flying_lap",
                "driver_laptimes_distribution",
                "team_pace_ranking",
            ],
            "optional": [
                "driver_laptimes_scatterplot",
            ],
            "not_applicable": [
                "annotated_race_fatest_lap",
                "monte_carlo_race_strategy",
                "driver_race_evolution_heatmap",
                "driver_fuel_corrected_laptimes_gaussian_processes",
            ],
        },
        "S": {
            "recommended": [
                "driver_race_evolution_heatmap",
                "team_pace_ranking",
                "driver_laptimes_scatterplot",
            ],
            "optional": [
                "annotated_race_fatest_lap",
                "race_fatest_lap_telemetry_data",
                "driver_fuel_corrected_laptimes_scatterplot",
            ],
            "not_applicable": [
                "annotated_qualifying_flying_lap",
                "annotated_sprint_qualifying_flying_lap",
                "monte_carlo_race_strategy",
            ],
        },
        "R": {
            "recommended": [
                "annotated_race_fatest_lap",
                "driver_race_evolution_heatmap",
                "race_fatest_lap_telemetry_data",
                "monte_carlo_race_strategy",
                "team_pace_ranking",
            ],
            "optional": [
                "driver_fuel_corrected_laptimes_gaussian_processes",
                "driver_fuel_corrected_laptimes_scatterplot",
                "driver_laptimes_scatterplot",
                "driver_laptimes_distribution",
            ],
            "not_applicable": [
                "annotated_qualifying_flying_lap",
                "annotated_sprint_qualifying_flying_lap",
            ],
        },
    }

    # Plot descriptions
    PLOT_DESCRIPTIONS = {
        "plot_track_with_annotated_corners": "Track layout with corner numbers",
        "annotated_qualifying_flying_lap": "Fastest qualifying lap with sector times",
        "annotated_race_fatest_lap": "Fastest race lap with telemetry",
        "annotated_sprint_qualifying_flying_lap": "Fastest sprint shootout lap",
        "race_fatest_lap_telemetry_data": "Detailed telemetry comparison",
        "driver_fuel_corrected_laptimes_gaussian_processes": "Fuel-corrected lap times (GP model)",
        "driver_fuel_corrected_laptimes_scatterplot": "Fuel-corrected lap times scatter",
        "driver_laptimes_scatterplot": "Raw lap times scatter plot",
        "driver_laptimes_distribution": "Lap time distribution analysis",
        "driver_race_evolution_heatmap": "Race position evolution heatmap",
        "monte_carlo_race_strategy": "Strategy simulation with pit stops",
        "team_pace_ranking": "Team performance ranking",
    }

    def __init__(self):
        """Initialize smart plot selector."""
        logger.info("SmartPlotSelector initialized")

    def get_recommended_plots(
        self,
        session: str,
        include_optional: bool = False,
    ) -> List[str]:
        """Get recommended plots for a session.

        Args:
            session: Session identifier (FP1, FP2, FP3, Q, SS, S, R)
            include_optional: Include optional plots

        Returns:
            List of plot function names
        """
        session_upper = session.upper()

        if session_upper not in self.SESSION_PLOTS:
            logger.warning(f"Unknown session type: {session}, returning all plots")
            return list(self.PLOT_DESCRIPTIONS.keys())

        plots = self.SESSION_PLOTS[session_upper]["recommended"].copy()

        if include_optional:
            plots.extend(self.SESSION_PLOTS[session_upper]["optional"])

        return plots

    def get_all_applicable_plots(self, session: str) -> List[str]:
        """Get all applicable plots (recommended + optional).

        Args:
            session: Session identifier

        Returns:
            List of all applicable plot function names
        """
        return self.get_recommended_plots(session, include_optional=True)

    def is_plot_applicable(self, plot_name: str, session: str) -> bool:
        """Check if a plot is applicable for a session.

        Args:
            plot_name: Plot function name
            session: Session identifier

        Returns:
            True if plot is applicable
        """
        session_upper = session.upper()

        if session_upper not in self.SESSION_PLOTS:
            return True

        not_applicable = self.SESSION_PLOTS[session_upper]["not_applicable"]
        return plot_name not in not_applicable

    def filter_plots(
        self,
        plot_names: List[str],
        session: str,
        remove_not_applicable: bool = True,
    ) -> List[str]:
        """Filter plot list based on session type.

        Args:
            plot_names: List of plot function names
            session: Session identifier
            remove_not_applicable: Remove plots not applicable to session

        Returns:
            Filtered list of plot function names
        """
        if not remove_not_applicable:
            return plot_names

        return [plot for plot in plot_names if self.is_plot_applicable(plot, session)]

    def get_plot_description(self, plot_name: str) -> str:
        """Get description for a plot.

        Args:
            plot_name: Plot function name

        Returns:
            Description string
        """
        return self.PLOT_DESCRIPTIONS.get(plot_name, "Custom plot function")

    def suggest_plots(
        self,
        session: str,
        available_plots: Optional[List[str]] = None,
    ) -> Dict[str, List[Dict[str, str]]]:
        """Get plot suggestions with descriptions.

        Args:
            session: Session identifier
            available_plots: Optional list of available plots to filter

        Returns:
            Dictionary with categorized suggestions
        """
        session_upper = session.upper()

        if session_upper not in self.SESSION_PLOTS:
            logger.warning(f"Unknown session type: {session}")
            return {"recommended": [], "optional": [], "not_applicable": []}

        session_data = self.SESSION_PLOTS[session_upper]

        result = {
            "recommended": [],
            "optional": [],
            "not_applicable": [],
        }

        # Filter by available plots if provided
        for category in ["recommended", "optional", "not_applicable"]:
            for plot_name in session_data[category]:
                if available_plots is None or plot_name in available_plots:
                    result[category].append(
                        {
                            "name": plot_name,
                            "description": self.get_plot_description(plot_name),
                        }
                    )

        return result

    def print_suggestions(self, session: str) -> None:
        """Print plot suggestions for a session.

        Args:
            session: Session identifier
        """
        suggestions = self.suggest_plots(session)

        logger.info(f"\nPlot suggestions for {session}:")

        if suggestions["recommended"]:
            logger.info("\n✅ Recommended:")
            for plot in suggestions["recommended"]:
                logger.info(f"  • {plot['name']}: {plot['description']}")

        if suggestions["optional"]:
            logger.info("\n⭕ Optional:")
            for plot in suggestions["optional"]:
                logger.info(f"  • {plot['name']}: {plot['description']}")

        if suggestions["not_applicable"]:
            logger.info("\n❌ Not applicable:")
            for plot in suggestions["not_applicable"]:
                logger.info(f"  • {plot['name']}: {plot['description']}")

    def get_compatibility_matrix(self) -> Dict[str, Dict[str, str]]:
        """Get compatibility matrix for all plots and sessions.

        Returns:
            Dictionary mapping plot names to session compatibility
        """
        all_plots = set(self.PLOT_DESCRIPTIONS.keys())
        matrix = {}

        for plot in all_plots:
            matrix[plot] = {}
            for session in ["FP1", "FP2", "FP3", "Q", "SS", "S", "R"]:
                session_data = self.SESSION_PLOTS[session]

                if plot in session_data["recommended"]:
                    matrix[plot][session] = "recommended"
                elif plot in session_data["optional"]:
                    matrix[plot][session] = "optional"
                elif plot in session_data["not_applicable"]:
                    matrix[plot][session] = "not_applicable"
                else:
                    matrix[plot][session] = "applicable"

        return matrix


# Global selector instance
_global_selector: Optional[SmartPlotSelector] = None


def get_selector() -> SmartPlotSelector:
    """Get global plot selector instance.

    Returns:
        Global SmartPlotSelector instance
    """
    global _global_selector
    if _global_selector is None:
        _global_selector = SmartPlotSelector()
    return _global_selector


def suggest_plots_for_session(
    session: str, include_optional: bool = False
) -> List[str]:
    """Convenience function to get plot suggestions.

    Args:
        session: Session identifier
        include_optional: Include optional plots

    Returns:
        List of recommended plot names
    """
    return get_selector().get_recommended_plots(session, include_optional)
