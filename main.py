from matplotlib import pyplot as plt

import fastf1
import fastf1.plotting

from driver_laptimes_distribution import driver_laptimes_distribution
from team_pace_ranking import team_pace_ranking


Year: int = 2024
EventName: str = "Monaco"
SessionName: str = "R"

# @brief main: Plot F1 data analysis results
# @ref: https://github.com/theOehrly/Fast-F1
if __name__ == "__main__":

    fastf1.plotting.setup_mpl(mpl_timedelta_support=False, misc_mpl_mods=False)
    race = fastf1.get_session(Year, EventName, SessionName)

    plt.ion()

    driver_laptimes_distribution(Year, EventName, SessionName, race)
    team_pace_ranking(Year, EventName, SessionName, race)

    plt.ioff()
    plt.show(block=True)
