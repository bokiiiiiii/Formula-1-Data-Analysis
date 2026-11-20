"""Anonymous access to completed OpenF1 sessions and their plotting data."""

import gzip
import hashlib
import json
import logging
import os
import re
from dataclasses import dataclass
from functools import cached_property
from pathlib import Path
from tempfile import NamedTemporaryFile
import time

import numpy as np
import pandas as pd
import requests
from scipy.integrate import cumulative_trapezoid

logger = logging.getLogger(__name__)
API = "https://api.openf1.org/v1"
SESSION_CODES = {
    "Practice 1": "FP1",
    "Practice 2": "FP2",
    "Practice 3": "FP3",
    "Qualifying": "Q",
    "Sprint Qualifying": "SQ",
    "Sprint Shootout": "SQ",
    "Sprint": "S",
    "Race": "R",
}


class ArchiveError(RuntimeError):
    """The archive cannot supply the requested data."""


class MissingTelemetry(ArchiveError):
    """A recorded lap has too few samples to plot reliably."""


class OpenF1Archive:
    def __init__(self, cache_path):
        self.cache = Path(cache_path) / "openf1"
        self.cache.mkdir(parents=True, exist_ok=True)
        self.http = requests.Session()
        self.next_request = 0.0

    def read(self, endpoint, params, *, max_age=None, circuit=False):
        url = endpoint if circuit else f"{API}/{endpoint}"
        key = hashlib.sha256(
            json.dumps([url, params], sort_keys=True).encode()
        ).hexdigest()
        path = self.cache / f"{key}.json.gz"
        if path.exists() and (
            max_age is None or time.time() - path.stat().st_mtime < max_age
        ):
            try:
                with gzip.open(path, "rt", encoding="utf-8") as src:
                    return json.load(src)
            except (ValueError, EOFError, OSError):
                logger.warning("Refreshing unreadable archive cache %s", path.name)

        for attempt in range(4):
            time.sleep(max(0, self.next_request - time.monotonic()))
            self.next_request = time.monotonic() + 2.1
            try:
                response = self.http.get(url, params=params, timeout=(10, 45))
            except (requests.Timeout, requests.ConnectionError) as exc:
                if attempt == 3:
                    raise ArchiveError(f"Could not reach {url}: {exc}") from exc
            else:
                if response.status_code == 200:
                    break
                if response.status_code == 401:
                    raise ArchiveError(
                        "OpenF1 has not released this data for anonymous historical access."
                    )
                if (
                    response.status_code not in (429, 500, 502, 503, 504)
                    or attempt == 3
                ):
                    raise ArchiveError(
                        f"{url} returned HTTP {response.status_code}: {response.text[:180]}"
                    )
                delay = response.headers.get("Retry-After", "")
                if delay.isdigit():
                    self.next_request = max(
                        self.next_request, time.monotonic() + int(delay)
                    )
            self.next_request = max(self.next_request, time.monotonic() + 2**attempt)

        data = response.json()
        if not isinstance(data, dict if circuit else list):
            raise ArchiveError(f"Unexpected response from {url}")
        if data:
            with NamedTemporaryFile(dir=self.cache, delete=False) as tmp:
                tmp_path = Path(tmp.name)
            try:
                with gzip.open(
                    tmp_path, "wt", encoding="utf-8", compresslevel=1
                ) as handle:
                    json.dump(data, handle, separators=(",", ":"))
                os.replace(tmp_path, path)
            finally:
                tmp_path.unlink(missing_ok=True)
        return data

    def calendar(self, year):
        now = pd.Timestamp.now(tz="UTC")
        if not 2023 <= year <= now.year:
            return {}
        ttl = 300 if year == now.year else None
        meetings = self.read("meetings", {"year": year}, max_age=ttl)
        sessions = self.read("sessions", {"year": year}, max_age=ttl)
        completed = {}
        for row in sessions:
            code = SESSION_CODES.get(row.get("session_name"))
            end = pd.to_datetime(row.get("date_end"), utc=True, errors="coerce")
            if code and not row.get("is_cancelled") and pd.notna(end) and end < now:
                completed.setdefault(row["meeting_key"], {})[code] = row
        return {
            m["meeting_name"]: (m, completed[m["meeting_key"]])
            for m in meetings
            if not m.get("is_cancelled") and m["meeting_key"] in completed
        }

    def session(self, year, event, code):
        try:
            meeting, sessions = self.calendar(year)[event]
            session = sessions[code]
        except KeyError:
            raise ArchiveError(
                f"No completed {code} session for {year} {event}."
            ) from None
        return HistoricalSession(self, meeting, session)


def quick_laps(laps, threshold=1.07):
    laps = laps[~laps["Deleted"]]
    times = laps["LapTime"]
    return laps[times < times.min() * threshold].copy()


@dataclass
class Circuit:
    rotation: float
    corners: pd.DataFrame


class HistoricalSession:
    def __init__(self, archive, meeting, session):
        self.archive = archive
        self.meeting = meeting
        self.info = session
        self.key = session["session_key"]
        self.event_name = meeting["meeting_name"]
        self._windows = {}
        age = pd.Timestamp.now(tz="UTC") - pd.Timestamp(session["date_end"])
        self.max_age = 300 if age < pd.Timedelta(2, unit="D") else None

    def _rows(self, endpoint, **filters):
        return self.archive.read(
            endpoint, {"session_key": self.key, **filters}, max_age=self.max_age
        )

    @cached_property
    def roster(self):
        rows = self._rows("drivers")
        df = pd.DataFrame(
            rows, columns=["driver_number", "name_acronym", "team_name", "team_colour"]
        )
        df.columns = ["DriverNumber", "Abbreviation", "Team", "TeamColor"]
        df = df.dropna(subset=["DriverNumber"]).drop_duplicates(
            "DriverNumber", keep="last"
        )
        df["Abbreviation"] = df["Abbreviation"].fillna(df["DriverNumber"].astype(str))
        return df.set_index("DriverNumber", drop=False)

    @cached_property
    def results(self):
        df = pd.DataFrame(
            self._rows("session_result"),
            columns=["driver_number", "position", "duration", "dsq", "dns"],
        )
        df = df.dropna(subset=["position"]).sort_values("position")
        df = df[~(df["dsq"].eq(True) | df["dns"].eq(True))]
        df = df.rename(
            columns={"position": "Position", "driver_number": "DriverNumber"}
        )
        return df.merge(
            self.roster.reset_index(drop=True), on="DriverNumber"
        ).reset_index(drop=True)

    @cached_property
    def laps(self):
        columns = [
            "driver_number",
            "lap_number",
            "date_start",
            "lap_duration",
            "is_pit_out_lap",
            "duration_sector_1",
            "duration_sector_2",
            "duration_sector_3",
        ]
        laps = (
            pd.DataFrame(self._rows("laps"), columns=columns)
            .rename(
                columns={
                    "driver_number": "DriverNumber",
                    "lap_number": "LapNumber",
                    "date_start": "Start",
                    "is_pit_out_lap": "PitOut",
                }
            )
            .dropna(subset=["DriverNumber", "LapNumber"])
        )
        laps = laps.drop_duplicates(
            ["DriverNumber", "LapNumber"], keep="last"
        ).sort_values(["DriverNumber", "LapNumber"])
        laps["Start"] = pd.to_datetime(
            laps["Start"], format="ISO8601", errors="coerce", utc=True
        )
        seconds = pd.to_numeric(laps.pop("lap_duration"), errors="coerce")
        laps["LapTime"] = pd.to_timedelta(seconds.where(seconds > 0), unit="s")
        sectors = laps[[f"duration_sector_{i}" for i in range(1, 4)]].apply(
            pd.to_numeric, errors="coerce"
        )
        laps["CompleteSectors"] = (
            seconds.sub(sectors.sum(axis=1, min_count=3)).abs() <= 0.01
        )
        laps = laps.join(
            self.roster[["Abbreviation", "Team"]], on="DriverNumber"
        ).rename(columns={"Abbreviation": "Driver"})
        laps["Stint"] = np.nan
        laps["Compound"] = "UNKNOWN"
        laps["TyreLife"] = np.nan
        laps["StintLapNumber"] = np.nan
        for stint in self._rows("stints"):
            first, last = stint.get("lap_start"), stint.get("lap_end")
            if first is None or last is None:
                continue
            mask = laps["DriverNumber"].eq(stint["driver_number"]) & laps[
                "LapNumber"
            ].between(first, last)
            laps.loc[mask, "Stint"] = stint["stint_number"]
            laps.loc[mask, "Compound"] = stint.get("compound") or "UNKNOWN"
            laps.loc[mask, "StintLapNumber"] = laps.loc[mask, "LapNumber"] - first + 1
            if stint.get("tyre_age_at_start") is not None:
                laps.loc[mask, "TyreLife"] = (
                    laps.loc[mask, "LapNumber"] - first + 1 + stint["tyre_age_at_start"]
                )
        # A timed lap immediately before a pit-out lap is the in-lap.
        next_out = laps.groupby("DriverNumber")["PitOut"].shift(-1).eq(True)
        consecutive = (
            laps.groupby("DriverNumber")["LapNumber"]
            .shift(-1)
            .eq(laps["LapNumber"] + 1)
        )
        laps["PitIn"] = next_out & consecutive
        laps["Deleted"] = False
        for row in self.control_messages:
            message = row.get("message") or ""
            if "DELETED" not in message and "REINSTATED" not in message:
                continue
            car = re.search(r"\bCAR (\d+)\b", message)
            number = re.search(r"\bLAP (\d+)\b", message)
            duration = re.search(r"\bTIME (\d+):(\d+\.\d+)\b", message)
            if not car or not (number or duration):
                continue
            match = laps["DriverNumber"].eq(int(car[1]))
            if number:
                match &= laps["LapNumber"].eq(int(number[1]))
            if duration:
                seconds = int(duration[1]) * 60 + float(duration[2])
                match &= (laps["LapTime"].dt.total_seconds() - seconds).abs() < 0.00051
            laps.loc[match, "Deleted"] = "REINSTATED" not in message
        return laps.reset_index(drop=True)

    @cached_property
    def control_messages(self):
        start = pd.Timestamp(self.info["date_start"])
        # Some archives include messages carried over from an earlier session.
        return sorted(
            (
                row
                for row in self._rows("race_control")
                if pd.Timestamp(row["date"]) >= start
            ),
            key=lambda row: row["date"],
        )

    def driver_laps(self, driver):
        drivers = [driver] if np.isscalar(driver) else driver
        values = {str(value) for value in drivers}
        return self.laps[
            self.laps["Driver"].isin(values)
            | self.laps["DriverNumber"].astype(str).isin(values)
        ].copy()

    def fastest_lap(self, driver=None, *, qualifying=False):
        laps = self.laps if driver is None else self.driver_laps(driver)
        laps = laps.dropna(subset=["LapTime", "Start"])
        laps = laps[~laps["Deleted"]]
        if qualifying:
            results = self.results.loc[
                self.results["Abbreviation"] == driver, "duration"
            ]
            times = results.iloc[0] if len(results) else None
            if not isinstance(times, list) or len(times) < 3 or not times[2]:
                raise ArchiveError(f"No classified final qualifying lap for {driver}.")
            # A driver may have set a faster lap in an earlier qualifying phase.
            laps = laps[
                (laps["LapTime"].dt.total_seconds() - times[2]).abs() < 0.00051
            ].tail(1)
        if laps.empty:
            raise ArchiveError(f"No matching timed lap for {driver or 'this session'}.")
        return laps.loc[laps["LapTime"].idxmin()]

    def team_color(self, team):
        colors = self.roster.loc[self.roster["Team"] == team, "TeamColor"].dropna()
        return "#" + str(colors.iloc[0]).lstrip("#") if len(colors) else "#808080"

    @cached_property
    def total_laps(self):
        return int(self.laps["LapNumber"].max()) if not self.laps.empty else 0

    @cached_property
    def green_laps(self):
        """Timed laps with complete sectors that do not overlap a reported caution."""
        laps = self.laps[
            self.laps["CompleteSectors"]
            & ~self.laps["Deleted"]
            & self.laps["Start"].notna()
        ].copy()
        controls = self.control_messages
        if not controls or laps.empty:
            return laps.iloc[:0]
        start = pd.Timestamp(self.info["date_start"])
        end = pd.Timestamp(self.info["date_end"])
        active = {}
        intervals = []
        for row in controls:
            at = pd.Timestamp(row["date"])
            if at < start or at > end:
                continue
            flag = row.get("flag")
            message = (row.get("message") or "").upper()
            scope = row.get("sector") if row.get("scope") == "Sector" else "track"
            if row.get("scope") == "Driver":
                continue
            if (
                flag in ("YELLOW", "DOUBLE YELLOW", "RED")
                or "SAFETY CAR DEPLOYED" in message
            ):
                active.setdefault(scope, at)
            elif flag in ("GREEN", "CLEAR") or message == "SESSION RESUMED":
                scopes = list(active) if scope == "track" else [scope]
                for cleared in scopes:
                    if cleared in active:
                        intervals.append((active.pop(cleared), at))
        intervals.extend((begin, end) for begin in active.values())
        finish = laps["Start"] + laps["LapTime"]
        keep = pd.Series(True, index=laps.index)
        for begin, clear in intervals:
            keep &= ~((laps["Start"] < clear) & (finish > begin))
        return laps[keep]

    @cached_property
    def circuit(self):
        key = self.meeting.get("circuit_key")
        if key is None:
            return None
        url = f"https://api.multiviewer.app/api/v1/circuits/{int(key)}/{int(self.info['year'])}"
        if self.meeting.get("circuit_info_url") != url:
            return None
        try:
            data = self.archive.read(url, {}, circuit=True)
            corners = pd.DataFrame(
                [
                    {
                        "Number": c["number"],
                        "Letter": c.get("letter") or "",
                        "Angle": c["angle"],
                        "X": c["trackPosition"]["x"] / 10,
                        "Y": c["trackPosition"]["y"] / 10,
                        "Distance": c["length"] / 10,
                    }
                    for c in data["corners"]
                ]
            )
            return Circuit(data["rotation"], corners) if not corners.empty else None
        except (ArchiveError, KeyError, ValueError, TypeError) as exc:
            logger.warning(
                "Corner metadata unavailable for %s: %s", self.event_name, exc
            )
            return None

    def _samples(self, endpoint, driver, start, end):
        # Fetch bounded blocks once per driver. Race-wide plots revisit the same blocks on successive laps.
        block_size = pd.Timedelta(15, unit="min")
        blocks = []
        for block in pd.date_range(
            start.floor("15min"), end.floor("15min"), freq=block_size
        ):
            key = (endpoint, int(driver), block)
            if key not in self._windows:
                rows = self._rows(
                    endpoint,
                    driver_number=int(driver),
                    **{
                        "date>": block.isoformat(),
                        "date<": (block + block_size).isoformat(),
                    },
                )
                frame = pd.DataFrame(rows)
                if not frame.empty:
                    frame.index = pd.to_datetime(
                        frame.pop("date"), format="ISO8601", utc=True
                    )
                    frame = frame[~frame.index.duplicated()].sort_index()
                self._windows[key] = frame
            blocks.append(self._windows[key])
        frame = pd.concat(blocks)
        if frame.empty:
            return frame
        return frame.loc[start:end].loc[lambda d: ~d.index.duplicated()]

    def _lap_samples(self, endpoint, lap, channels):
        start, duration = lap["Start"], lap["LapTime"]
        if pd.isna(start) or pd.isna(duration):
            raise MissingTelemetry("Missing lap timing.")
        end = start + duration
        margin = pd.Timedelta(2, unit="s")
        samples = self._samples(
            endpoint, lap["DriverNumber"], start - margin, end + margin
        )
        if samples.empty:
            raise MissingTelemetry(
                f"No {endpoint} for driver {lap['DriverNumber']}, lap {lap['LapNumber']}."
            )
        samples = samples.reindex(columns=channels).dropna().rename(columns=channels)
        if (
            len(samples) < 2
            or samples.index[0] > start + margin
            or samples.index[-1] < end - margin
        ):
            raise MissingTelemetry(f"Incomplete {endpoint} at the lap boundaries.")
        seconds = (samples.index - start).total_seconds().to_numpy()
        axis = np.r_[
            0.0,
            seconds[(seconds > 0) & (seconds < duration.total_seconds())],
            duration.total_seconds(),
        ]
        if np.diff(axis).max() > 3:
            raise MissingTelemetry(f"Gap in {endpoint} exceeds three seconds.")
        return pd.DataFrame(
            {
                "Time": pd.to_timedelta(axis, unit="s"),
                **{
                    name: np.interp(axis, seconds, samples[name])
                    for name in channels.values()
                },
            }
        )

    def car_data(self, lap):
        frame = self._lap_samples(
            "car_data",
            lap,
            {"speed": "Speed", "brake": "Brake", "throttle": "Throttle"},
        )
        frame["Brake"] = (frame["Brake"] >= 50).astype(int)
        frame["Distance"] = cumulative_trapezoid(
            frame["Speed"] / 3.6, frame["Time"].dt.total_seconds(), initial=0
        )
        return frame

    def position_data(self, lap):
        frame = self._lap_samples("location", lap, {"x": "X", "y": "Y"})
        frame[["X", "Y"]] /= 10  # Feed coordinates are decimetres.
        return frame

    def telemetry(self, lap):
        car = self.car_data(lap)
        pos = self.position_data(lap)
        for name in ("X", "Y"):
            car[name] = np.interp(
                car["Time"].dt.total_seconds(),
                pos["Time"].dt.total_seconds(),
                pos[name],
            )
        return car

    def delta_time(self, first, second):
        reference, other = self.car_data(first), self.car_data(second)

        # Normalize integrated lap lengths so both finish lines align.
        def progress(frame):
            length = frame["Distance"].iloc[-1]
            if length <= 0:
                raise ArchiveError("No distance covered in telemetry.")
            return frame["Distance"] / length

        other_time = np.interp(
            progress(reference), progress(other), other["Time"].dt.total_seconds()
        )
        return other_time - reference["Time"].dt.total_seconds(), reference, other
