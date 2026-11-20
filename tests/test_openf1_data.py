import tempfile
import unittest
from copy import deepcopy
from unittest.mock import Mock, patch

import numpy as np
import pandas as pd

from openf1_data import ArchiveError, HistoricalSession, OpenF1Archive, quick_laps


BASE = pd.Timestamp("2024-05-01T12:00:00Z")


def timestamp(offset):
    return (BASE + pd.Timedelta(offset, unit="s")).isoformat()


def timed_lap(driver, number, start, seconds, out=False):
    return {
        "driver_number": driver,
        "lap_number": number,
        "date_start": timestamp(start),
        "lap_duration": seconds,
        "duration_sector_1": 1,
        "duration_sector_2": 1,
        "duration_sector_3": seconds - 2 if seconds else None,
        "is_pit_out_lap": out,
    }


class SessionTests(unittest.TestCase):
    def setUp(self):
        self.feeds = {
            "drivers": [
                {
                    "driver_number": n,
                    "name_acronym": abbr,
                    "team_name": "Ferrari",
                    "team_colour": "FF0000",
                }
                for n, abbr in [(16, "LEC"), (55, "SAI")]
            ],
            "session_result": [
                {"driver_number": 55, "position": 2, "duration": [9, 10, 11]},
                {"driver_number": 16, "position": 1, "duration": [9, 11, 10]},
            ],
            "laps": [
                timed_lap(16, 1, 0, 9),
                timed_lap(16, 2, 10, 10),
                timed_lap(55, 1, 0, None),
                timed_lap(55, 2, 10, 11, True),
            ],
            "stints": [
                {
                    "driver_number": 16,
                    "stint_number": 1,
                    "lap_start": 1,
                    "lap_end": 2,
                    "tyre_age_at_start": 3,
                    "compound": "SOFT",
                }
            ],
            "race_control": [],
            "car_data": [
                {
                    "date": timestamp(s),
                    "speed": 36,
                    "throttle": 60,
                    "brake": 100 if s >= 11 else 0,
                }
                for s in range(0, 25)
            ],
            "location": [
                {"date": timestamp(s + 0.1), "x": s * 100, "y": 20}
                for s in range(0, 25)
            ],
        }
        archive = Mock()
        archive.read.side_effect = lambda endpoint, params, **kw: deepcopy(
            self.feeds[endpoint]
        )
        self.session = HistoricalSession(
            archive,
            {"meeting_name": "Test GP"},
            {
                "session_key": 10,
                "date_start": timestamp(0),
                "date_end": timestamp(500),
                "year": 2024,
            },
        )

    def test_result_order_and_timed_laps(self):
        s = self.session
        self.assertEqual(list(s.results.Abbreviation), ["LEC", "SAI"])
        self.assertEqual(s.fastest_lap(16)["LapNumber"], 1)
        self.assertEqual(s.fastest_lap("LEC", qualifying=True)["LapNumber"], 2)
        self.assertEqual(list(s.driver_laps(["16", "SAI"]).LapNumber), [1, 2, 1, 2])
        self.assertEqual(len(quick_laps(s.driver_laps(16))), 1)
        self.assertEqual(s.team_color("Ferrari"), "#FF0000")
        self.assertEqual(list(s.driver_laps(16).TyreLife), [4, 5])
        self.assertEqual(list(s.driver_laps(16).StintLapNumber), [1, 2])
        self.assertEqual(list(s.driver_laps(55).Compound), ["UNKNOWN", "UNKNOWN"])
        self.assertTrue(s.driver_laps(55).iloc[0].PitIn)

    def test_deleted_lap_and_subsequent_reinstatement(self):
        decision = {
            "date": timestamp(11),
            "message": "CAR 16 (LEC) TIME 0:09.000 DELETED - LAP 1",
        }
        self.feeds["race_control"] = [decision]
        self.assertEqual(self.session.fastest_lap(16)["LapNumber"], 2)
        self.assertEqual(list(quick_laps(self.session.driver_laps(16)).LapNumber), [2])
        self.feeds["race_control"].append(
            {"date": timestamp(20), "message": "CAR 16 (LEC) LAP 1 REINSTATED"}
        )
        restored = HistoricalSession(
            self.session.archive, self.session.meeting, self.session.info
        )
        self.assertEqual(restored.fastest_lap(16)["LapNumber"], 1)

    def test_optional_corner_data_uses_metres(self):
        self.session.meeting.update(
            circuit_key=9,
            circuit_info_url="https://api.multiviewer.app/api/v1/circuits/9/2024",
        )
        self.feeds[self.session.meeting["circuit_info_url"]] = {
            "rotation": 23,
            "corners": [
                {
                    "number": 1,
                    "angle": 15,
                    "length": 1000,
                    "trackPosition": {"x": 250, "y": -120},
                }
            ],
        }
        corner = self.session.circuit.corners.iloc[0]
        self.assertEqual((corner.Distance, corner.X, corner.Y), (100, 25, -12))

    def test_missing_final_qualifying_time_never_uses_an_earlier_phase(self):
        self.feeds["session_result"][1]["duration"] = [9, 11, None]
        with self.assertRaisesRegex(ArchiveError, "final qualifying"):
            self.session.fastest_lap("LEC", qualifying=True)

    def test_excludes_cautions_overlapping_either_end_of_a_lap(self):
        self.feeds["race_control"] = [
            {"date": timestamp(8), "scope": "Sector", "sector": 2, "flag": "YELLOW"},
            {"date": timestamp(10), "scope": "Sector", "sector": 2, "flag": "CLEAR"},
        ]
        self.assertEqual(list(self.session.green_laps.LapNumber), [2, 2])

    def test_distance_brake_position_and_delta(self):
        s = self.session
        first, second = (s.fastest_lap(d, qualifying=True) for d in ("LEC", "SAI"))
        car = s.telemetry(first)
        np.testing.assert_allclose(car.Distance, car.Time.dt.total_seconds() * 10)
        self.assertEqual(set(car.Brake), {0, 1})
        self.assertAlmostEqual(car.Y.iloc[0], 2)
        self.assertAlmostEqual(car.X.iloc[1], 109)
        delta, ref, _ = s.delta_time(first, second)
        self.assertAlmostEqual(delta.iloc[0], 0)
        self.assertAlmostEqual(delta.iloc[-1], 1)
        self.assertEqual(ref.Distance.iloc[-1], 100)
        car_calls = [
            c for c in s.archive.read.call_args_list if c.args[0] == "car_data"
        ]
        self.assertEqual(len(car_calls), 2)

    def test_rejects_gaps_instead_of_filling_a_missing_lap(self):
        self.feeds["car_data"] = [
            r
            for r in self.feeds["car_data"]
            if not timestamp(12) <= r["date"] <= timestamp(18)
        ]
        with self.assertRaisesRegex(ArchiveError, "Gap"):
            self.session.car_data(self.session.fastest_lap("LEC", qualifying=True))

    def test_missing_driver_or_times(self):
        with self.assertRaisesRegex(ArchiveError, "timed lap"):
            self.session.fastest_lap("MISSING")
        self.assertTrue(pd.isna(self.session.driver_laps(55).iloc[0].LapTime))


class ArchiveTests(unittest.TestCase):
    def setUp(self):
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        self.archive = OpenF1Archive(directory.name)
        self.addCleanup(self.archive.http.close)

    def test_only_completed_supported_sessions_and_meetings(self):
        sessions = [
            {
                "meeting_key": 1,
                "session_key": 3,
                "session_name": "Sprint Shootout",
                "date_end": timestamp(0),
            },
            {
                "meeting_key": 1,
                "session_key": 4,
                "session_name": "Race",
                "date_end": "2099-01-01T12:00:00Z",
            },
            {
                "meeting_key": 1,
                "session_name": "Practice 1",
                "date_end": timestamp(0),
                "is_cancelled": True,
            },
            {"meeting_key": 1, "session_name": "Qualifying", "date_end": None},
            {"meeting_key": 2, "session_name": "Race", "date_end": timestamp(0)},
        ]
        meetings = [
            {"meeting_key": 1, "meeting_name": "One"},
            {"meeting_key": 2, "meeting_name": "Two", "is_cancelled": True},
        ]
        with patch.object(
            self.archive,
            "read",
            side_effect=lambda ep, *a, **k: meetings if ep == "meetings" else sessions,
        ):
            calendar = self.archive.calendar(2024)
            self.assertEqual(list(calendar), ["One"])
            self.assertEqual(list(calendar["One"][1]), ["SQ"])
            self.assertEqual(self.archive.calendar(2022), {})
            with self.assertRaisesRegex(ArchiveError, "No completed"):
                self.archive.session(2024, "One", "R")

    @patch("openf1_data.time.sleep")
    def test_rate_limit_then_disk_cache(self, sleep):
        limited = Mock(status_code=429, headers={"Retry-After": "4"})
        ok = Mock(status_code=200)
        ok.json.return_value = [{"lap_number": 1}]
        with patch.object(self.archive.http, "get", side_effect=[limited, ok]) as get:
            for _ in range(2):
                self.assertEqual(
                    self.archive.read("laps", {"session_key": 1}), ok.json.return_value
                )
            self.assertEqual(get.call_count, 2)
            self.assertGreater(sleep.call_args.args[0], 3)

    @patch("openf1_data.time.sleep")
    def test_archive_denial_and_empty_results_are_not_cached(self, sleep):
        denied, empty = Mock(status_code=401), Mock(status_code=200)
        empty.json.return_value = []
        with patch.object(self.archive.http, "get", side_effect=[denied, empty]) as get:
            with self.assertRaisesRegex(ArchiveError, "historical access"):
                self.archive.read("laps", {"session_key": 5})
            self.assertEqual(self.archive.read("laps", {"session_key": 5}), [])
            self.assertEqual(get.call_count, 2)
        self.assertFalse(list(self.archive.cache.iterdir()))
