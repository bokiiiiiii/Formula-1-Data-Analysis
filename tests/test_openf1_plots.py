"""Opt-in end-to-end rendering of every configured plot from public historical sessions."""

import os
import tempfile
import unittest
from pathlib import Path

from PIL import Image

from config import Config
from plot_functions.plot_runner import PlotRunner


@unittest.skipUnless(
    os.environ.get("OPENF1_TEST_CACHE"),
    "set OPENF1_TEST_CACHE to run archive rendering",
)
class HistoricalPlotTests(unittest.TestCase):
    def test_all_chart_types(self):
        verified = set()
        with tempfile.TemporaryDirectory() as output:
            config = Config(
                folder_path=output, cache_path=os.environ["OPENF1_TEST_CACHE"]
            )
            for year, event, code in [
                (2026, "Mexico City Grand Prix", "R"),
                (2026, "Mexico City Grand Prix", "FP1"),
                (2026, "Mexico City Grand Prix", "Q"),
                (2025, "United States Grand Prix", "SQ"),
            ]:
                with self.subTest(year=year, event=event, code=code):
                    runner = PlotRunner(config)
                    try:
                        results = runner.run_all(year, event, code)
                    finally:
                        runner.cleanup()
                    expected = {
                        name
                        for name, plot in config.plot_functions.items()
                        if plot.session == code
                    }
                    self.assertEqual(set(results), expected)
                    for name, item in results.items():
                        with self.subTest(chart=name):
                            self.assertTrue(item.get("success"), item.get("caption"))
                            file = Path(item["filename"])
                            self.assertEqual(file.parent, Path(output))
                            with Image.open(file) as image:
                                self.assertEqual(image.format, "PNG")
                                self.assertGreater(min(image.size), 500)
                                self.assertTrue(
                                    any(
                                        lo != hi
                                        for lo, hi in image.convert("RGB").getextrema()
                                    )
                                )
                            verified.add(name)
            self.assertEqual(verified, set(config.plot_functions))
