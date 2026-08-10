from __future__ import annotations

import contextlib
import io
import unittest

from longitudinal_dbt.cli import main


class CliTests(unittest.TestCase):
    def test_top_level_help(self) -> None:
        output = io.StringIO()
        with contextlib.redirect_stdout(output), self.assertRaises(SystemExit) as result:
            main(["--help"])
        self.assertEqual(result.exception.code, 0)
        self.assertIn("convert-dicom", output.getvalue())
        self.assertIn("build-graphs", output.getvalue())

    def test_version(self) -> None:
        output = io.StringIO()
        with contextlib.redirect_stdout(output), self.assertRaises(SystemExit) as result:
            main(["--version"])
        self.assertEqual(result.exception.code, 0)
        self.assertIn("longitudinal-dbt 0.1.0", output.getvalue())

    def test_unknown_command_fails(self) -> None:
        with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit) as result:
            main(["not-a-command"])
        self.assertEqual(result.exception.code, 2)


if __name__ == "__main__":
    unittest.main()
