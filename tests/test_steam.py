import subprocess
import unittest
from contextlib import redirect_stdout
from io import StringIO
from unittest.mock import patch

from csdemo_mapextractor.steam import main, public_build


class SteamTests(unittest.TestCase):
    def test_reads_public_build_from_steamcmd_output(self) -> None:
        output = '''
"config"
{
    "public"
    {
        "buildid" "wrong"
        "timeupdated" "wrong"
    }
}
"branches"
{
    "public"
    {
        "description" "Public branch"
        "pwdrequired" "0"
        "timeupdated" "1789000000"
        "buildid" "25218825"
    }
}
'''
        completed = subprocess.CompletedProcess(["steamcmd"], 0, stdout=output)
        with patch(
            "csdemo_mapextractor.steam.subprocess.run", return_value=completed
        ) as run:
            self.assertEqual(
                ("25218825", "1789000000"), public_build("steamcmd")
            )

        self.assertIn("+app_info_update", run.call_args.args[0])
        self.assertTrue(run.call_args.kwargs["check"])

    def test_standalone_key_output(self) -> None:
        output = StringIO()
        with (
            patch(
                "csdemo_mapextractor.steam.public_build",
                return_value=("25218825", "1789000000"),
            ),
            redirect_stdout(output),
        ):
            self.assertEqual(0, main(["--steamcmd", "steamcmd", "--key"]))

        self.assertEqual("25218825\n", output.getvalue())


if __name__ == "__main__":
    unittest.main()
