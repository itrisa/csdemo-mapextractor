"""Steam build metadata used to gate expensive CI extraction runs."""

from __future__ import annotations

import re
import shutil
import subprocess
import time


_PUBLIC_BRANCH = re.compile(r'"public"\s*\{')
_BUILD_ID = re.compile(r'"buildid"\s*"(\d+)"')
_UPDATED_AT = re.compile(r'"timeupdated"\s*"(\d+)"')


def public_build(steamcmd: str | None = None) -> tuple[str, str]:
    executable = steamcmd or shutil.which("steamcmd")
    if executable is None:
        raise FileNotFoundError("steamcmd was not found; pass --steamcmd or add it to PATH")
    command = [
        executable,
        "+login",
        "anonymous",
        "+app_info_update",
        "1",
        "+app_info_print",
        "730",
        "+logoff",
        "+quit",
    ]
    last_error: Exception | None = None
    for attempt in range(3):
        try:
            completed = subprocess.run(
                command,
                check=True,
                capture_output=True,
                text=True,
                timeout=300,
            )
            branch = _find_public_branch(completed.stdout)
            build_id = _BUILD_ID.search(branch)
            updated_at = _UPDATED_AT.search(branch)
            if build_id is not None and updated_at is not None:
                return build_id.group(1), updated_at.group(1)
            last_error = RuntimeError(
                "could not find CS2 public build ID in SteamCMD output"
            )
        except (subprocess.SubprocessError, RuntimeError) as error:
            last_error = error
        if attempt < 2:
            time.sleep(2)
    assert last_error is not None
    raise last_error


def _find_public_branch(output: str) -> str:
    marker = _PUBLIC_BRANCH.search(output)
    if marker is None:
        raise RuntimeError("could not find the CS2 public branch in SteamCMD output")
    start = output.find("{", marker.start())
    depth = 0
    for position in range(start, len(output)):
        if output[position] == "{":
            depth += 1
        elif output[position] == "}":
            depth -= 1
            if depth == 0:
                return output[start + 1 : position]
    raise RuntimeError("CS2 public branch block is incomplete in SteamCMD output")
