"""Download and verify IDSSE v1 tracking inputs and export full-match CSVs."""

import argparse
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
from pathlib import Path
import subprocess
import sys
from tempfile import TemporaryDirectory
from urllib.request import urlopen

RELEASE = "https://api.figshare.com/v2/articles/28196177/versions/1"
LICENSE = "https://creativecommons.org/licenses/by/4.0/"
MATCHES = ("J03WMX", "J03WN1", "J03WOH", "J03WOY", "J03WPY", "J03WQQ", "J03WR9")


def verify(path, source):
    with path.open("rb") as stream:
        digest = hashlib.file_digest(stream, "md5").hexdigest()
    if path.stat().st_size != source["size"] or digest != source["computed_md5"]:
        raise ValueError(f"Publisher checksum or size mismatch for {path}")


def prepare(match, files, data, output):
    folder = data / match
    folder.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(dir=folder) as temporary:
        for name, kind in (("metadata.xml", "matchinformation"), ("tracking.xml", "positions_raw")):
            source, = [f for f in files if kind in f["name"] and f["name"].endswith(f"-{match}.xml")]
            path = folder / name
            if not path.exists():
                partial = Path(temporary) / name
                subprocess.run(["curl", "-fsSL", "--retry", "2", "--max-time", "300",
                                source["download_url"], "-o", str(partial)], check=True)
                verify(partial, source)
                partial.replace(path)
            else:
                verify(path, source)

        export = output / f"{match}-full.csv"
        if not export.exists():
            with TemporaryDirectory(dir=output) as export_dir:
                partial = Path(export_dir) / export.name
                subprocess.run([sys.executable, str(Path(__file__).with_name("build_dfl_sample.py")),
                                "--metadata", str(folder / "metadata.xml"),
                                "--tracking", str(folder / "tracking.xml"),
                                "--full-match", "--output", str(partial)], check=True)
                partial.replace(export)
    print(f"{match}: inputs verified and full export available", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    with urlopen(RELEASE, timeout=30) as response:
        release = json.load(response)
    if release["license"]["url"] != LICENSE:
        raise ValueError("Unexpected license on IDSSE v1")
    args.output.mkdir(parents=True, exist_ok=True)
    with ThreadPoolExecutor(max_workers=3) as pool:
        list(pool.map(lambda match: prepare(match, release["files"], args.data, args.output), MATCHES))


if __name__ == "__main__":
    main()
