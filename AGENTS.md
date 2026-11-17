# Working on this repository

Start with README.md and docs/SUBMISSION_1.md. The checked-in model and upstream revision pin identify the submitted version.

- Keep match assets, tracking exports, videos and generated outputs in ignored directories.
- Preserve the submitted model and its output checksum. Keep experiments separate from the reproduction command.
- Use the official validator to check frame order and file format. Use the public five-minute labels only for pipeline checks. Train and select variants on permitted external matches.
- When changing the reproduction path, regenerate the CSV in a fresh output directory and compare its checksum with the README.
