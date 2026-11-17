# rda2run · Phase 1

This repository reproduces our More Than a Hack 2027 possession submission. The model estimates whether play is live and which team controls the ball. A temporal decoder turns those estimates into a sequence for each half.

## Generate the submission

Use Python 3.14.5. Put the organizers' tactical tracking files in `data/tactical/` with their original names, `BAR-ATH_tactical_atd.xml` and `BAR-ATH_tactical_atd.txt`.

```sh
sh scripts/bootstrap.sh
.venv/bin/python scripts/run_submission_1.py infer \
  --model models/submission-1.json --data data --output artifacts/submission-1
```

The command writes and validates `artifacts/submission-1/rda2run.csv`. For a repeat run, pass a fresh output directory. The expected SHA-256 is `2491ba0e306b993d1b883c6b2a6a6291701fcc0013bbe23fe7b564cba0f1cb0b`.

The included parameters were fitted on external DFL matches. See the [technical document](docs/SUBMISSION_1.md) for the method, data attribution and training steps.
