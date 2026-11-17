# rda2run · Phase 1 technical document

## Approach

Submission 1 combines two logistic regression models with Architecture A's three-state hidden Markov decoder. One estimates whether play is live using ball movement, the distance to the nearest player, player counts and ball visibility. A centered three-second window provides offline context. The second estimates which team controls the ball from current and one-second nearest-player distance advantages. Mirroring home and away during fitting and averaging the mirrored predictions makes the team signal symmetric.

The decoder combines the probability of dead ball with the conditional probability of home or away control, then finds the most likely state sequence independently for each half. State persistence lets a possession continue across a loose or missing ball when team evidence is weak. A brief change in nearest player is only evidence of a turnover. Dead ball ends the spell, and the decoded controlling team starts the next spell when play resumes. Missing observations are handled with finite observations from the local window and means fitted on external training data.

## Assets and training data

For the competition match, inference reads the official tactical tracking XML and TXT. It converts positions to metres for distance and ball movement features, uses the official tactical frame numbering, and assigns `X` outside the two scored halves. The frozen submission uses tracking only. The broadcast feed, MP4s, homography, and Smart Tagging were outside this variant's inputs.

All fitted parameters come from IDSSE v1, seven public 2022/23 DFL matches with 1,002,644 labeled tracking frames at 25 Hz. DFL supplied synchronized tracking and match data. Its recorded ball status and team in control supply the training targets. The release is CC BY 4.0. Four matches fitted the models (`J03WOH`, `J03WOY`, `J03WPY`, `J03WQQ`), `J03WR9` was used for development, and `J03WN1` and `J03WMX` were used as external benchmarks. The live model used 227,328 examples after 5 Hz subsampling and adding masked copies. The team model uses labeled live frames at 5 Hz with mirrored copies. Model fitting, tuning, and selection used external data only.

[IDSSE v1 release](https://doi.org/10.6084/m9.figshare.28196177.v1) · [Bassek et al. dataset paper](https://doi.org/10.1038/s41597-025-04505-y) · [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/)

## Evaluation and limitations

Using the official spellF1 implementation on the two DFL benchmarks, the combined variant averaged 0.3273 on original tracking and 0.2776 with simulated gaps. The adapted official baseline averaged 0.2955 and 0.2175, respectively. The stress test hides four consecutive seconds of ball observations in every ten seconds and retains player tracks. The selected variant was the strongest on these external comparisons.

DFL's labeling conventions and tracking system can differ from the challenge. Periodic gaps approximate only one kind of missing observation. The nearest-player model can confuse a loose ball with team control. On one masked DFL match, the combined model matched 68 of 101 restarts within one second and made 29 spurious restart calls. More representative external matches and better evidence of sustained control would help evaluate those errors. The competition score remains unknown.

## Reproduction

The public submission repository includes the frozen parameters and a command that reproduces the CSV from the supplied tactical tracking files. See the [repository README](https://github.com/yasinnerten/rda2run-public) for setup and inference. The verified run used Python 3.14.5 and the pinned dependency versions.

To rebuild the model, unpack the IDSSE release under `data/external/idsse-v1`, with `metadata.xml` and `tracking.xml` in each match directory as described by the release. Keep these inputs local.

From a fresh checkout, run:

```sh
sh scripts/bootstrap.sh
external="$PWD/data/external/idsse-v1"
exports="$PWD/artifacts/external/idsse-v1"
output="$PWD/artifacts/submission-1"

.venv/bin/python scripts/prepare_dfl_matches.py --data "$external" --output "$exports"
for match in J03WOH J03WOY J03WPY J03WQQ; do
  .venv/bin/python scripts/build_dfl_sample.py --full-match --possession \
    --metadata "$external/$match/metadata.xml" \
    --tracking "$external/$match/tracking.xml" \
    --output "$exports/possession/$match.csv"
done
.venv/bin/python scripts/train_dfl_live_ball.py --exports "$exports" \
  --output "$output/reproduction/training"
.venv/bin/python scripts/run_submission_1.py freeze --exports "$exports/possession" \
  --live-model "$output/reproduction/training/model.json" \
  --output "$output/reproduction/frozen-model.json"
.venv/bin/python scripts/run_submission_1.py infer \
  --model "$output/reproduction/frozen-model.json" \
  --data "$PWD/data" --output "$output"
```

The inference command validates `rda2run.csv` with the pinned official validator. The verified CSV contains 199,319 ordered tactical frames from 0 through 199318. Its SHA-256 is `2491ba0e306b993d1b883c6b2a6a6291701fcc0013bbe23fe7b564cba0f1cb0b`.
