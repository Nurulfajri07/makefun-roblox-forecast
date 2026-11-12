# Roblox Up-and-Coming Forecast — make.fun bounty entry

Entry for [make.fun bounty `roblox-up-and-coming-forecast-01`](https://make.fun/bounty/roblox-up-and-coming-forecast-01)
("Predict Roblox's Up-and-Coming top 10", $500).

## What it does

`main.py` (Python 3.13, standard library only, no network, deterministic)
reads the chart history at `/input/history.ndjson` + `/input/task.json` and
writes `/output/forecast.json`: for each of places 1–10, a probability for
each game named at that place, plus the top 10 in order. That one forecast is
scored against every hourly read of the chart in the 72 hours after entries
close.

## Method

For every game ever seen on the chart, fit a recency-weighted linear trend
of its rank over the last ~5 days and extrapolate it over the scored window
with mean reversion toward the middle of the chart. At each future hour the
game's rank is a heavy-tailed (two-component normal mixture) distribution
around that extrapolated rank; averaging over the window gives its
per-place probabilities. Games near the 28-day age limit (from games/v1
`details.created` when the history carries it, else from first-seen time)
fade out as they age off the chart, and games no longer on the chart fade
with time since last seen. Only games whose modelled probability beats what
they would get as unnamed are named; ~25% of each place's mass is left
unnamed so it spreads over every other game on a future read's chart —
including brand-new games, which reach the top 10 about 8% of the time and
cannot be named in advance. Probabilities are deliberately spread: the log
score punishes confident wrong calls far more than it rewards confident
right ones.

## Run it

```sh
mkdir -p out
MAKEFUN_INPUT_DIR=<kit>/input MAKEFUN_OUTPUT_DIR=out python3 main.py
# writes out/forecast.json
```

## Backtest

On the public chart archive (6 rolling 72-hour windows, 2026-10-30 to
2026-11-12): mean log score **-37.8 nats**, worst window -40.6. On the
kit's practice window: **-35.2 nats** (6.9 top-10 hits), vs -43.5 for the
house persistence baseline.

## License

MIT — see `LICENSE`.
