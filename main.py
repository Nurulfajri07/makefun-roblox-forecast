#!/usr/bin/env python3
"""Forecast Roblox's Up-and-Coming chart top 10 (make.fun bounty
roblox-up-and-coming-forecast-01).

Reads /input/history.ndjson (hourly chart snapshots: each game's place,
name and player count) and /input/task.json, writes /output/forecast.json:
for each of places 1..10, a probability for each game named at that place,
plus the top 10 in order. One static forecast, scored against every hourly
read of the 72 hours after entries close.

Method (stdlib only, deterministic, no network, no wall-clock):
  For every game ever seen on the chart, fit a weighted linear trend of its
  rank over the last ~5 days (recent reads weighted more) and extrapolate it
  over the scored window with mean reversion toward the middle of the chart.
  At each future hour the game's rank is a heavy-tailed (two-component
  normal mixture) distribution around that extrapolated rank; averaging over
  the window gives its per-place probabilities. Games close to the 28-day
  age limit (from games/v1 `details.created` when the history carries it,
  else from first-seen time) fade out as they age off the chart, and games
  not seen lately fade with time since last seen. Only games whose modelled
  probability beats what they would get as unnamed are named; the rest of
  each place's mass is left unnamed so it spreads over every other game on
  a future read's chart -- including brand-new games, which reach the top 10
  about 8% of the time and cannot be named in advance.

  Probabilities are deliberately spread (each place names ~20 games and
  leaves ~25% unnamed): the log score punishes confident wrong calls far
  more than it rewards confident right ones.

Backtest (public archive, 6 rolling 72 h windows, 2026-10-30..2026-11-12):
  mean log score -37.8 nats, worst window -40.6; on the kit's practice
  window -35.2, vs -43.5 for the house persistence baseline.
"""

import json
import math
import os
from datetime import datetime, timezone

INPUT_DIR = os.environ.get("MAKEFUN_INPUT_DIR", "/input")
OUTPUT_DIR = os.environ.get("MAKEFUN_OUTPUT_DIR", "/output")

# --- tuned constants (see module docstring for the backtest) -----------------
SIG0 = 2.5        # rank-diffusion sigma at the window's start (ranks)
SIG1 = 0.6        # sigma growth per sqrt(hour)
DRIFT_CAP = 0.4   # max rank drift per hour (ranks/hour)
TAU_REV = 120.0   # mean-reversion timescale toward the chart middle (hours)
FADE_H = 48.0     # presence half-life for games no longer on the chart (hours)
MASS = 0.8        # named mass per place; 1 - MASS is left unnamed
MIN_P = 0.004     # only name a game if its modelled p is at least this
TOPN = 100        # max named games per place (rules cap)
MIX_W = 0.25      # weight of the wide component of the rank kernel
MIX_MULT = 3.0    # wide component's sigma multiplier
H_STEP = 6        # horizon grid step (hours)
TREND_WINDOW_H = 120.0  # rank-trend fit window (hours)
TREND_HALF_LIFE_H = 48.0  # recency weight half-life in the trend fit (hours)
AGE_LIMIT_DAYS = 28.0     # chart only lists games published in the last 28 days
EPS = 1e-6        # probability floor (rules floor the log at 1e-4 anyway)


def parse_time(s):
    """ISO-8601 UTC timestamp -> hours since the epoch."""
    return datetime.fromisoformat(s.replace("Z", "+00:00")).timestamp() / 3600.0


def norm_cdf(x):
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))


def load_history(input_dir):
    snaps = []
    with open(os.path.join(input_dir, "history.ndjson"), encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                snaps.append(json.loads(line))
    snaps.sort(key=lambda s: s.get("at", ""))
    return snaps


def created_by_uid(snaps):
    """Latest games/v1 `created` per universe id, from `details` when present.

    `details` is null in the archive reads and a per-game record in the
    house recorder's reads; accept either a dict keyed by universe id or a
    list of records carrying a universe id.
    """
    created = {}
    for snap in snaps:
        details = snap.get("details")
        if not details:
            continue
        if isinstance(details, dict):
            items = details.items()
            for key, rec in items:
                if isinstance(rec, dict) and rec.get("created"):
                    try:
                        created[int(key)] = rec["created"]
                    except (TypeError, ValueError):
                        pass
        elif isinstance(details, list):
            for rec in details:
                if not isinstance(rec, dict):
                    continue
                uid = rec.get("universe_id") or rec.get("universeId")
                if uid is not None and rec.get("created"):
                    try:
                        created[int(uid)] = rec["created"]
                    except (TypeError, ValueError):
                        pass
    return created


def build_forecast(snaps, places, horizon_hours):
    t_last = parse_time(snaps[-1]["at"])
    created = created_by_uid(snaps)

    # Per-game observations: (hours relative to last read, rank, log playing).
    obs = {}
    first_seen = {}
    for snap in snaps:
        t = parse_time(snap["at"]) - t_last
        for row in snap.get("up_and_coming") or []:
            uid = int(row["universe_id"])
            playing = row.get("playing") or 0
            obs.setdefault(uid, []).append(
                (t, float(row["rank"]), math.log(max(playing, 1)))
            )
            if uid not in first_seen:
                first_seen[uid] = t
    horizons = list(range(0, int(horizon_hours) + 1, H_STEP))
    if horizons[-1] != horizon_hours:
        horizons.append(int(horizon_hours))

    raw = {}  # uid -> [P(place k) averaged over the window, k = 1..places]
    for uid in sorted(obs):
        points = sorted(obs[uid])
        r0 = points[-1][1]
        hours_since_seen = -points[-1][0]

        # Weighted linear rank trend over the recent window.
        recent = [(t, r) for (t, r, _lp) in points if t >= -TREND_WINDOW_H]
        drift = 0.0
        if len(recent) >= 2:
            weights = [math.exp(t / TREND_HALF_LIFE_H) for (t, _r) in recent]
            sw = sum(weights)
            mt = sum(w * t for w, (t, _r) in zip(weights, recent)) / sw
            mr = sum(w * r for w, (_t, r) in zip(weights, recent)) / sw
            den = sum(w * (t - mt) ** 2 for w, (t, _r) in zip(weights, recent))
            if den > 1e-9:
                drift = (
                    sum(w * (t - mt) * (r - mr) for w, (t, r) in zip(weights, recent))
                    / den
                )
        drift = max(-DRIFT_CAP, min(DRIFT_CAP, drift))

        # Presence: fade games that have dropped off the chart.
        presence = math.exp(-hours_since_seen / FADE_H) if hours_since_seen > 0 else 1.0

        # Age-out: the chart only lists games published in the last 28 days.
        # With a real `created` date, fade out across the 28-day boundary;
        # without one, first-seen time is a lower bound on age, so only fade
        # games first seen more than 20 days ago.
        exit_h = None
        if uid in created:
            exit_h = parse_time(created[uid]) + AGE_LIMIT_DAYS * 24.0 - t_last
        elif (0.0 - first_seen[uid]) / 24.0 > 20.0:
            exit_h = first_seen[uid] + AGE_LIMIT_DAYS * 24.0

        weights = [0.0] * places
        for h in horizons:
            mu = r0 + drift * h
            mu = 49.0 + (mu - 49.0) * math.exp(-h / TAU_REV)
            sig = SIG0 + SIG1 * math.sqrt(h + 1.0)
            surv = presence
            if exit_h is not None:
                if h >= exit_h + 6.0:
                    surv = 0.0
                elif h > exit_h - 6.0:
                    surv *= max(0.0, (exit_h + 6.0 - h) / 12.0)
            if surv <= 0.0:
                continue
            sig_wide = sig * MIX_MULT
            for k in range(1, places + 1):
                p = norm_cdf((k + 0.5 - mu) / sig) - norm_cdf((k - 0.5 - mu) / sig)
                p_wide = norm_cdf((k + 0.5 - mu) / sig_wide) - norm_cdf(
                    (k - 0.5 - mu) / sig_wide
                )
                weights[k - 1] += surv * ((1.0 - MIX_W) * p + MIX_W * p_wide)
        raw[uid] = [w / len(horizons) for w in weights]

    probabilities = []
    for k in range(1, places + 1):
        cands = sorted(
            ((raw[uid][k - 1], uid) for uid in raw if raw[uid][k - 1] > 1e-5),
            reverse=True,
        )[:TOPN]
        total = sum(v for v, _uid in cands)
        if total <= 0:
            continue
        for value, uid in cands:
            p = MASS * value / total
            if p >= MIN_P:
                probabilities.append(
                    {"universe_id": uid, "place": k, "p": round(max(p, EPS), 6)}
                )

    # Top 10: most total top-10 mass first, expected rank breaks ties.
    def sort_key(uid):
        total_mass = sum(raw[uid])
        exp_rank = (
            sum((k + 1) * raw[uid][k] for k in range(places)) / total_mass
            if total_mass > 0
            else 99.0
        )
        return (-total_mass, exp_rank, uid)

    top10 = sorted(raw, key=sort_key)[:places]
    # Fixed row order, so identical input bytes give identical output bytes.
    probabilities.sort(key=lambda r: (r["place"], -r["p"], r["universe_id"]))
    return {"probabilities": probabilities, "top10": top10}


def main():
    with open(os.path.join(INPUT_DIR, "task.json"), encoding="utf-8") as fh:
        task = json.load(fh)
    places = int(task.get("places", 10))
    output_name = task.get("output", "forecast.json")
    window = task.get("window") or {}
    horizon_hours = int(window.get("slots", 72)) * int(window.get("slot_minutes", 60)) // 60

    snaps = load_history(INPUT_DIR)
    if not snaps:
        raise SystemExit("no snapshots in history.ndjson")
    forecast = build_forecast(snaps, places, horizon_hours)

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    out_path = os.path.join(OUTPUT_DIR, output_name)
    with open(out_path, "w", encoding="utf-8") as fh:
        fh.write(json.dumps(forecast) + "\n")
    return out_path


if __name__ == "__main__":
    main()
