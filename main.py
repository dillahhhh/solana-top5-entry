#!/usr/bin/env python3
"""Pump.fun top-5 forecaster (pumpfun-top5-forecast-01).

Monte-Carlo trajectory forecaster. Reads the house's 10-minute snapshots of
every pump.fun coin launched since the opening (/input/history.ndjson +
/input/task.json) and writes one forecast (/output/forecast.json):

  - for each of the 5 places, a probability for each coin named being at that
    place at an answer read (the one forecast is scored at every read);
  - top5: the best guess at the top 5 in order (the tie-break);
  - for each place, a probability for each size bin its market cap falls in
    (82 bins of log10(cap in SOL), 0.05 wide, edges 2.00..6.00).

Model (all parameters estimated from the history itself where possible):
  * Every coin seen in the history gets a log-cap trajectory. Its level is
    its last seen log10 cap (SOL); its drift is the recent slope of log-cap,
    shrunk toward an age prior (young coins still rise, old coins fade) and
    clipped; its volatility is estimated from its own increments.
  * Forward, drift decays toward a slow long-run decay (pump.fun runners
    peak and fade; a +0.3 log/h rise cannot last 10 hours), volatility
    grows as sqrt(horizon), and a per-hour death hazard collapses a coin
    to the bonding-curve floor (it leaves the top 5).
  * Curve coins may graduate: with a per-hour chance they jump to the
    young-runner cap distribution (as the history shows: curve coins do
    graduate and reach 1000+ SOL within minutes).
  * Newcomers (coins launched after the cut, which no forecast can name)
    are simulated as fresh runners whose caps are drawn from the young
    runner distribution fitted from the history; the share of places they
    take is left unnamed, exactly as the scorer shares it.
  * All horizons come from task.json (answer reads minus history_until),
    so the same program serves the 1-hour practice and the 7-13h final.

Deterministic: a fixed seed; two runs on the same history write the same
forecast byte for byte. Standard library only, no network.
"""

import json
import math
import os
import random
import sys
from datetime import datetime

INPUT_DIR = os.environ.get("MAKEFUN_INPUT_DIR", "/input")
OUTPUT_DIR = os.environ.get("MAKEFUN_OUTPUT_DIR", "/output")

SEED = 20261112
N_SIMS = 6000

# Tunable via env (v5 grid search; defaults = v5 tuned values. v4's
# shift 0.15 + sharpen 2.0 scored WORSE on the real practice window
# (-37.3) than v3 (-36.6): it overshot. Grid on practice-like cuts:
# shift 0.08 + sharpen 1.5 = -36.87 (best), vs v3 base -38.73.
MIN_P = float(os.environ.get("MAKEFUN_MIN_P", "0.002"))  # smallest prob row written
SHARPEN_ALPHA = float(os.environ.get("MAKEFUN_SHARPEN", "1.5"))  # per-place p^alpha renormalised
SIZE_SD_CAP = float(os.environ.get("MAKEFUN_SIZE_SD_CAP", "0.35"))
SIZE_SD_FLOOR = float(os.environ.get("MAKEFUN_SIZE_SD_FLOOR", "0.15"))
SIZE_SHIFT = float(os.environ.get("MAKEFUN_SIZE_SHIFT", "0.08"))  # v5: MC size means run ~+0.25 dex high
# v6: persistence blend. The v6 "oracle ceiling" (-34.79) was computed on
# cuts at 20:10/20:20 -- the EXTREME-CHURN regime (top-5 overlap 1/5 per
# 10 min). The real practice cut (21:00, 60 min of history) sits in the
# LATE-STABLE regime (20:30-20:50: overlap 4/5, survivors GROW +0.05..0.19
# dex/10min). The MC's age prior (drift -0.32/h, death 0.45/h for <1h coins)
# is exactly backwards there, and its wide size distributions waste the
# size score (sticky sd 0.10 beat MC -8.66 vs -12.24 at a 10-min horizon).
# Blend the MC forecast toward the sticky (last-chart) forecast with a
# weight that decays with mean horizon: practice (mean h ~0.5h) is sticky,
# final (mean h ~10h) stays MC. Log score is concave, so the blend scores
# at least the average of its parts.
PERSIST_TAU = float(os.environ.get("MAKEFUN_PERSIST_TAU", "0.001"))   # order blend tau (hours); v6: OFF (validated: order blend gains read 0 (+5) but loses more at later reads; net <= 0 over 7 reads)
STICKY_TAU = float(os.environ.get("MAKEFUN_STICKY_TAU", "1.5"))     # size blend tau (hours)
STICKY_SD = float(os.environ.get("MAKEFUN_STICKY_SD", "0.12"))      # sticky size sd (dex)
STICKY_SELF = float(os.environ.get("MAKEFUN_STICKY_SELF", "0.55"))  # sticky mass on same place
STICKY_NBR = float(os.environ.get("MAKEFUN_STICKY_NBR", "0.10"))    # sticky mass on +-1 place

# ---- model constants (priors; the history re-estimates what it can) ----
# Practice lesson: pump.fun runners peak within ~30-60 min and then fade
# fast (4 of the 5 top coins at the 20:50 cut were gone by the 22:00 read,
# replaced by fresh runners). A coin's recent rise does NOT continue: the
# forward drift after the peak is negative, and young coins collapse
# fastest. Old coins still in the top 5 are the durable ones.
SIGMA_H = 0.50        # log10-cap volatility per sqrt(hour) for pool coins
SIGMA_CURVE = 0.65    # curve coins are wilder
DRIFT_TAU_H = 2.0     # drift decays toward the long-run decay with this tau
DRIFT_LONG = -0.12    # long-run log drift per hour for a surviving coin
DRIFT_CLIP = (-0.8, 0.5)
DEATH_OLD_H = 0.08    # per-hour collapse hazard, coin older than 3h
DEATH_MID_H = 0.18    # ... 1-3h old
DEATH_YOUNG_H = 0.45  # ... younger than 1h: runners dump fast after peaking
FLOOR_LOG = 1.45      # bonding-curve floor, log10 SOL (~28 SOL)
GRAD_H = 0.18         # per-hour graduation-jump chance for a rising curve coin
NEWCOMER_MU = 3.15    # log10 SOL of a fresh runner at its peak
NEWCOMER_SD = 0.35
NEWCOMER_N = 6        # fresh-runner candidates simulated per read
TAIL_MASS = 0.04      # per place: long-tail chart coins we do not model
MAX_ROWS_PER_PLACE = 100  # HARD LIMIT (grader/forecast/ranked.ts parseRanked):
                          # a place with MORE than 100 probability rows is
                          # not a forecast at all -- the whole file scores
                          # the floor at every place and size (-92.1). v2
                          # wrote 150/171/175 rows at places 3/4/5 and was
                          # scored exactly that. Never exceed this.


def parse_time(s):
    return datetime.fromisoformat(s.replace("Z", "+00:00")).timestamp()


def log_cap_sol(market_cap_lamports):
    return math.log10(int(market_cap_lamports)) - 9.0


def norm_cdf(x, mean, sd):
    """P(X <= x) for X normal(mean, sd); x may be +/-inf."""
    if x == float("-inf"):
        return 0.0
    if x == float("inf"):
        return 1.0
    return 0.5 * (1.0 + math.erf((x - mean) / (sd * math.sqrt(2.0))))


def bin_of(logcap, edges):
    """Size bin 0..81 for log10(cap in SOL); edges are 2.00..6.00 at 0.05."""
    if logcap < edges[0]:
        return 0
    if logcap >= edges[-1]:
        return len(edges)
    # binary search for the edge interval
    lo, hi = 0, len(edges)
    while lo < hi:
        mid = (lo + hi) // 2
        if edges[mid] <= logcap:
            lo = mid + 1
        else:
            hi = mid
    return lo  # edges[lo-1] <= x < edges[lo]; bin = lo


def main():
    with open(os.path.join(INPUT_DIR, "task.json")) as f:
        task = json.load(f)
    places = int(task["places"])
    edges = [float(e) for e in task["size"]["edges"]]
    t0 = parse_time(task["t0"])
    cut = parse_time(task["history_until"])
    reads = [parse_time(r) for r in task["reads"]]
    horizons = [(r - cut) / 3600.0 for r in reads]  # hours after the cut
    horizon_max = max(horizons)
    h_mean = sum(horizons) / len(horizons)
    w_persist = math.exp(-h_mean / PERSIST_TAU)  # practice ~0.61, final ~0.0
    w_size = math.exp(-h_mean / STICKY_TAU)      # practice ~0.72, final ~0.0

    marks = []
    with open(os.path.join(INPUT_DIR, "history.ndjson")) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            m = json.loads(line)
            if m.get("ok") and isinstance(m.get("rows"), list):
                marks.append(m)
    if not marks:
        raise SystemExit("the history holds no ok mark")

    # ---- trajectories: mint -> {obs: [(t_hours, logcap, venue)], created, symbol} ----
    traj = {}
    for m in marks:
        t = (parse_time(m["at"]) - t0) / 3600.0
        for r in m["rows"]:
            mint = r["mint"]
            d = traj.get(mint)
            if d is None:
                d = traj[mint] = {
                    "obs": [],
                    "created": parse_time(r["created_at"]),
                    "symbol": r.get("symbol", ""),
                    "mayhem": bool(r.get("mayhem")),
                }
            d["obs"].append((t, log_cap_sol(r["market_cap_lamports"]), r["venue"]))
    for d in traj.values():
        d["obs"].sort()

    last_mark = marks[-1]
    last_rows = sorted(last_mark["rows"], key=lambda r: r["rank"])
    last_at = (parse_time(last_mark["at"]) - t0) / 3600.0
    last_caps = [log_cap_sol(r["market_cap_lamports"]) for r in last_rows[:places]]
    # top-5 cap trend (dex/hour) from the last two marks: survivors in the
    # late regime grow; clip to [0, 0.8] (never extrapolate a dump).
    growth_h = 0.0
    if len(marks) >= 2:
        prev_by = {r["mint"]: log_cap_sol(r["market_cap_lamports"])
                   for r in sorted(marks[-2]["rows"], key=lambda r: r["rank"])[:places]}
        incs = sorted(log_cap_sol(r["market_cap_lamports"]) - prev_by[r["mint"]]
                      for r in last_rows[:places] if r["mint"] in prev_by)
        if incs:
            dt_h = max((parse_time(last_mark["at"]) - parse_time(marks[-2]["at"])) / 3600.0, 1e-9)
            growth_h = max(0.0, min(0.8, incs[len(incs) // 2] / dt_h))

    # ---- fit the young-runner peak distribution from the history ----
    # A coin's peak log-cap and its age at peak; runners peak young.
    peaks = []
    for d in traj.values():
        if len(d["obs"]) >= 1:
            pk = max(o[1] for o in d["obs"])
            if pk >= 2.5:
                peaks.append(pk)
    if len(peaks) >= 5:
        peaks.sort()
        mu_new = sum(peaks) / len(peaks)
        var = sum((p - mu_new) ** 2 for p in peaks) / len(peaks)
        sd_new = max(0.20, min(0.45, math.sqrt(var)))
        mu_new = max(2.8, min(3.4, mu_new))
    else:
        mu_new, sd_new = NEWCOMER_MU, NEWCOMER_SD

    # ---- per-coin forward parameters ----
    # Age prior for drift (log per hour): a runner's rise is brief; after
    # the peak the drift is negative, and the younger the coin the faster
    # it fades. Old coins still ranked are the durable ones.
    # Practice evidence: 4 of the 5 top coins at the 20:50 cut were gone
    # by the 22:00 read -- young coins dump ~0.3 dex/hour after peaking.
    def age_prior_drift(age_h):
        if age_h < 0.25:
            return 0.15
        if age_h < 1.0:
            return -0.32
        if age_h < 3.0:
            return -0.15
        return -0.08

    # Candidate universe: every coin in the last mark plus any coin that was
    # ever top 10. (Curve coins included: they can graduate and explode.)
    cand_mints = []
    seen = set()
    for r in last_rows:
        cand_mints.append(r["mint"])
        seen.add(r["mint"])
    for m in marks:
        for r in sorted(m["rows"], key=lambda x: x["rank"])[:10]:
            if r["mint"] not in seen:
                seen.add(r["mint"])
                cand_mints.append(r["mint"])

    coins = []
    for mint in cand_mints:
        d = traj[mint]
        obs = d["obs"]
        t_last, y_last, venue_last = obs[-1]
        age_h = max(0.0, (cut - d["created"]) / 3600.0)
        # Recent slope is a trap for young coins: their rise is the peak,
        # not a trend. Only old coins (enough history) get any slope weight,
        # and even then little. An extreme rise in the last 10 minutes
        # mean-reverts hard.
        drift0 = age_prior_drift(age_h)
        pts = [(t, y) for (t, y, v) in obs if t >= t_last - 2.0 and y > 1.6]
        if len(pts) >= 4 and age_h >= 1.0:
            n = len(pts)
            mt = sum(p[0] for p in pts) / n
            my = sum(p[1] for p in pts) / n
            num = sum((p[0] - mt) * (p[1] - my) for p in pts)
            den = sum((p[0] - mt) ** 2 for p in pts)
            slope = num / den if den > 1e-9 else 0.0
            drift0 = 0.8 * drift0 + 0.2 * slope
        if len(obs) >= 2:
            last_jump = obs[-1][1] - obs[-2][1]
            if last_jump > 0.15:
                drift0 -= 0.25
            elif last_jump < -0.15:
                drift0 -= 0.10
        drift0 = max(DRIFT_CLIP[0], min(DRIFT_CLIP[1], drift0))
        # volatility from own increments (floored)
        incs = [obs[i + 1][1] - obs[i][1] for i in range(len(obs) - 1)]
        if len(incs) >= 3:
            mi = sum(incs) / len(incs)
            var = sum((x - mi) ** 2 for x in incs) / len(incs)
            sig = max(0.20, min(0.70, math.sqrt(var) * math.sqrt(6.0)))
        else:
            sig = SIGMA_H
        if venue_last == "curve":
            sig = max(sig, SIGMA_CURVE)
        coins.append({
            "mint": mint,
            "y": y_last,
            "drift0": drift0,
            "sig": sig,
            "age": age_h,
            "curve": venue_last == "curve",
            "mayhem": d["mayhem"],
        })

    # ---- Monte Carlo ----
    rng = random.Random(SEED)
    n_reads = len(reads)
    # place_counts[place][mint] and size_counts[place][bin]
    place_counts = [dict() for _ in range(places)]
    newcomer_counts = [0] * places
    tail_counts = [0] * places
    size_counts = [[0] * (len(edges) + 1) for _ in range(places)]
    top5_votes = [dict() for _ in range(places)]

    # precompute per-read newcomer cap draws are done inside the loop
    for _ in range(N_SIMS):
        # per-coin simulated log-cap at each read: list per coin of list per read
        # (draw once per coin per read; death/graduation applied per read)
        for ri, h in enumerate(horizons):
            # drift at horizon h: decays toward DRIFT_LONG
            decay = math.exp(-h / DRIFT_TAU_H)
            scored = []  # (logcap, mint or None for newcomer, is_newcomer)
            for c in coins:
                # death: young runners collapse fastest after their peak
                if c["age"] < 1.0:
                    lam = DEATH_YOUNG_H
                elif c["age"] < 3.0:
                    lam = DEATH_MID_H
                else:
                    lam = DEATH_OLD_H
                if rng.random() < 1.0 - math.exp(-lam * h):
                    y = FLOOR_LOG + rng.random() * 0.1
                else:
                    d_eff = c["drift0"] * decay + DRIFT_LONG * (1.0 - decay)
                    y = c["y"] + d_eff * h + c["sig"] * math.sqrt(h) * rng.gauss(0, 1)
                    if c["curve"] and rng.random() < 1.0 - math.exp(-GRAD_H * h):
                        # graduates and jumps to a young-runner level
                        y = max(y, rng.gauss(mu_new, sd_new))
                scored.append((y, c["mint"], False))
            # newcomers: fresh runners born after the cut
            n_new = rng.randint(3, NEWCOMER_N + 1)
            for _ in range(n_new):
                scored.append((rng.gauss(mu_new, sd_new), None, True))
            # rank: top `places` (and a little more for the tail mass calc)
            scored.sort(key=lambda x: -x[0])
            # tail: with a small chance a long-tail chart coin takes a place;
            # model it as: each place has TAIL_MASS probability of going to the tail
            for k in range(places):
                y, mint, is_new = scored[k]
                if rng.random() < TAIL_MASS:
                    tail_counts[k] += 1
                    # size still from the simulated cap at that place
                    size_counts[k][bin_of(y, edges)] += 1
                    continue
                if is_new:
                    newcomer_counts[k] += 1
                else:
                    d = place_counts[k]
                    d[mint] = d.get(mint, 0) + 1
                    top5_votes[k][mint] = top5_votes[k].get(mint, 0) + 1
                size_counts[k][bin_of(y, edges)] += 1

    total = N_SIMS * n_reads

    def floor6(p):
        return math.floor(p * 1_000_000) / 1_000_000

    probabilities = []
    for k in range(places):
        named = 0.0
        rows = []
        # Top MAX_ROWS_PER_PLACE by p only: the rows dropped return their
        # mass to the place's unnamed share (shared by the chart's unnamed
        # coins, i.e. newcomers) -- honest and proper.
        for mint, cnt in sorted(place_counts[k].items(), key=lambda kv: -kv[1])[:MAX_ROWS_PER_PLACE]:
            p = floor6(cnt / total)
            if p >= MIN_P:
                rows.append({"mint": mint, "place": k + 1, "p": p})
                named += p
        # v6: blend toward the sticky (last-chart) forecast. Same-place mass
        # STICKY_SELF, +-1 place STICKY_NBR; weight decays with horizon.
        if w_persist > 0.01:
            st = {}
            for kk, r in enumerate(last_rows[:places], 1):
                if kk == k + 1:
                    st[r["mint"]] = STICKY_SELF
                elif abs(kk - (k + 1)) == 1:
                    st[r["mint"]] = STICKY_NBR
            merged = {}
            for r in rows:
                merged[r["mint"]] = (1.0 - w_persist) * r["p"]
            for mint, sp in st.items():
                merged[mint] = merged.get(mint, 0.0) + w_persist * sp
            rows = [{"mint": m, "place": k + 1, "p": floor6(p)}
                    for m, p in merged.items() if floor6(p) >= MIN_P]
            rows.sort(key=lambda r: -r["p"])
            rows = rows[:MAX_ROWS_PER_PLACE]
        # v4: sharpen -- p^alpha renormalised to the same named sum.
        # Long-horizon backtests show the winners are usually among the top
        # candidates; concentrating on them gains order score. alpha=1 is v3.
        if SHARPEN_ALPHA != 1.0 and rows:
            w = [r["p"] ** SHARPEN_ALPHA for r in rows]
            tw = sum(w)
            if tw > 0:
                for r, wi in zip(rows, w):
                    r["p"] = floor6(named * wi / tw)
                rows = [r for r in rows if r["p"] > 0]
        # unnamed mass = newcomers + tail + the sub-threshold named mass.
        # (Anything not named is shared by the scorer over the chart's
        # unnamed coins, which is exactly the newcomer/tail treatment.)
        probabilities.extend(rows)

    # top5: most-voted coin at each place, distinct
    top5 = []
    used = set()
    for k in range(places):
        best = None
        best_v = -1
        for mint, v in sorted(top5_votes[k].items(), key=lambda kv: -kv[1]):
            if mint not in used and v > best_v:
                best, best_v = mint, v
        if best is None:
            # fall back to the last mark's order
            for r in last_rows:
                if r["mint"] not in used:
                    best = r["mint"]
                    break
        top5.append(best)
        used.add(best)
    # v6: short horizons -> the tie-break top5 is the last chart's order.
    if w_persist > 0.3:
        top5 = [r["mint"] for r in last_rows[:places]]

    # Sizes v2: fit a normal (in log10 SOL) to the simulated caps at each
    # place, sd floored at 0.15 and capped at 0.35. The raw histogram is
    # diffuse/bimodal (decayed survivors + newcomers); a tight unimodal
    # distribution around the simulated level scores better: the place
    # sizes are roughly stationary (new runners replenish the top) even
    # as the coins churn.
    sizes = []
    for k in range(places):
        tot = sum(size_counts[k])
        if tot == 0:
            continue
        centres = []
        for b in range(len(edges) + 1):
            c = size_counts[k][b]
            if c == 0:
                continue
            if b == 0:
                cen = edges[0] - 0.025
            elif b == len(edges):
                cen = edges[-1] + 0.025
            else:
                cen = (edges[b - 1] + edges[b]) / 2.0
            centres.append((cen, c))
        mean_raw = sum(c * n for c, n in centres) / tot
        var = sum(n * (c - mean_raw) ** 2 for c, n in centres) / tot
        mean = mean_raw - SIZE_SHIFT  # v4: MC size means run ~+0.25 dex high
        sd = max(SIZE_SD_FLOOR, min(SIZE_SD_CAP, math.sqrt(var)))
        # v6: blend the MC size normal toward a tight sticky normal centred
        # on the last cap at this place + trend growth over the mean horizon.
        mean_sticky = last_caps[k] + growth_h * h_mean
        for b in range(len(edges) + 1):
            lo = float("-inf") if b == 0 else edges[b - 1]
            hi = float("inf") if b == len(edges) else edges[b]
            p_mc = norm_cdf(hi, mean, sd) - norm_cdf(lo, mean, sd)
            p_st = norm_cdf(hi, mean_sticky, STICKY_SD) - norm_cdf(lo, mean_sticky, STICKY_SD)
            p = floor6((1.0 - w_size) * p_mc + w_size * p_st)
            if p > 0:
                sizes.append({"place": k + 1, "bin": b, "p": p})

    forecast = {"probabilities": probabilities, "top5": top5, "sizes": sizes}

    # ---- self-validation (mirrors grader/forecast/ranked.ts parseRanked) ----
    # An invalid forecast scores the floor everywhere (-92.1). Fail loudly
    # instead of writing one.
    assert set(forecast.keys()) == {"probabilities", "top5", "sizes"}
    assert len(top5) == places and len(set(top5)) == places
    for k in range(places):
        pr = [r for r in probabilities if r["place"] == k + 1]
        assert len(pr) <= MAX_ROWS_PER_PLACE, f"place {k+1} has {len(pr)} rows"
        assert len({r["mint"] for r in pr}) == len(pr)
        assert all(set(r.keys()) == {"mint", "place", "p"} and 0 <= r["p"] <= 1 for r in pr)
        assert sum(r["p"] for r in pr) <= 1 + 1e-9
        sr = [r for r in sizes if r["place"] == k + 1]
        assert len({r["bin"] for r in sr}) == len(sr)
        assert all(set(r.keys()) == {"place", "bin", "p"} and 0 <= r["bin"] <= len(edges) and 0 <= r["p"] <= 1 for r in sr)
        assert sum(r["p"] for r in sr) <= 1 + 1e-9
    blob = (json.dumps(forecast) + "\n").encode()
    assert len(blob) <= 1_048_576, f"forecast {len(blob)} bytes over 1 MiB"

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    with open(os.path.join(OUTPUT_DIR, task.get("output", "forecast.json")), "w") as f:
        f.write(json.dumps(forecast) + "\n")


if __name__ == "__main__":
    main()
