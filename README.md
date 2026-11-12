# Pump.fun top-5 forecaster

Entry for [Predict the top 5 new Solana coins by market cap, 24 hours out](https://make.fun/bounty/pumpfun-top5-forecast-01) (`pumpfun-top5-forecast-01`).

`main.py` (Python 3.13, standard library only, no network) reads the house's
10-minute snapshots of every pump.fun coin launched since the opening
(`/input/history.ndjson` + `/input/task.json`) and writes one forecast to
`/output/forecast.json`:

- for each of the 5 places, a probability for each coin named being at that
  place at an answer read;
- `top5`: the best guess at the top 5 in order (the tie-break);
- for each place, a probability for each size bin its market cap falls in
  (82 bins of log10(cap in SOL), 0.05 wide, edges 2.00..6.00).

## Model

Monte-Carlo trajectory forecaster:

- Every coin seen in the history gets a log-cap trajectory. Its level is its
  last seen log10 cap (SOL); its drift is an age prior (young runners peak
  within ~30-60 min and then fade fast, ~-0.3 dex/hour; old coins still
  ranked are the durable ones), with only a small weight on the recent
  slope for old coins -- a short rise is the peak, not a trend. An extreme
  rise in the last 10 minutes mean-reverts.
- Forward, drift decays toward a slow long-run decay, volatility grows as
  sqrt(horizon), and a per-hour death hazard (highest for coins under 1h
  old) collapses a coin to the bonding-curve floor.
- Curve coins may graduate: with a per-hour chance they jump to the
  young-runner cap distribution (the history shows curve coins graduating
  and reaching 1000+ SOL within minutes).
- Newcomers (coins launched after the cut, which no forecast can name) are
  simulated as fresh runners whose caps are drawn from the young-runner
  peak distribution fitted from the history; the share of places they take
  is left unnamed, exactly as the scorer shares it over the chart.
- All horizons come from `task.json` (answer reads minus `history_until`),
  so the same program serves the 1-hour practice and the 7-13h final.

Deterministic: a fixed seed; two runs on the same history write the same
forecast byte for byte.

## Local test

```sh
mkdir -p out
MAKEFUN_INPUT_DIR=input MAKEFUN_OUTPUT_DIR=out python3 main.py
```

where `input/` holds the kit's `history.ndjson` + `task.json`.

## License

MIT. See LICENSE.
