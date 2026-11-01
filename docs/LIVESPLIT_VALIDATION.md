# Validating vis-split against a real LiveSplit

Everything in the automated suite and the integration runs used a **fake** LiveSplit
server written from LiveSplit's source. This guide checks the assumptions that fake
cannot prove, then walks the OCR undo feature from a harmless dry run up to live use.

**Golden rule: use a throwaway splits file** (a copy with the same number of segments
as the game, or at least 3) for every step that sends commands. Never your PB splits.
When you reset afterwards, choose **not** to update/save the splits.

## Assumptions the probe checks (and what depends on them)

| # | Assumption | Used by | Probe check |
|---|---|---|---|
| A2/A3 | `getcurrenttimerphase` ∈ NotRunning/Running/Paused/Ended; `getsplitindex` is -1 / 0..N-1 / N | follow_timer, undo | A2, A3 |
| A4 | `getsplitcount` exists | skip validation | A4 |
| A5 | unknown commands get no reply | client | A5 |
| A6 | `unsplit` from **Ended** goes back to Running at N-1 | last-split undo | A6 |
| A7 | `unsplit` is applied **before** the next `getsplitindex` reply (else late-apply path: ~0.3 s re-checks, then follow_timer repairs) | undo executor | A7 |
| A8/A9 | `start` works from NotRunning; a second `start` while Running is ignored | auto_start_timer | A8, A9 |
| A10 | `split` advances one segment | all splits | A10 |
| A11 | `split` is ignored while Paused | `follow_pause` (default off) | A11 (`--test-pause`) |
| A12 | `unsplit` while Running → N-2 | undo | A12 |
| A13 | `split` while Ended is ignored | a split after the finish | A13 |
| A14 | `unsplit` at index 0 is a no-op | death on level 1 | A14 |

## Results of the first real-LiveSplit run (2026-10-30)

Run by the project owner on their own LiveSplit with a throwaway 4-segment splits file
(`probe_readonly.json`, `probe_active.json`):

| Check | Result |
|---|---|
| A0 ping, A2/A3 phase strings and index values, A5 unknown command silent | PASS (replies end in `\n`) |
| **A4 `getsplitcount`** | **not implemented by this LiveSplit server** (no reply) |
| A8/A9 `start` from NotRunning; a second `start` ignored | PASS |
| A10 `split` advances one segment | PASS |
| **A11 `split` ignored while Paused** | PASS (`follow_pause` assumption holds) |
| A12 `unsplit` while Running -> N-2 | PASS |
| **A6 `unsplit` from Ended -> Running at N-1** | PASS, 5/5 rounds (last-split undo is possible) |
| **A7 `unsplit` applied before the next `getsplitindex` reply** | PASS, 5/5 immediate, 0 ms (no late-apply path needed here) |
| A13 `split` while Ended ignored; A14 `unsplit` at index 0 is a no-op | PASS |
| A15 `undosplit` alias | INFO (inconclusive at index 0; the detector only sends `unsplit`) |

Consequences:

- Because `getsplitcount` is missing, vis-split prints `split count unavailable; skip-forward runs
  without validating against len(splits)` once per connection. **The number of splits in your game
  config must equal the number of segments in the LiveSplit splits file** (Start Powerless: 15);
  nothing else checks it.
- The ordering/Ended/Paused assumptions behind undo, last-split undo and `follow_pause` now have
  one real-world confirmation. Other LiveSplit versions, or a different Wine setup, may differ:
  re-run the probe when you change LiveSplit.
- Still not validated for real: real OBS/SRT compression of the HUD, real HUD lag and
  disappearance timing, real death visuals (Steps 3-6 below).

## Step 0: prepare

1. Under Wine, open LiveSplit with the **throwaway** splits file.
2. Right-click → **Control → Start TCP Server** (needed every time LiveSplit starts).
3. Check the port (default 16834): `nc -zv 127.0.0.1 16834`.

## Step 1: read-only probe (safe)

```fish
cd ~/Documents/coding/"auto splitters"
uv run python scripts/livesplit_probe.py --report probe_readonly.json
```

Sends only queries plus one deliberately bogus command (must get no reply). Expect
PASS for A0, A2, A3, A5; A4 may WARN on an old LiveSplit (then skip-forward runs
without count validation, which is documented).

## Step 2: active probe (drives the timer!)

With the timer **NotRunning** (press Reset first), on the throwaway splits:

```fish
uv run python scripts/livesplit_probe.py --active --test-pause --report probe_active.json
```

It asks you to type `START`, then starts the timer, splits to the last segment, runs
the Ended→`unsplit` ordering test 5×, and checks the no-op cases. It never sends
`reset` (add `--allow-reset` if you want it to); press Reset yourself afterwards.
`--test-pause` also pauses/resumes (checks A11, the `follow_pause` assumption).

How to read it:

- **A6 FAIL** → do **not** enable `apply_to_last_split` (last-split undo needs Ended→Running).
- **A7 PASS** → the executor's immediate verify works. **A7 WARN** (late but <275 ms) is
  handled by its re-checks. **A7 FAIL** (later than ~275 ms) → undo still works only
  because `follow_timer` (required) repairs it; expect "confirmed late"/"could not
  confirm" lines. Consider whether that is acceptable before enabling `mode: undo`.
- **A11 FAIL** (a paused LiveSplit *accepts* `split`) → keep `follow_pause: false`.
- Any other FAIL → stop and report it (attach `probe_*.json`); the detector assumes it.

## Step 3: dry run on real footage (no LiveSplit involved)

Record 3-5 real runs in OBS (include deaths and a death on the last level). Replay them
**in real time** into the splitter. Do not pass the file as the source: it would decode
at thousands of fps and the judge's timers are wall-clock.

```fish
cp configs/start-powerless.yaml /tmp/replay.yaml
# edit /tmp/replay.yaml: capture.fps/width/height = the recording's; DELETE the `record:` block
uv run python -m autosplitter.main --config /tmp/replay.yaml --live 2>&1 | tee replay1.log
# in a second terminal:
ffmpeg -re -i run1.mkv -an -c:v libx264 -preset veryfast -tune zerolatency -bf 0 -g 60 \
  -f mpegts "udp://127.0.0.1:9733?pkt_size=1316"
```

No `--livesplit`: nothing can reach LiveSplit. Then read the three tuning lines:

```fish
grep -E "OCR:|HUD advanced|would UNDO|HUD gone|SPLIT fired|WARN" replay1.log
```

Acceptance for this step:
- every real **death** has exactly one `would UNDO … dry run: nothing sent` about 4 s after its `SPLIT fired`;
- **no genuine level** transition has a `would UNDO`; its `HUD advanced +X.XXs` is **below ~2.7 s** (settle 3.0 s minus margin; your stated lag is 1-2 s);
- the final real finish prints `HUD gone after the last split — run finished, no undo` and never `would UNDO`;
- `ocr=N=` (agree) dominates the status column; read confidences stay ≥ 85.

If a genuine level shows `would UNDO`, raise `ocr.undo.settle_seconds` (≈ 1.5× the largest `HUD advanced`).
If deaths show no `would UNDO`, check `ocr=`/`--check`, the ROI, the pattern and the confidence floor.

## Step 4: live game, still dry run (throwaway splits)

Same config (`mode: verify_only`), now with LiveSplit and the splitter running normally,
plus `follow_timer` so you also exercise that path:

```fish
# in your game config: livesplit.follow_timer: true
uv run python -m autosplitter.main --config configs/start-powerless.yaml --live --livesplit --loop 2>&1 | tee live1.log
```

Play 3-5 runs including deaths. Look for `desync`/`SKIP/manual split`/`UNSPLIT detected` lines
that you did not cause, and for the same `would UNDO` pattern as in Step 3.

## Step 5: `mode: undo` (opt-in), last split OFF first

Edit the game config: `ocr.mode: undo`, `livesplit.follow_timer: true` (required — the config
is rejected otherwise), `ocr.undo.apply_to_last_split: false`. `--check` should print
`ocr undo : ACTIVE (will send unsplit)`.

```fish
uv run python -m autosplitter.main --config configs/start-powerless.yaml --check
uv run python -m autosplitter.main --config configs/start-powerless.yaml --live --livesplit 2>&1 | tee undo1.log
```

Play ~10 runs with deaths. Acceptance:
- every death → one `OCR: UNDO … confirmed` then `LiveSplit: UNSPLIT sent -> index k (verified)`; LiveSplit and the game agree afterwards;
- **zero** unsplits on genuine levels;
- `grep -c "UNSPLIT sent" undo1.log` equals the number of deaths you had;
- no `could not confirm unsplit`, no `undo regret` (a regret means a genuine level was wrongly undone — raise `settle_seconds` and re-test).

## Step 6: last split (only after Steps 2-5 passed, and A6 passed)

Set `apply_to_last_split: true`. First do **one more dry run** (`mode: verify_only`) over several real
finishes and confirm `HUD gone after the last split` each time and **no** `would UNDO` on a finish.
Then `mode: undo`. Remember the documented risk: if the HUD number lingers ≥ ~6 s after a real
finish, the run is un-ended with no automatic recovery (finish by hand; follow_timer follows it).

## Kill switches

- `ocr.mode: verify_only` → no commands from OCR at all.
- Omit `--livesplit` → nothing is sent to LiveSplit (undo degrades to a dry run with a startup WARN).
- `ocr.undo.recover_late_hud: false` → no automatic re-split after a wrong undo.
- `git revert` the feature commit for a full rollback.

## Reporting back

Keep `probe_readonly.json`, `probe_active.json` and the `*.log` files from `tee`. The log lines
(`OCR:`, `LiveSplit:`, `SPLIT fired`, `desync`) are the audit trail.
