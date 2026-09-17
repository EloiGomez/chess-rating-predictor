# Chess Rating Predictor

Machine learning project that predicts a player's **Lichess** rating (their
Glicko-2 Elo on lichess.org, specifically -- not a FIDE/USCF rating or a
generic "chess skill" score, and not transferable to those) from a
handful of their own games in PGN format, either fetched live from the
Lichess API or uploaded as a PGN file.

The core idea: the outcome of a single game is noisy (a bad day, an odd
opponent), but averaging move-quality features over 5-15 games of the
same player gives a much more stable signal to predict their rating from.

## Results

Trained on real Lichess data (August 2026 dump), one dedicated model per
time control, evaluated on a held-out 20% of players. As of the latest
run, players are sampled not just evenly across the Elo range but also
across each category's most common exact time controls (see "Design
notes"), so the pool is 3x bigger and considerably more diverse than
earlier versions of this table:

| Time control | Players | Model | MAE (Elo points) | R² |
|---|---|---|---|---|
| Bullet | 9,221 | Ridge | 286 | 0.749 |
| Bullet | 9,221 | Random Forest | 211 | 0.849 |
| Bullet | 9,221 | **Gradient Boosting (deployed)** | **202** | **0.859** |
| Blitz | 8,970 | Ridge | 212 | 0.832 |
| Blitz | 8,970 | Random Forest | 176 | 0.878 |
| Blitz | 8,970 | **Gradient Boosting (deployed)** | **168** | **0.887** |

For context: always predicting the dataset mean gets a MAE of roughly
580-615 points, so the model is explaining the large majority of the
variance in players' ratings from move-quality features alone -- without
ever being told the player's Elo history.

The single most useful thing that got these numbers this high wasn't a
better model -- it was **sampling training players evenly across the
whole Elo range** (see "Design notes" below) instead of however a scan of
the dump happens to encounter them. Before that fix, R² on the same
features and models was 0.28-0.62.

**A note on these specific numbers going slightly down from an earlier,
smaller version of this table** (bullet was 192 MAE / 0.889 R² on 3,258
players; blitz was 164 / 0.918 on 3,437): that's not a regression in the
model. It's testing on a harder, more honest population. The larger run
deliberately added players from rare Elo bands and minority time controls
(down to a handful of real players in some cells) that the smaller,
more curated pool never had to contend with. A spot-check against random
real players outside the curated training set (`spot_check.py`) backs
this up: the *old* blitz model's real-world MAE against truly random
players was ~211-213, well above its own formal 164 -- while the *new*
model's real-world MAE came out at 205.5, closer to its (higher-looking)
formal 168. The gap between "how the model scores on its own test set"
and "how it actually does against a random stranger" shrank from ~47
points to ~37. That gap closing is the real signal that this dataset
change helped, even though the headline MAE number went up.

Most influential features (bullet):
`avg_num_plies` (average game length) dominates on its own (0.51) --
under bullet's time pressure, how long a player's games tend to last says
more about their level than raw move accuracy.

Most influential features (blitz):
`avg_opening_acpl` dominates heavily (0.76). Take this one with a grain
of salt: it's highly correlated with the overall `avg_acpl` and the other
phase-specific ACPL features, and Random Forest importances tend to dump
most of the credit onto one feature from a correlated cluster rather than
splitting it evenly -- so this reads more as "move quality in general
matters most" than "the opening specifically is what matters."

![Feature importance, bullet vs blitz](docs/feature_importance.png)

*(Random Forest importances shown for both, since HistGradientBoostingRegressor
-- the winning model for both categories -- doesn't expose them at all;
regenerate with `python src/plot_feature_importance.py`.)*

## Try it: the Streamlit demo

```bash
streamlit run app.py
```

Opens a local web app where you type a Lichess username (or upload/paste
a PGN) and get back a predicted rating for bullet or blitz, computed from
their own recent games -- run through the exact same Stockfish analysis
and feature pipeline the models were trained on. Needs at least 5 games
in the chosen time control, and a trained model in `models/` (see
"Full pipeline" below to build one, or train against the sample data).
If `train_baseline.py` was run with `--save-model-dir`, all three models
(Ridge / Random Forest / Gradient Boosting) get saved, and the app lets
you pick which one to use, not just the winner. The prediction also comes
with a rough ± range based on the model's own MAE, and an expander
showing exactly which features fed into that prediction (versus ones that
were computed but aren't used by the currently-loaded model).

## Project structure

```
chess-rating-predictor/
├── app.py                        # Streamlit demo (predicts from a Lichess username or an uploaded PGN)
├── data/
│   ├── raw/                      # original PGNs (not committed, except the small samples)
│   └── processed/                # CSVs/PGNs generated by the scripts
├── docs/                         # charts referenced from this README
├── models/                       # trained models (not committed -- regenerate with train_baseline.py)
├── src/
│   ├── parse_pgn.py              # MVP: PGN -> per-game CSV with basic features, no engine
│   ├── extract_sample.py         # cuts the first N games out of a huge Lichess .pgn.zst
│   ├── select_games_by_player.py # scans a full Lichess dump, selects games per player per time control
│   ├── analyze_with_stockfish.py # runs Stockfish (in parallel, resumable) for move-quality features
│   ├── extract_clock_features.py # derives time-per-move from the PGN's clock comments
│   ├── aggregate_by_player.py    # averages per-game features into one row per player
│   ├── train_baseline.py         # trains and compares models (Ridge / Random Forest / Gradient Boosting)
│   ├── predict_from_pgn.py       # shared feature extraction for live predictions (used by app.py)
│   ├── lichess_api.py            # fetches a player's games from the public Lichess API
│   ├── plot_feature_importance.py # renders the chart in the Results section above
│   └── benchmark_stockfish.py    # measures analysis speed at a given depth before committing to a run
├── requirements.txt
└── README.md
```

## Setup

1. Create a virtual environment:
   ```bash
   python -m venv .venv
   source .venv/bin/activate      # on Windows: .venv\Scripts\activate
   ```
2. Install the dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Download [Stockfish](https://stockfishchess.org/download/) and place the
   executable somewhere the scripts can find it (see `STOCKFISH_PATH` in
   `analyze_with_stockfish.py` / `benchmark_stockfish.py`).

## Quick check with the sample PGN

The repo includes `data/raw/sample.pgn` (3 sample games) and
`data/raw/lichess_sample.pgn` (20,000 real games) so you can verify the
basic pipeline works without downloading anything:

```bash
python src/parse_pgn.py --input data/raw/sample.pgn --output data/processed/sample_games.csv
```

This should generate `data/processed/sample_games.csv` with one row per
game (White/Black rating, opening, number of plies, result).

## Full pipeline with real Lichess data

1. **Download a Lichess dump.** Go to
   [database.lichess.org](https://database.lichess.org/) and grab one of
   the monthly files (`.pgn.zst`, tens of GB compressed). Every script
   here streams the decompression on the fly -- none of them ever write
   the fully decompressed file to disk, so you only need free space for
   the compressed download itself.

2. **Select a manageable, player-grouped subset -- one dedicated dataset
   per time control:**
   ```bash
   python src/select_games_by_player.py
   ```
   Edit the constants at the top of the script (`ZST_PATH`,
   `TARGET_SPEED_CATEGORIES`, `MIN_GAMES_PER_PLAYER`,
   `MAX_GAMES_PER_PLAYER`, `TARGET_PLAYERS_PER_CATEGORY`,
   `ELO_BUCKET_WIDTH`) to match your setup. Writes one PGN per category,
   e.g. `data/processed/selected_games_bullet.pgn`. Meant to be left
   running unattended: two full passes over the dump (a few hours each on
   a monthly file), progress logged to `logs/` in UTF-8, and if
   interrupted it still saves whatever was already selected.

3. **Extract move-quality features with Stockfish** (repeat per category):
   ```bash
   python src/analyze_with_stockfish.py \
       --input data/processed/selected_games_bullet.pgn \
       --output data/processed/games_engine_bullet.csv \
       --max-games 200000 --depth 12
   ```
   **`--max-games` defaults to 2,000 if you omit it** -- fine for a quick
   test, but pass a number comfortably above your selected PGN's actual
   game count (check with `grep -c "^\[Event" selected_games_bullet.pgn`)
   or the run will silently stop early and every step downstream
   (aggregation, training) will be working off a tiny, incomplete slice.
   Runs several Stockfish processes in parallel (`--workers`, defaults to
   roughly half the machine's logical CPU count, since Stockfish gets
   little benefit from hyperthreading) and is the slowest step by far --
   use `benchmark_stockfish.py --depth N` first to check how a given
   depth trades off against speed before committing to a full run.
   Checkpoints to `--output` every 200 games and, if that file already
   exists when you (re-)run it, resumes from it instead of redoing
   already-analyzed games -- safe to re-run after any interruption,
   including a crash, or to deliberately pause a long run (e.g.
   `TaskStop`/Ctrl+C) and continue it later.

4. **Extract time-per-move from clock comments** (optional but recommended;
   repeat per category):
   ```bash
   python src/extract_clock_features.py \
       --input data/processed/selected_games_bullet.pgn \
       --output data/processed/games_clock_bullet.csv
   ```

5. **Aggregate by player** (repeat per category):
   ```bash
   python src/aggregate_by_player.py \
       --input data/processed/games_engine_bullet.csv \
       --clock-input data/processed/games_clock_bullet.csv \
       --output data/processed/players_bullet.csv
   ```
   Collapses the per-game CSV into one row per player, averaging their
   features over their games (restricted to whichever time control they
   actually played, in case games from another format snuck in).

6. **Train and save a model** (repeat per category):
   ```bash
   python src/train_baseline.py \
       --data data/processed/players_bullet.csv \
       --speed-category bullet \
       --save-model-dir models
   ```
   Trains Ridge, Random Forest and Gradient Boosting, reports MAE/R² for
   each, and (with `--save-model-dir`) saves all three separately (e.g.
   `models/model_bullet_gradient_boosting.joblib`) plus the best one under
   the plain `models/model_bullet.joblib` name, for `app.py` to load.

## Design notes

A few non-obvious decisions, in case the reasoning matters more than the
code:

- **Sample players evenly across the Elo range, not however they show up.**
  Early versions just took whichever eligible players appeared first while
  scanning the dump -- since a small number of very active players near
  the median contribute a disproportionate share of raw games, this
  produced a training set concentrated around the median with almost no
  players at the tails. A model trained on that skew is unreliable for
  anyone far from the middle (a 3200-rated player looks nothing like
  anyone it's seen). `select_games_by_player.py` now buckets eligible
  players into `ELO_BUCKET_WIDTH`-point bands and samples a roughly even
  number from EACH band (capped by how many actually exist in a sparse
  band). This one change took R² from 0.28-0.62 to the 0.85-0.92 range
  seen in the Results table above (the exact number moves around a bit as
  the dataset and features evolve, but the jump from this fix is the
  single biggest lever pulled in this project).

- **Never mix time controls for the same player.** Lichess gives every
  player a separate rating per speed (someone can be 2400 in bullet and
  2000 in rapid). A dataset that mixes a player's bullet and blitz games
  would average together ratings -- and Stockfish features, since bullet
  games are short and sloppy almost regardless of skill -- that don't mean
  the same thing. `select_games_by_player.py` builds a fully separate,
  dedicated player pool per time control from the start.

- **Exclude the opponent's Elo, but keep the opponent's move quality.**
  Lichess's matchmaking pairs similarly-rated players, so a player's own
  Elo and their average opponent's Elo turn out correlated at ~0.97 in
  this data -- including it lets a model predict Elo almost entirely via
  the opponent's Elo, trivially inflating R² without learning anything
  from actual play. The opponent's ACPL/blunders don't have that problem
  (matchmaking doesn't pair by how cleanly someone happens to play in one
  game) and are kept: a sloppy game from the opponent is a sign the
  position itself was messy, which puts the player's own ACPL in that
  game into context.

- **Predict from several games, not one.** The whole "5-15 games,
  averaged" design exists because a single game's outcome is dominated by
  noise (an opponent's blunder, a bad day). Averaging Stockfish-derived
  features over several games of the same player is much more stable, and
  is what actually makes prediction from move quality feasible at all.

- **Normalize time-per-move by that game's own clock, but don't rely on it
  alone.** "Bullet" spans several actual time controls with very different
  clocks (e.g. 60+0 vs 120+1, roughly double the budget). A raw
  seconds-per-move average isn't comparable across them: a player who
  exclusively plays 120+1 got mispredicted by 500+ Elo points because
  their time usage looked nothing like the 60+0 majority the model mostly
  learned from. `avg_time_ratio` (time spent as a fraction of that
  specific game's budget) helps, but a direct test of *replacing*
  `avg_time_per_move` with it made the model meaningfully worse overall,
  so it's kept as an additional feature rather than a replacement. See the
  next two notes for the more complete fix that followed.

- **Stratify sampling by exact time control too, not just Elo -- weighted
  by real popularity.** The Elo-only stratification above still let
  whichever time control happens to be most common within a speed
  category (e.g. bullet's 60+0) dominate the sample, which is exactly what
  caused the 120+1 mispredictions above. `select_games_by_player.py` now
  buckets players by (Elo band, dominant exact time control), capping the
  time-control axis to each category's `N_TOP_TIME_CONTROLS_PER_CATEGORY`
  most common ones (everything rarer gets grouped as "other"). The split
  across those buckets is weighted by the **square root** of each time
  control's real population, not evenly and not fully proportionally:
  fully even would make a time control played by 12K people count for
  almost as much of the sample as one played by 291K, making the dataset
  unrepresentative of a typical real user; fully proportional would swing
  back to the original 120+1 problem, crushing rare formats down to almost
  nothing. The square root is a deliberate middle ground. `aggregate_by_player.py`
  then restricts each player to only their games at that single dominant
  exact time control (not just the same speed category) when computing
  their averages -- and `predict_from_pgn.py` applies the same restriction
  live, but *only* when the loaded model was actually trained with it
  (checked via whether `base_time` is one of its feature columns), so an
  older deployed model doesn't silently get fed features computed a
  different way than it learned from.

- **Split move quality by game phase.** A single ACPL number per game
  hides *where* in the game a player is strong or weak -- someone can be
  booked-up and precise in the opening but shaky in technique once
  simplified, or vice versa. `analyze_with_stockfish.py` now classifies
  each move into opening (first 20 plies), middlegame, or endgame (once
  both queens are off the board -- the simplest correct rule available)
  and computes ACPL/blunders/mistakes/inaccuracies separately for each,
  with `predict_from_pgn.py`/`aggregate_by_player.py` falling back to a
  player's overall stat when a short game never reaches a given phase.

## Next steps

- [ ] A dedicated rapid dataset -- currently too small a slice of Lichess
      traffic to train (and evaluate) reliably.
- [ ] Cross-validation instead of a single train/test split, for a more
      robust MAE/R² estimate.
- [ ] Filter out high-RD/provisional players (few total games, unreliable
      Glicko-2 rating) from training and spot-checks -- found to skew at
      least one spot-check outlier (a player with only 9 total games).
- [ ] A quick "estimate from just 1 game" mode, trained separately on
      per-game (not per-player-averaged) rows. A rough experiment on older
      data put single-game R² at only ~0.26 vs ~0.86-0.89 from averaging
      5-15 games, so this would be a deliberately low-precision, fast
      alternative to the main predictor, not a replacement for it.
- [ ] Retest that single-game idea with a much larger player pool (the
      full eligible pool per category, not just the curated few thousand),
      since a single game per player is cheap to collect at scale -- to
      see whether more (if still individually noisy) examples narrow the
      gap at all.
