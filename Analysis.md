# Predicting Arrests in Traffic Stops: Analysis

**Data:** 3,383,056 traffic stops from five state police agencies: Montana (MT), Nevada (NV), Rhode Island (RI), Vermont (VT) and Wisconsin (WI).
**Target:** `is_arrested`, true for 79,278 stops (2.34%).
**Source notebooks:** `Data Scientist Case Study_EDA.ipynb` (sections 1–2) and `Data Scientist Case Study_FeatureEngineeringTraining.ipynb` (sections 1–6).

## Summary

* The table is really **five different state datasets stacked together**. Each state recorded different fields, used different vocabularies and has a different base arrest rate (1.2%–5.7% after cleaning). Almost every analysis below is therefore done per state.
* Several fields are **the target in another form** (`stop_outcome`) or are decided *because of* the arrest (e.g. "search incident to arrest", warrant charges). These were identified and excluded.
* Some periods carry **unreliable labels**: Nevada recorded almost no arrests before mid-2014. Montany, Rhode Island and Wisconsin all have inconsistent labels in the first months of recording. These entries proved to be outliers using the Iglewicz and Hoaglin method and were removed from the data set.
* Four free-text fields were turned into **interpretable, cross-state features**: search basis, stop outcome, location and charges hidden in raw violation text. The original fields are: search_type_raw, stop_outcome, fine_grained_location, violation_raw
* Four models were tuned with 5-fold grouped cross-validation. **LightGBM wins:** on the untouched natural test set it scores **average precision 0.476** (19× chance), **ROC AUC 0.931** and **F1 0.497**, and it is the best model in every state.

---

## Contents
1. [Exploratory data analysis](#exploratory-data-analysis)
2. [Feature engineering](#feature-engineering)
3. [Results](#results)

---

# Exploratory data analysis

### How to read the charts
* Each state keeps one fixed color in every chart: MT blue, NV orange, RI aqua, VT yellow, WI pink.
* **Arrest-rate heatmaps** use a single blue scale.
* **Lift heatmaps** show *rate in a group ÷ that state's overall rate* on a blue ↔ red scale centred on 1.0×. Lift is used because
  base rates differ by state, so it shows whether a feature matters *within* a state.
* Groups with too few stops are left blank rather than shown as noisy rates.

## 1. Data coverage: what each state actually recorded

![Share of non-missing values by state](docs/figures/01_coverage_by_state.png)

* The five states are close to five different schemas. Only `stop_date`, `violation(_raw)`, the four boolean flags, `stop_outcome` and `is_arrested` are populated everywhere.
* **NV is the sparsest**: there is no time of day, no location of any kind, no gender, and no search details. Only age, race and violation describe an NV stop.
* **WI has no driver age** and is missing time of day on 27% of stops. **RI has no county**; its only geography is 6 patrol zones (`district`).
* `search_type`/`search_type_raw` exist for only a few percent of rows: VT's 100% is mostly the literal string "No Search Conducted". `aerial_enforcement` and `search_reason` are empty.
* MT-only fields (`ethnicity`, `city`, `vehicle_*`, `stop_outcome_raw`), RI-only fields (`stop_duration`, `district`) and the WI-only `vehicle_type` can only ever help within one state.

**Implication:** missingness is almost perfectly determined by state. A model can learn "this row has no `county_name`" = NV/RI =
higher base rate, which is state identity rather than stop behavior. Every feature should be read as *state-conditional*.

## 2. Data quality: boolean flags that are constant within a state

The boolean columns are never null, which hides a problem: in some states they are **the same value for every single stop**.

| State | Flag(s) constant for every stop |
|---|---|
| NV | `search_conducted`, `contraband_found`, `out_of_state` = True for 100% of 737k stops |
| MT, VT | `drugs_related_stop` = True for every stop |
| VT, WI | `out_of_state` = True for every stop |
| MT | `contraband_found` is an exact copy of `search_conducted` (100% agreement) |

These are fill-in defaults, not observations, so they are treated as **missing** in those states (the `*_clean` columns). Used raw, they would make a model believe that *every* NV stop involved a search with contraband found, at an NV arrest rate of 3.6%. That would drag the learned effect of searching far down.

The parquet's stored index also restarts at 0 for each state (2.3M duplicate labels), which silently breaks joins. It is reset before
any processing. There are no duplicate rows or ids.

## 3. The target across states and over time

![Arrest rate by state and by year](docs/figures/02_arrest_rate_by_state_year.png)

* `is_arrested` is **highly imbalanced** (2.34%). The arrest rate ranges **3×** across states, from VT at 1.2% to NV at 3.6%. NV and RI together supply 54% of all arrests from only 36% of stops, so a pooled model's positives are dominated by those two states.
* The **time windows differ**: RI runs 2005–2015, MT 2009–2016, NV 2012–2016, VT and WI 2010–2016. The first and last years are partial for most states, and MT's partial 2009 (18k stops) has an unusually high 7.5% rate.
* MT, RI and WI drift **downward** over time. NV shows a cliff-like **upward jump**.

### Nevada: a change in how arrests were recorded

![NV monthly arrest rate](docs/figures/03_nv_monthly_arrest_rate.png)

* NV's arrest rate is **0.39% before July 2014 and 6.05% after**, with the transition happening over about 2 months (May–July 2014). Need to confirm the cause here. Suspician is that this 15× step does not represent a real-world policing change, so arrests were almost certainly **not being recorded** in the earlier dates.
* The single date **2012-02-14 holds 59,570 stops (8.1% of NV) with zero arrests**, and their violations are 67% "Other (non-mapped)" / 33% "Paperwork". This looks like a bulk back-load stamped with one date, not real daily activity.
* **Implication:** about 324k NV rows (44% of NV) likely carry **false-negative labels**. They are removed in [§12](#12-validating-the-arrest-rate-over-time-and-removing-outlier-periods).

## 4. Target-leakage audit

* Every `(state, stop_outcome)` combination maps to exactly one `is_arrested` value. `is_arrested` is literally derived from `stop_outcome` ("Arrest", "Arrest Driver", "Arrest Passenger", "Arrest for Violation", "Warrant Arrest").
* RI's **"Arrest Passenger"** (1,973 stops) counts as an arrest, so in RI the target includes stops where the driver was not arrested.
* **Excluded from modeling:** `stop_outcome`, `stop_outcome_raw` and anything derived from them. The search-type and raw-violation analyses below identify further post-stop fields (see [Feature sets](#5-feature-sets-and-the-leakage-decision)).

## 5. Driver demographics

![Arrest rate by driver age](docs/figures/04_arrest_rate_by_age.png)

* **Gender:** male drivers are arrested at 1.1–1.2× their state's base rate and female drivers at 0.6–0.7×. MT shows no gender difference.
* **Race:** Black and Hispanic drivers are arrested at 1.5–2.5× the state base rate in NV, RI, VT and WI (NV Black drivers: 8.9% vs
  3.0% for white drivers). MT shows no gap for Black or Hispanic drivers, but "Other" (which includes Native American drivers) is 1.5×.
  Direction and size are state-specific.
* In VT, stops with **missing race** almost never end in arrest (0.18% vs 1.20%), a recording artifact.
* **Age** is one of the clearer gradients. Risk peaks in the mid-20s to early 30s (NV ~5.5%, RI ~4.4%) and declines to under 1% by 70.
  MT is flatter (~2–2.5% until the mid-50s). WI has no age.
* Race and gender are predictive but are **protected attributes**. Including them is an ethical and legal decision, not just a
  statistical one (see [Feature sets](#5-feature-sets-and-the-leakage-decision)).

## 6. Time of stop

![Arrest rate by hour and day of week](docs/figures/05_hour_and_weekday.png)

* **Hour of day has a strong, consistent shape**: arrests concentrate between midnight and 4am (WI ~8% at 2am, RI ~6–7%, MT spiking at 3am),
  bottom out around 6–8am (<1%), and rise slowly through the evening. Late-night stops are a different population (impaired driving, fewer
  routine speeding stops).
* **Weekends** run 1.2–1.6× the base rate (NV Sunday 1.6×, VT/WI Saturday 1.3×), consistent with the overnight/DUI pattern.
* This motivates a `night` flag and a cyclical hour encoding, since 23:00 and 01:00 are neighbors.

## 7. Violations

### Standardized violation (multi-label)

![Violation arrest rate and lift](docs/figures/06_violation_lift.png)

`violation` is standardized but **multi-valued** in WI and MT (e.g. "Paperwork,Speeding", 1,500+ distinct strings). It was split into one
indicator per category plus a count.
* **DUI** is the strongest: 10.8× base in MT, 16.7× in VT, 23.7× in NV (84.5% of NV DUI stops end in arrest).
* **"Other" / "Other (non-mapped)"** come second (2.3×–15×), which is surprising for a catch-all bucket (explained next).
* Routine violations (**speeding, truck, cell phone, seat belt**) carry *below*-average risk (0.1–0.4× in NV/RI/WI).
* Some categories **flip direction by state**: registration/plates is 2.5× in RI but 0.3× in MT, and equipment is 1.7× in RI but 0.2× in NV.
  The same label means different things in different states' coding.
* **More violations means more arrests**: the rate rises steadily with the number of violations cited (MT and WI).

### What hides inside "Other": charges in `violation_raw`

![Raw charge flags](docs/figures/07_violation_raw_charges.png)

* **Failure-to-appear / warrant charges end in arrest 98% of the time in NV** and account for **25.7% of NV arrests**.
* Alcohol charges account for 18% of NV arrests and **41.7% of WI arrests** (71% of WI alcohol-charge stops end in arrest). Drug charges
  run ~31% arrest in MT and NV.
* Warrant, drug and alcohol charges together appear on **49% of NV, 42% of WI and 18% of MT arrests**.
* **Suspended/revoked license** is the main *routine* signal: 25% arrest rate in WI and 10% in MT, on 1–2% of stops.
* RI and VT record only coarse violation categories, so these flags are essentially absent there.
* These charges are typically **recorded as a result of the stop**: a warrant hit after running the license, a failed sobriety test, drugs found during a search.

## 8. Search, contraband, drug and residency flags

![Search and contraband flags](docs/figures/08_search_contraband_flags.png)

With placeholder states removed:
* **Searches are the strongest behavioral signal**: searched stops end in arrest 26% (MT), 35% (RI) or 28% (VT) of the time, 10–23× the base rate.
* **WI is different**: 15.7% of WI stops are flagged as searched (vs 0.4–3.7% elsewhere), but only 5.5% of those end in arrest, and only 9% of WI
  "searches" have a recorded basis. The same column measures something broader in WI.
* **Contraband found** and **drug-related stop** push arrest rates to 27–56%, most extremely in WI (contraband 51%; drug-related 54%).
* **Out-of-state drivers** are *less* likely to be arrested in RI (0.5×) and slightly more in MT (1.2×). This is only measurable in those two states.
* A search does **not** imply an arrest: roughly 65–95% of searched stops end without one, and most MT and VT arrests involve no search.
  All of these flags are decided *during* the stop.

## 9. Geography

![County, zone and department arrest rates](docs/figures/09_geography_spread.png)

* **MT has extreme between-county variation**: from ~0% to **15.8%** (Powell County, 23.5k stops), with Glacier (13.1%), Deer Lodge (10.3%)
  and Rosebud (7.9%) also far above the state's 2.1%. Most MT counties sit below 1%.
* Other states are comparatively homogeneous: RI zones 2.4–4.7%, VT counties 0.3–2.3%, WI counties 0.4–3.7%.
* Department/barracks adds little beyond county in VT and WI.

## 10. EDA of the engineered free-text features

### Search basis (`search_type_raw`)
* Raw values from four vocabularies were mapped to 11 legal bases. Every raw value maps except one VT row.
* The basis is recorded for **100% of searches in MT, RI and VT but only 9% in WI** (14k of 166k).
* **"Incident to arrest"** and **"inventory/tow"** searches happen *because of* an arrest, so they leak by definition. Even so, only 44–62% of
  them have `is_arrested` = True (the arrest may have been of a passenger), which shows the label itself carries some noise.
* Among the other bases, **probable-cause** searches are the most arrest-prone (WI 69%, MT 44%). **Consent** varies widely (MT/VT ~23–26% vs
  WI 63%). Reasonable-suspicion and frisk searches are the least arrest-prone (11–23%).

### Stop outcome (`stop_outcome`, MT `stop_outcome_raw`)
* Outside arrests, enforcement style differs: **NV (95%) and RI (89%) almost always cite**, while MT, VT and WI split roughly evenly between
  citations and warnings.
* In MT, the multi-valued raw field matches `is_arrested` exactly. 97.5% of MT arrests are misdemeanors and 4% felonies. 37% of MT arrests also
  list a warning and 12.5% a citation.
* This confirms the leakage conclusion: these are diagnostic only.

### Location (`fine_grained_location`, MT/VT/WI)

| ![Road class](docs/figures/10_road_class.png) | ![Interstate corridor](docs/figures/11_interstate_corridor.png) |
|---|---|

* The rules classify all but a small residual (MT 6.6%, VT 1.1%, WI 1.6% unclassified). What remains is genuine free text: landmarks, typos, bare mileposts.
* **Road class matters, differently by state**: local streets carry the highest risk in WI (2.2×) and VT (1.5×). MT interstates are 1.4× and US
  highways 0.5×. County roads run 1.2–1.6× everywhere.
* **Milepost stops** (between-exit highway stops, mostly speed enforcement) are *below* base in every state (0.6–0.9×).
* **Corridor:** MT's **I-90 runs at 4.1%**, twice MT's base and 4× MT's I-15. I-90 passes through the high-arrest Powell and Deer Lodge counties,
  so corridor and county overlap.

### Vehicle age (MT, WI)
* **WI shows a strong, monotonic gradient**: about 0.3% for new vehicles rising to ~2.5% for vehicles 15+ years old (8×). This matters because
  WI lacks driver age.
* **MT is flat** (~2% at every age). The same feature is strong in one state and useless in another.

## 11. Feature relationships

* **`state` is strongly associated with almost everything** (Cramér's V 0.4–0.73), mostly through **missingness**.
* Several large feature-to-feature values are the **same artifact**: gender ↔ hour (0.64), gender ↔ search (0.69) and hour ↔ search (0.63)
  are high because NV is missing gender, hour *and* the cleaned search flag together.
* Genuine relationships are moderate: contraband ↔ search (0.74, partly MT's copied column), n_violations ↔ paperwork/license (0.42–0.57),
  search ↔ race (0.52).
* **Pooled association with `is_arrested` is weak for every single feature** (all ≤ 0.2 except the leaky outcome). Arrest is driven by a few
  specific signals, not broad shifts.

![Within-state strength of each feature](docs/figures/12_within_state_strength.png)

**Within each state** (Cramér's V with `is_arrested`):
* The **violation description is the strongest pre-stop feature everywhere** (0.12–0.57). It is extreme in NV only because "Other" hides warrant charges.
* **Location is decisive in MT only** (county 0.23). Elsewhere county/zone is weak (≤ 0.05).
* **Hour** is moderate where recorded (0.06–0.09). **Race** is moderate in NV, RI and WI. **Vehicle age** is moderate in WI and nil in MT.
  **Suspended license** is the strongest routine signal in WI (0.19).
* **Post-stop features dominate**: alcohol charges (WI 0.54), warrants (NV 0.50), search (0.10–0.34), contraband (0.10–0.34),
  search-incident-to-arrest (RI 0.45).
* Each state has a different set of available features **and** different strengths for the same feature. This is the clearest evidence
  that a pooled model without state context would blur five different relationships.

## 12. Validating the arrest rate over time and removing outlier periods

![Monthly arrest rate with removed periods](docs/figures/13_outlier_period_timeline.png)

**Yearly view:** VT is the only state stable throughout (1.11–1.34%). MT, RI and WI all start high and settle: MT 2009 is ~4.5× its recent
level and 2010 ~1.8×, RI 2005 ~2.4×, and WI's tiny 2010 is elevated. NV runs the other way (2012–2013 at ~3% of its recent level). The confidence
intervals are narrow, so these gaps are not sampling noise.

**Monthly rule (preferring the most recent years):** each month is scored against the state's latest 3 years with a robust z-score.

| State | Dropped | Why |
|---|---|---|
| MT | through Nov 2010 | 2009 ramp-up months (10–70 stops/month at 11–70% arrest rates, mainly arrest records captured) and a 2010 plateau ≈ 3% |
| NV | through Apr 2014 | almost no recorded arrests. May 2014 (≈1%) is kept as a transition month, sitting just inside the threshold. |
| RI | through Jan 2006 | 2005 ramp-up: 2–70 stops/month, erratic 0–15% rates |
| WI | through Nov 2010 | a trickle of 1–19 stops/month before the full feed |
| VT | nothing | never leaves its band |

![Monthly rate distributions before and after](docs/figures/14_monthly_rate_distribution.png)

* **9 isolated outlier months** inside the kept windows (MT 2011–12, RI 2006 and 2009–10, WI Jan 2011) were also dropped. **All 9 are high**
  and sit in the older part of the window, so they reflect the long-run decline as much as one-off anomalies. Dropping them follows the
  recent-years preference but slightly lowers the base rate. This is a setting (`drop_isolated`).
* **15.1% of stops removed** (511,743): 440k before a structural break (mainly NV and MT), 71k in isolated months, and 109 with no date.
* Base rates move in different directions: **NV rises from 3.56% to 5.69%**, MT falls from 2.08% to 1.71%, RI from 3.45% to 3.27%, while WI and VT
  barely change. The **between-state gap widens from 3× to almost 5×**, which is the more accurate picture.

---

# Feature engineering

All feature engineering lives in reusable functions in `Data Scientist Case Study_FeatureEngineeringTraining.ipynb`, called by one entry point:
`engineer_features(raw, params=None)`. Called without `params`, it **fits** the two data-dependent pieces: the placeholder-flag table and the
monthly keep/drop table. Passing the returned `params` applies **exactly the same rules** to new data, so a small new batch is never judged
against itself. Everything else is deterministic regex or mapping rules, collected in one cell so they are easy to review.

## 1. Pipeline steps and the reasoning behind them

| # | Step | Source | Output | Decision and reason |
|---|---|---|---|---|
| 1 | Index reset | – | – | The stored index restarts per state, which breaks alignment. |
| 2 | Placeholder flags → missing | 4 boolean flags | `*_clean` | Constant-per-state values are defaults, not data (EDA §2). Treating them as missing stops them distorting the search/contraband effects. |
| 3 | Time | `stop_date`, `stop_time` | `year`, `month`, `day_of_week`, `is_weekend`, `hour`, `hour_sin/cos`, `is_night` | The strong overnight and weekend pattern (EDA §6). The cyclical encoding keeps 23:00 next to 01:00. Nullable integer types keep dtypes identical between batches. |
| 4 | Violation, multi-label | `violation` | 15 × `viol_*`, `n_violations` | 1,500+ combination strings would fragment the signal. Indicators plus a count capture both the type and the "stacking" effect (EDA §7). |
| 5 | Raw charges | `violation_raw` | 9 × `vraw_*` | Reveals warrant, drug, alcohol and license-suspension charges hidden in "Other" (EDA §7). |
| 6 | Search basis | `search_type_raw` | 11 × `search_*`, `search_basis_known`, `n_search_bases` | Four state vocabularies and multi-valued strings mapped onto common legal bases. `search_type_raw` is used because the normalized column drops VT's and MT's values. |
| 7 | Stop outcome | `stop_outcome`, MT `stop_outcome_raw` | `outcome_std`, 8 × `outcome_raw_*` | A standardized severity scale and the MT arrest types. **Diagnostic only**: this is the target in another form. |
| 8 | Location | `fine_grained_location` | `road_class`, `has_milepost`, `is_intersection`, `is_ramp_exit`, `interstate_id` | See below. |
| 9 | Geography and vehicle | `county_name`/`district`, `police_department`, `vehicle_year`, WI `vehicle_type` | `geo_unit`, normalized department, `vehicle_age` | RI has zones instead of counties. RI department codes mix "500"/"500.0". WI hides the model year at the end of `vehicle_type`. |
| 10 | Outlier-period removal | `state`, `stop_date`, `is_arrested` | `period_status`, `period_keep` | Removes recording-artifact periods (EDA §12). Months *after* the fitted range are kept for new data. |

## 2. Turning the free-text fields into features

**`fine_grained_location`**: formats differ completely by state, so geocoding was not attempted. The text was reduced to features
that plausibly relate to arrest risk:
* **WI** is semi-structured, `"<direction> <highway number> <cross street / milepost>"` (e.g. "E 094 MP 109"). The highway number decides the class:
  interstate set {39, 43, 90, 94, 535, 794, 894}, US-highway set, other numbers = state highway, letter codes = county trunk highway.
  Rows with no highway number ("NA NA STH 13") fall back to the keyword rules, which recognize WI's `USH`/`STH`/`CTH` codes.
* **MT and VT** are free text ("I-90 MM 313 WB", "VT RT 103 & TREBO RD", "browns trace"). Ordered keyword rules assign interstate → US
  highway → state highway → county road → local street. Two named roads with no highway marker ("PALMER AND GREAT NORTHERN") count
  as a town intersection.
* Mileposts (including "MM12", "MP-59"), intersections ("&", " AND ", "/", " X "), ramps/exits and the interstate number are extracted
  with regexes.

**`search_type_raw`**: keyword patterns map, for example, "CONSENT SEARCH CONDUCTED", "Consent Search-Rsnbl Suspicion" and "Consent" onto
`consent`, and "Terry Frisk" and "STOP AND FRISK (OFFICER SAFETY)" onto `frisk`. Multi-valued strings set several flags at once.

**`stop_outcome`**: 12 state-specific labels ("Arrest Driver", "Written Warning", "N/D"…) were collapsed into arrest / citation / warning /
no action / unknown. MT's comma-separated raw outcome was split into whole-token indicators.

**`violation_raw`**: keyword flags such as `\bFTA\b|FAILURE TO APPEAR|WARRANT` and `SUSPEND|REVOK|\bOAR\b`. The "obstruct" pattern was narrowed
to "obstructing an officer" after it matched WI equipment charges.

## 3. Training data selection

| Decision | Reason |
|---|---|
| **70/30 split stratified by state × `is_arrested`**, done *before* any resampling | Every state's size and arrest rate is identical in train (2,009,919) and test (861,394). No duplicated row can reach the test set. |
| **Test set left at its natural distribution** (2.45% arrests) | AP, F1 and ROC AUC must reflect the population the model will face. |
| **Bootstrap-balanced training set**: each of the 10 state × label cells resampled with replacement to 20,000 rows (200k total) | Balances the target (50/50) **and** the states (20% each instead of WI 36% / VT 10%), so large states do not drown out the others. |
| `n_per_cell = 20,000` | Keeps most of every arrest pool while limiting duplication. VT's 2,329 training arrests are each used 8.6×, the others 1.6–2.8×. |

![Bootstrap balance](docs/figures/15_bootstrap_balance.png)

**Consequences that are handled explicitly:**
* In the balanced set every state's arrest rate is 50%, so probabilities are **prior-corrected** back to each state's natural rate:
  `p' = p·r / (p·r + 1 − p)`, with `r = π_state / (1 − π_state)`.
* Bootstrap duplicates would leak across CV folds, so cross-validation is **grouped by source stop** (`StratifiedGroupKFold`).
* The F1 threshold is chosen on **out-of-fold** predictions reweighted to the natural state × label mix, never on the test set.
* A sanity check confirms feature means within each cell are unchanged by resampling.

## 4. Preprocessing per model

A `FrameCaster` turns the mixed pandas types into float numerics plus fixed-category categoricals learned on training data, so unseen
categories at inference become missing. The model uses **53 features: 47 numeric and 6 categorical** (`state`, `day_of_week`,
`geo_unit`, `police_department`, `road_class`, `interstate_id`).

| Model | Categorical | Numeric |
|---|---|---|
| Logistic regression | one-hot, with rare levels (< 50) pooled | median imputation + missing indicators, standardized |
| Random forest | ordinal codes, missing = −1 | missing → −1 |
| HistGradientBoosting, LightGBM | native categorical support | native missing-value support |

## 5. Feature sets and the leakage decision

Columns are grouped by **when the information becomes available**:

| Group | Count | Contents | Used? |
|---|---|---|---|
| `LEAK` | 18 | `stop_outcome(_raw)`, `outcome_std`, `outcome_raw_*`, search incident-to-arrest / inventory-tow, **plus** `vraw_warrant_fta`, `vraw_alcohol_dui`, `viol_dui`, `drugs_related_stop_clean`, `contraband_found_clean` | **Never** |
| `POST_STOP` | 17 | search conducted, search bases, `search_basis_known`, `n_search_bases`, `vraw_drugs`, `vraw_evade_resist`, `vraw_weapon`, `vraw_theft_stolen`, `viol_other` | Yes (`include_post_stop=True`) |
| `PRE_STOP` | 36 | state, time features, age, residency, routine violations, license-suspended / no-insurance / reckless, location and road features, vehicle age | Yes |
| `SENSITIVE` | 2 | `driver_gender`, `driver_race` | No (`include_sensitive=False`) |

Five post-stop features were moved to `LEAK` because, in at least one state, they almost always come with an arrest:

| Feature | Arrest rate when True |
|---|---|
| `vraw_warrant_fta` | 98.0% in NV (6,896 stops); RI 59% (117) |
| `vraw_alcohol_dui` / `viol_dui` | 84.5% in NV, 71.2% in WI, ~20% in MT and VT |
| `drugs_related_stop_clean` | 55.8% NV, 54.3% WI, 26.9% RI |
| `contraband_found_clean` | 51.4% WI, ~30% RI and VT |

The remaining post-stop features (e.g. any search: 26–35% arrest outside WI) are informative without being near-deterministic. They are
kept because the model scores **completed** stop records. To predict before a stop unfolds, use `include_post_stop=False`.

---

# Results

## 1. Model selection process

* **Candidates:** logistic regression (interpretable baseline), random forest (bagged trees), HistGradientBoosting and LightGBM (boosted trees).
* **Tuning:** `RandomizedSearchCV` with 6 / 6 / 8 / 10 configurations, each scored by **5-fold grouped CV** on the bootstrapped training set,
  refit on **average precision**. All 30 configurations, with every fold's score, are in the notebook and in `models/cv_results_all_configs.csv`.
* **Final validation** on the untouched natural test set (861,394 stops, 2.45% arrests).

![CV search](docs/figures/16_cv_search.png)

| Model | CV AP (± SD) | CV ROC AUC | CV F1 | CV accuracy | Train − CV AP gap |
|---|---|---|---|---|---|
| LightGBM | **0.9168** ± 0.0031 | **0.9099** | 0.8176 | **0.8274** | 0.056 |
| HistGradientBoosting | 0.9150 ± 0.0034 | 0.9092 | **0.8192** | 0.8257 | 0.036 |
| Random forest | 0.9072 ± 0.0031 | 0.8995 | 0.8070 | 0.8163 | 0.045 |
| Logistic regression | 0.8873 ± 0.0047 | 0.8842 | 0.7793 | 0.7924 | 0.002 |

*CV runs on 50/50 balanced folds, so these values are much higher than the natural-prevalence test values. They are for comparing configurations.*

## 2. Test-set performance

| Metric | **LightGBM** | HistGradientBoosting | Random forest | Logistic regression |
|---|---|---|---|---|
| **Average precision** | **0.476** | 0.471 | 0.446 | 0.319 |
| **ROC AUC** | **0.931** | 0.930 | 0.921 | 0.906 |
| **F1** (OOF-tuned threshold) | **0.497** | 0.486 | 0.475 | 0.412 |
| Precision / recall | 0.499 / 0.495 | 0.470 / 0.502 | 0.508 / 0.446 | 0.377 / 0.454 |
| Accuracy | 0.9754 | 0.9739 | **0.9758** | 0.9682 |
| Balanced accuracy | 0.741 | **0.744** | 0.718 | 0.717 |
| Brier score (lower = better) | **0.0167** | 0.0170 | 0.0176 | 0.0197 |
| Model file | 8.5 MB | 3.6 MB | 122.9 MB | 0.1 MB |

![Precision-recall and ROC on test](docs/figures/17_test_pr_roc.png)

## 3. The winner: LightGBM

**LightGBM is recommended.** Tuned settings: 600 trees, learning rate 0.037, 127 leaves, `min_child_samples` 20, row subsample 0.9,
column subsample 0.8, L2 0.54. F1 threshold 0.196.

**Why it wins**
1. **Best at the headline metric.** Test AP 0.476 is **19.4× the 0.0245 a random model scores**. At the F1-optimal threshold it finds about
   half of all arrests (recall 0.495), and about half of the stops it flags are real arrests (precision 0.499). That operating point can be
   moved along the precision-recall curve to suit the use case.
2. **Consistent across states.** It has the highest AP in all five states, and its weakest state (RI, 9.7× lift) beats every other model's weakest state.

   | State | Base rate | AP | Lift | ROC AUC |
   |---|---|---|---|---|
   | MT | 1.71% | 0.404 | 23.6× | 0.969 |
   | NV | 5.69% | 0.570 | 10.0× | 0.902 |
   | RI | 3.27% | 0.317 | 9.7× | 0.850 |
   | VT | 1.18% | 0.189 | 16.0× | 0.852 |
   | WI | 1.49% | 0.529 | 35.5× | 0.943 |

   Balancing the training set by state worked: VT, with only 2,329 training arrests, still gets a 16× lift.
3. **Best-calibrated probabilities** after prior correction (lowest Brier score and log loss), so its scores can be read as arrest probabilities.
4. **Its flexibility generalizes.** It has the largest train − CV gap (0.056), mainly from memorizing duplicated bootstrap rows. But its
   *grouped* CV score and its untouched-test score are both the best.
5. **Practical:** an 8.5 MB self-contained file that scores 861k stops in ~12 s.

![Test performance by state](docs/figures/18_test_by_state.png)

**Why not the others**
* **HistGradientBoosting** is a very close second (AP −0.005, AUC −0.002) and the **recommended fallback**. It overfits less, scores ~2× faster,
  is smaller, needs only scikit-learn, and has slightly higher recall and balanced accuracy. It trails LightGBM on AP, F1 and Brier, and on AP in every state.
* **Random forest** ranks well (AUC 0.921) and has the highest raw accuracy, but its AP trails by 0.03. The gap is largest in MT (0.349 vs 0.404),
  where boosting captures the county/corridor effects better. Its 123 MB file is impractical.
* **Logistic regression** barely overfits but trails on AP by 0.157. A single linear model cannot represent the state-specific feature effects
  found in the EDA, which confirms the non-linear models earn their complexity.

**Reading accuracy correctly:** every model's accuracy (96.8–97.6%) sits at or just below the **97.55% scored by always predicting "no
arrest"**. The F1 threshold accepts some false positives in exchange for catching half the arrests (the all-negative model catches none).
With a 2.45% positive class, accuracy says little about quality. AP, F1, ROC AUC and balanced accuracy drive the decision.

## 4. What drives the model

![LightGBM feature importance](docs/figures/19_lightgbm_importance.png)

* Top features by split gain: **location (`geo_unit`, 16%)**, whether a **search basis was recorded** (11%), the **"Other" violation** (7%),
  **driver age** (7%) and **speeding** (6%), which mostly *lowers* risk. These match the EDA: MT's county effect, the post-stop search signal,
  the hidden charges in "Other" and the age gradient.
* **Post-stop features supply 24% of the top-25 gain.**

## 5. Caveats and next steps

* **Search range edge:** the best LightGBM configuration sits at the edge of the search space (`num_leaves=127`, `min_child_samples=20`,
  `n_estimators=600`). Extending it, with early stopping on a grouped validation fold, may add a little AP.
* **Weakest states:** RI and VT (AUC ≈ 0.85). They record the fewest usable fields (RI has no county; VT has coarse violations), so more
  data is the likely lever rather than more tuning.
* **Use case and post-stop features:** the model includes post-stop information. Retrain with `include_post_stop=False` if predictions are
  needed at the start of a stop.
* **Fairness:** race and gender are not used, but location and vehicle features can act as proxies. A disparate-impact audit is advisable
  before any operational use.
* **Label noise:** RI counts passenger arrests, and some "incident to arrest" searches are not labeled as arrests, so the target is imperfect.
* **Consider state specific models** that can focus on the specific data engineering needs of that state and model tuning specific to those patterns. 
