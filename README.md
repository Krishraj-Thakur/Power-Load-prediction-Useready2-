# Power System Load Type Classification

Predicting the `Load_Type` (**Light_Load**, **Medium_Load**, **Maximum_Load**) of a steel-industry power system from 15-minute meter readings, using a time-based train/test split (last month = test set).

**Final result (test month, December 2018):** LightGBM, accuracy **0.991**, macro F1 **0.990**.

---

## 1. Problem and data

- **Source file:** `load_data.csv`, 35,041 rows, one reading every 15 minutes for all of 2018.
- **Target:** `Load_Type` (imbalanced: about 52% Light, 28% Medium, 21% Maximum).
- **Validation:** train on January to November (32,064 rows), test on December (2,976 rows). The split is by date, never shuffled, so the model is always evaluated on data from its future.
- **Metrics:** accuracy, and macro-averaged precision, recall and F1 (macro, because the classes are imbalanced and accuracy alone hides weak performance on the smaller classes).

| Column | Meaning |
|---|---|
| `Date_Time` | Timestamp of the reading |
| `Usage_kWh` | Energy used in the 15-minute interval |
| `Lagging_Current_Reactive.Power_kVarh` | Lagging reactive power |
| `Leading_Current_Reactive_Power_kVarh` | Leading reactive power |
| `CO2(tCO2)` | Emissions |
| `Lagging/Leading_Current_Power_Factor` | Percentage of supplied power that is useful. Derived from usage and reactive power. |
| `NSM` | Number of seconds from midnight |
| `Load_Type` | Target label |

---

## 2. Data quality: the dataset is deliberately noisy

| Issue | Finding | Action |
|---|---|---|
| Duplicate timestamp | One timestamp (2018-12-31 20:00) appeared twice | Kept the copy with fewer missing values (35,040 rows = 365 × 96) |
| `NSM` corrupted | 1,773 values above 86,400 s; 3,414 values that do not match the timestamp | Dropped; rebuilt time of day cleanly from `Date_Time` |
| Power factor above 100% | 3,226 lagging and 2,685 leading values | Dropped both PF columns (see below) |
| Hidden corruption | Many values were wrong but plausible-looking | Detected with the long-decimal test below |
| Missing values | Every numeric column had gaps | Left as missing for tree models; median imputation (fit on train only) for the others |

**Power factor is a percentage and follows a formula:**
`PF = 100 × kWh / √(kWh² + kVarh²)`. The stored values matched this formula in about 85% of rows, so the rest contain a corrupted value in at least one of the three columns.

**The long-decimal test.** Genuine readings have at most 2 decimal places, while corrupted ones usually have many (for example `8.753692`). Checking this against the formula:

| | Formula agrees | Formula disagrees |
|---|---|---|
| Value has at most 2 decimals | 100% | 0% |
| Value has more than 2 decimals | 15.5% | 84.5% |

So every value with more than 2 decimals was set to missing. After that, the leading power factor matched the formula in 100% of the remaining rows. This raised missing values to about 14% for `Usage_kWh`, 10% for lagging reactive power, 6.5% for leading reactive power and 5% for CO2.

---

## 3. What drives `Load_Type`

Exploratory analysis showed that the label is mostly a **schedule**, not a measure of how much power was used:

- **Time of day:** hours 0 to 9 are always Light in every month.
- **Season:** April, July and October share the same Maximum-load window (about 10:15 to 16:xx). January is different: no Maximum from 13:00 to 16:00, but Maximum in the evening. The window boundaries fall on quarter-hours.
- **Usage alone cannot separate Medium from Maximum.** Their usage ranges overlap heavily.
- **Weekends and holidays are decided per day.** Across 104 weekend days (10:00 to 22:59), every day was either entirely Light (49 days) or contained no Light readings (55 days). Thirteen weekdays were also entirely Light and fall on public-holiday or shutdown dates (for example Jan 1, May 1, Dec 25, Dec 31, and blocks around Feb 14 to 16, Aug 1 to 3 and Sep 24 to 26).
- On weekend days that are entirely Light, daily median usage is very low and tight (2.7 to 5.0 kWh). On the other weekend days it ranges from 3.5 to 76.2 kWh.

---

## 4. Features

**Raw columns kept:** `Usage_kWh`, lagging reactive power, leading reactive power, `CO2(tCO2)`.
**Dropped:** `NSM` (corrupted), both power-factor columns (redundant), `Date_Time` (replaced by the features below).

| Feature | Description |
|---|---|
| `minutes` | Minutes since midnight (0 to 1425), so trees can split on exact quarter-hours |
| `tod_sin`, `tod_cos` | Time of day on a circle, so 23:45 and 00:00 are close |
| `dayofweek`, `is_weekend` | Weekday information |
| `month_sin`, `month_cos` | Season on a circle, so December sits next to January |
| `apparent_power` | √(usage² + lagging reactive²) |
| `total_reactive` | Lagging plus leading reactive power |
| `day_median_usage`, `day_max_usage` | Median and maximum usage of the calendar day a row belongs to |

---

## 5. Models and results: from 0.75 to 0.99

Models compared: Logistic Regression, Random Forest, XGBoost, LightGBM, and a probability-averaging blend of the three tree models.

### Stage 1: first attempt (accuracy about 0.75)

The first feature set had the raw energy columns plus time-of-day, weekday and weekend features, but **no seasonal features**. Every model plateaued at about 0.75 accuracy and 0.64 macro F1. Light was easy (F1 about 0.95), while Medium and Maximum were confused with each other (F1 about 0.47 to 0.51).

| Model | Accuracy | Macro F1 |
|---|---|---|
| Random Forest | 0.7534 | about 0.64 |
| LightGBM | 0.7453 | 0.6406 |
| XGBoost | 0.7436 | 0.6314 |

Switching to gradient boosting, and leaving gaps unfilled, did not help. The cause was missing information, not the algorithm.

### Stage 2: finding the cause with an ablation

Removing one feature group at a time from a Random Forest (macro F1 on the test month):

| Feature set | Macro F1 |
|---|---|
| All features | 0.9414 |
| Without month features | **0.6265** |
| Without hour features | 0.8628 |
| Without `NSM` | 0.8937 |
| Without engineered energy features | 0.9375 |
| Without power-factor features | 0.9474 |

The seasonal (month) features were responsible for the entire gap, and the power-factor features were not needed (the score was slightly better without them). The month features still help for December even though December is never in the training data, because the circular encoding places it next to November and January.

### Stage 3: seasonal features and minute-level time (accuracy about 0.95)

| Model | Accuracy | Precision | Recall | Macro F1 |
|---|---|---|---|---|
| Random Forest | 0.9553 | 0.9396 | 0.9632 | 0.9506 |
| XGBoost | 0.9499 | 0.9320 | 0.9616 | 0.9456 |
| Blend (RF+XGB+LGBM) | 0.9483 | 0.9298 | 0.9615 | 0.9443 |
| LightGBM | 0.9422 | 0.9215 | 0.9596 | 0.9385 |
| Logistic Regression | 0.6109 | 0.5207 | 0.5126 | 0.5099 |

### Error analysis, and the next fix

The remaining errors were concentrated in the evening hours (18 to 22), around noon, and on Saturdays, Christmas Day and New Year's Eve. These are the days where the label depends on whether the factory ran, which a single 15-minute reading cannot reveal. The fix was to give each row a summary of its own day: `day_median_usage` and `day_max_usage`.

### Stage 4: day-level usage features (final, accuracy about 0.99)

| Model | Accuracy | Precision | Recall | Macro F1 |
|---|---|---|---|---|
| **LightGBM** | **0.9906** | 0.9890 | 0.9917 | **0.9903** |
| Blend (RF+XGB+LGBM) | 0.9896 | 0.9867 | 0.9914 | 0.9890 |
| XGBoost | 0.9862 | 0.9804 | 0.9899 | 0.9850 |
| Random Forest | 0.9856 | 0.9799 | 0.9888 | 0.9842 |
| Logistic Regression | 0.7201 | 0.5728 | 0.5654 | 0.5656 |

Testing the two day-level features separately (LightGBM, macro F1):

| Day-level features | Macro F1 |
|---|---|
| None | 0.9385 |
| Median only | 0.8536 |
| Max only | 0.9797 |
| Both | **0.9903** |

The **max** does most of the work. The median on its own is *worse* than having no day-level feature. Of the 144 days with a median usage below 5 kWh (which looks idle), 84 actually contained busy labels, because the factory ran for less than half of the day. The max catches those days (typical max of 75.5 kWh on them, against 4.3 kWh on truly idle days). The median adds a little on top of the max, which is why using both is best.

### Final error breakdown

LightGBM makes about 28 errors on 2,976 test readings, and all of them fall on two Saturdays: **Dec 1** (10 errors, day median usage 4.72) and **Dec 29** (18 errors, an all-Light day with median usage 3.69). Both have median usage inside the 3.5 to 5.0 range shared by idle and busy weekend days, and both have 10 to 16 missing usage readings, so they are ambiguous from usage alone.


## 7. Limitations

- **One test month.** Scores reflect December only, and December has a slightly different class mix from the rest of the year (58.6% Light against about 51% in training).
- **Differences between the tree models are small** (a few hundredths of a point on 2,976 highly similar rows) and should not be read as a reliable ranking.
- **A few corrupted values may remain.** Some days have a single very large usage value that survived the long-decimal filter (for example one otherwise-idle day with a maximum of 104.7 kWh), and the max feature is sensitive to such spikes.
- **No hyperparameter tuning** was done beyond fixed reasonable settings.

---

## 8. Repository contents and how to run

```
.
├── load_data.csv
├── Load_prediction.ipynb
└── README.md
```

```bash
pip install pandas numpy matplotlib seaborn scikit-learn xgboost lightgbm
jupyter notebook Load_prediction.ipynb
```
