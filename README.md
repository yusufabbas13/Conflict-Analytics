# Conflict-Analytics
 A machine learning pipeline that predicts **weekly conflict escalation** across countries using event-level data from [ACLED](https://acleddata.com/). The model identifies whether a country in a current week is likely to see a **surge in fatalities over the following 4 weeks**, enabling early warning signals for conflict analysts and policymakers

**Columns used:**
| Column | Description |
|---|---|
| `event_date` | Date of the event |
| `country` | Country where it occurred |
| `event_type` | Type of conflict event (Battles, Protests, etc.) |
| `interaction` | Actor types involved (State forces, Rebels, etc.) |
| `notes` | Free-text description of the event |
| `fatalities` | Number of fatalities recorded |

---

## Pipeline Summary
```
Raw ACLED CSV
    │
    ▼
Column Selection & Cleaning
    │
    ▼
One-Hot Encoding (event_type, actor interactions)
    │
    ▼
BERT Embeddings on event notes (all-MiniLM-L6-v2)
    │
    ▼
Weekly Aggregation per (country, week)
    │
    ▼
Feature Engineering (lags, ratios, diversity)
    │
    ▼
Target Variable: conflict_escalation (binary)
    │
    ▼
Temporal Train / Test Split (80/20 by time)
    │
    ▼
XGBoost Classifier
    │
    ▼
Evaluation + Risk Index (0–100)
```

---

## Features

### Structural Features
| Feature | Description |
|---|---|
| `fatalities` | Total fatalities in the country-week |
| `fatalities_lag2` | Sum of fatalities over the previous 2 weeks |
| `fatalities_lag4` | Sum of fatalities over the previous 4 weeks |
| `event_count` | Total number of events in the week |
| `violent_events` | Battles + Explosions + Violence against civilians |
| `violent_event_ratio` | Violent events as a fraction of total events |
| `state_actor_ratio` | State forces involvement relative to all actors |
| `event_diversity` | Number of distinct event types active in the week |

### One-Hot Features
- `event_type_*` — 6 event type flags (summed per week)
- `actor_*` — 8 actor type flags (summed per week)

### Semantic Features
- `bert_0` … `bert_383` — 384-dimensional mean BERT embedding of all event notes in the week, from `sentence-transformers/all-MiniLM-L6-v2`

### Target
| Feature | Description |
|---|---|
| `conflict_escalation` | `1` if next-4-week fatalities > 2× current week fatalities, else `0` |

---

## Model

**XGBoost Classifier**
```python
XGBClassifier(
    n_estimators=200,
    max_depth=6,
    learning_rate=0.05,
    eval_metric="logloss",
    random_state=42
)
```

**Train/Test Split:** Temporal — 80% of weeks for training, most recent 20% for testing. No shuffling. No future data seen during training.

---

## Results

Evaluated on the held-out **most recent 20% of weeks**:

| Metric | Score |
|---|---|
| Accuracy | *79* |
| Precision | *79* |
| Recall | *78* |
| F1-Score | *78.5* |
| ROC-AUC | *86.23* |


---

## Visualizations

The project includes 9 visualizations across two notebooks/sections:

**Model & Feature Analysis**
- Top-20 feature importances
- Feature correlation heatmap (engineered features only)
- Fatalities vs risk index scatter plot (coloured by escalation label)

**Country-Level Conflict Insights**
- Top 20 countries by escalation rate
- Fatalities vs escalation rate bubble chart
- Monthly escalation trend (small multiples, top 6 countries)
- What drives escalation? — grouped bar comparison
- Global escalation rate over time (line chart)
- Risk index heatmap (top 20 countries × last 20 weeks)

---
