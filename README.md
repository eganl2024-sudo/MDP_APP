# 🏃 Grow Irish Performance Analytics

**A GPS-based training load monitoring system for soccer teams.**

Built to help coaches and performance staff answer a simple question: *How hard are my players actually working?*
---

## The Problem

Raw GPS data gives you thousands of data points per session — speed, acceleration, heart rate, metabolic power — but no clear answer about session quality. Coaches need to know:
- Was today's session hard enough to drive adaptation?
- Which players are carrying heavy loads this week?
- Are we peaking at the right time before match day?

This app transforms raw tracking data into a single, interpretable **Session Intensity Index** that answers these questions at a glance.

---

## How It Works

### The Intensity Model

Session intensity isn't just about volume (how much) — it's about *how* the work was done. A player who logs 600m of high-speed running in short, explosive bursts is training differently than one who accumulates the same distance steadily.
The app calculates a composite **Session Intensity Index** using three components:
| Component | Default Weight | What It Captures |
|-----------|--------|------------------|
| **Explosiveness** | 30% | Peak power output in short windows (10s). Reflects sprint quality, acceleration demands, and match-like high-intensity moments. |
| **Repeatability** | 50% | Consistency of high-intensity efforts across the session. Can the player reproduce quality efforts, or do they fade? |
| **Volume** | 20% | Total accumulated metabolic load. The baseline "work done" regardless of intensity profile. |

These weights are configurable — shift toward explosiveness for speed-focused blocks, or toward volume for conditioning phases.

### Metabolic Power Demand (MDP)

Rather than relying on speed alone, the app uses **Metabolic Power** — a measure that accounts for both velocity and acceleration. A player accelerating from 0 to 15 km/h demands more energy than one cruising at 15 km/h, even though the speeds are identical.

MDP is calculated across rolling windows:

| Window | Use Case |
|--------|----------|
| **10 seconds** | Explosive capacity — sprints, direction changes, pressing actions |
| **20 seconds** | Sustained bursts — repeated sprint sequences, attacking runs |
| **30 seconds** | Extended high-intensity efforts — pressing traps, defensive recovery runs |

Peak values in each window tell you what the player *can* do. Comparing peaks across sessions reveals fitness trends and readiness.

### Z-Score Normalization

All metrics are converted to **z-scores** relative to the team's historical data. This means:

- A score of **0** = typical session for this team
- A score of **+1** = one standard deviation above typical (hard)
- A score of **-1** = one standard deviation below typical (light)

This makes sessions comparable across different team contexts and training phases.

> 📖 *Full mathematical methodology — including metabolic power derivation and z-score normalization formulas — is available in the app's Documentation page.*

---

## Features

### 📊 Session Explorer

Browse all sessions with filters for date range, players, and intensity threshold. Visualize how training load fluctuates across a week or mesocycle.

- **Team average intensity** with ±1 standard deviation bands
- **Session tagging** — automatically labels sessions as Light / Typical / Hard / Very Hard
- **Highlight detection** — surfaces top explosive efforts, biggest workloads, and sustained power outputs

### 👥 Player Analysis

Dive into individual player profiles with rolling window breakdowns. Compare a player's output to their own baseline and to team norms.

- Per-player intensity trends over time
- Percentile rankings within the squad
- Early / mid / late session fatigue patterns

### ⚙️ Configuration

Adjust the intensity model to match your training philosophy:

| Preset | Focus |
|--------|-------|
| **Match-like** | Balanced — prepares players for game demands |
| **Speed emphasis** | Prioritizes explosiveness for power development phases |
| **Conditioning** | Prioritizes volume and repeatability for fitness building |

### 🔄 Two View Modes

- **Coach View** — Headlines and key decisions. "Who worked hardest? Is anyone overloaded?"
- **Analyst View** — Full metric tables, charts, and CSV exports for deeper analysis.

---

## What I Learned

Building this app pushed me in directions I didn't expect.

**UX is harder than the math.** The intensity calculations came together relatively quickly, but making them *useful* for coaches who don't think in z-scores took real iteration. I learned that a good dashboard isn't about showing everything — it's about showing the right thing at the right moment. The Coach vs. Analyst toggle came directly from realizing that one interface couldn't serve both audiences.

**Methodology evolves with use.** We originally planned to use raw Metabolic Power as the core metric. But once I started building visualizations and testing with real session data, I realized MDP alone didn't capture the full picture. A player could have high average power but never hit a true peak. That insight led to the composite model — explosiveness, repeatability, volume — which tells a much richer story about how a session *felt*, not just how much work was done.

**Deployment is its own skill.** This was my first time taking an app from local development to a cloud server where others could actually use it. Learning to manage dependencies, environment variables, and configuration for production taught me that building something is only half the job — shipping it is the other half.

---

## Future Improvements

This project was built under time constraints, which shaped what I could include. Here's where I'd take it next:

**Longer time horizons.** Right now the app is strongest at session-level and short-term analysis. Coaches think in weeks, months, and seasons — I'd love to add views that show load accumulation over a mesocycle, weekly summaries, and season-long trends to support periodization planning.

**Real database and authentication.** The current CSV-based approach works for demos, but a production version would need proper user authentication and a database backend. This would enable multi-team support, persistent history, and role-based access (head coach vs. assistant vs. analyst).

**Direct integration with GPS providers.** The dream is to connect directly to platforms like Catapult or STATSports, pull session data automatically, and surface insights in near real-time. No more exporting CSVs — just finish training, open the app, and see how the session went.

---

## Quick Start
```bash
# Clone the repo
git clone https://github.com/eganl2024-sudo/MDP_APP.git
cd MDP_APP

# Install dependencies
pip install -r requirements.txt

# Run the app
streamlit run app.py
```

Open `http://localhost:8501` → Click **Load Default Data** → Explore.

---

## Tech Stack

- **Streamlit** — Interactive web interface
- **Pandas / NumPy** — Data processing and aggregation
- **Plotly** — Interactive visualizations
- **Scikit-learn / SciPy** — Statistical analysis and z-score normalization

---

## Project Structure
```
MDP_APP/
├── app.py                  # Main entry point
├── pages/                  # Multi-page app screens
│   ├── 1_Home.py           # Data loading
│   ├── 2_Sessions.py       # Session explorer
│   ├── 3_Players.py        # Player analysis
│   ├── 4_Configuration.py  # Settings
│   └── 5_Documentation.py  # Help & methodology
├── src/                    # Core modules
│   ├── config.py
│   ├── display_names.py
│   └── intensity_classification.py
├── utils.py                # Utility functions
├── mp_intensity_pipeline.py # Data aggregation engine
└── coach_metrics_engine.py  # Insight calculations
```

---

## Author

**Liam Egan**  
GitHub: [@eganl2024-sudo](https://github.com/eganl2024-sudo)

---

*Built for coaches who want data-driven answers without drowning in spreadsheets.*
