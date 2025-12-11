# Grow Irish Performance Analytics - Setup Instructions

## Quick Start

### 1. Install Python Dependencies
```bash
pip install -r requirements.txt
```

### 2. Run the App
```bash
streamlit run app.py
```

The app will open at `http://localhost:8501`

---

## Folder Structure

```
Grow_Irish_App/
├── app.py                          # Main Streamlit entry point
├── requirements.txt                # Python dependencies
├── utils.py                        # Core utility functions
├── mp_intensity_pipeline.py        # Data aggregation engine
├── coach_metrics_engine.py         # Coach insights calculation
├── intensity_utils.py              # Intensity computation
│
├── pages/                          # Multi-page app pages
│   ├── 1_Home.py                  # Data loading & startup
│   ├── 2_Sessions.py              # Session explorer
│   ├── 3_Players.py               # Player analysis
│   ├── 4_Configuration.py         # Settings & weights
│   └── 5_Documentation.py         # Help & reference
│
├── src/                            # Internal modules
│   ├── config.py                  # Configuration & window colors
│   ├── display_names.py           # Display name mapping
│   ├── intensity_classification.py # Intensity scoring
│   └── ui/
│       └── nav.py                 # Global navigation component
│
└── data/
    ├── full_players_df.csv        # Player database
    ├── session_summary.csv        # Session data
    ├── mdp_catalog.csv            # MDP reference
    └── player_mdp_profiles.csv    # Player MDP profiles
```

---

## Features

✅ **Coach View** - Quick snapshots and key metrics  
✅ **Analyst View** - Full metrics, charts, and exports  
✅ **Multi-Page App** - Home, Sessions, Players, Configuration, Documentation  
✅ **Intensity Analysis** - Window-based intensity classification  
✅ **Player Profiles** - Individual player performance metrics  
✅ **Data Export** - Download analysis results as CSV  

---

## System Requirements

- Python 3.8+
- 100 MB disk space (with data files)
- Modern web browser

---

## Troubleshooting

**App won't start?**  
→ Make sure all dependencies are installed: `pip install -r requirements.txt`

**Data not loading?**  
→ Ensure CSV files are in the same directory as app.py

**Import errors?**  
→ Verify you're running from the Grow_Irish_App directory

---

## First Time Using the App

1. Open the app on the **Home** page
2. Click "Load Default Data" to load example data
3. Navigate to other pages using the menu bar
4. Try Coach vs. Analyst view modes
5. Upload your own data when ready

---

For questions or issues, refer to the Documentation page within the app.
