# WanderWell ML Recommendation System

A hybrid machine learning travel recommendation system that personalizes place suggestions based on traveler type and vibe preferences.

## Project Overview

WanderWell uses a **Hybrid ML Architecture (50% ML + 50% Vibe Match)** to recommend travel destinations. The system was developed as an MVP to demonstrate personalized recommendations with limited training data.

**Key Features:**
- Personalized recommendations based on traveler type (Solo, Couples, Friends, Family)
- Vibe-based filtering (Chill, Cultural, Adventure, Mixed)
- 77 engineered features from 332 reviews across 28 places
- Pre-computed recommendation scenarios for fast lookup

## Repository Structure

```
AIPI590/
├── raw/                          # Raw data files
│   ├── Places.xlsx               # Place attributes (28 places)
│   └── Reviews.xlsx              # User reviews (332 reviews)
├── output/                       # Generated outputs
│   ├── WanderWell_ML_All_Scenarios.xlsx    # Pre-computed recommendations
│   ├── eda_visualizations.png              # EDA charts
│   ├── feature_engineering_analysis.png   # Feature analysis charts
│   └── model_evaluation.png               # Model comparison charts
├── wanderwell_ml_system.py       # Main ML system code
├── scenario_generator_ml.py      # Scenario generation script
├── wander-ai-plans-73-main.zip   # MVP Frontend (Lovable)
└── README.md
```

## 🔬 Methodology

### Data
- **332 reviews** across **28 places** (New York & Durham)
- **77 features** engineered:
  - Text Features (15): Sentiment indicators (mentions_beautiful, mentions_perfect, etc.)
  - TF-IDF Features (50): Place-specific important words
  - User Context (2): Traveler type encoding
  - Place Attributes (10): Cost, duration, vibe tags, time fit scores

### Model Selection
| Model | MAE | R² | Variance |
|-------|-----|-----|----------|
| Logistic Regression | 0.596 | -0.161 | ±0.178 |
| **Random Forest** | **0.621** | **0.156** | **±0.040** |
| XGBoost | 0.639 | 0.040 | ±0.039 |

**Random Forest** was selected for:
- Positive R² (0.156) - actually learns patterns beyond baseline
- Low variance (±0.040) - stable across 5-fold cross-validation
- Consistent performance across all folds

### Hybrid Architecture
```
Final Score = 0.5 × ML_Rating + 0.5 × Vibe_Match_Score
```

**Why Hybrid?**
- Pure ML with 332 reviews produced non-dynamic recommendations
- Small dataset insufficient for ML to learn vibe-based personalization
- Hybrid approach ensures personalization while leveraging ML quality patterns

## Usage

### Running the ML System
```python
python wanderwell_ml_system.py
```

### Generating Scenarios
```python
python scenario_generator_ml.py
```

## Results

- **MAE**: 0.621 stars (average prediction error)
- **R²**: 0.156 (explains 15.6% of variance)
- **Top Predictors**: is_cultural (5.1%), places_id (4.2%), lunch_fit (3.4%), avg_cost_usd (3.2%), mentions_fun (3.1%)

### Validation
Different vibes produce different recommendations:
- **Durham, Couples, Chill** → Sarah Duke Gardens, American Tobacco
- **Durham, Couples, Cultural** → Duke University, Nasher Museum

## Future Opportunities

1. **Data Collection**: Collect 1,000+ reviews (3x current) and track user vibe preferences
2. **A/B Testing**: Implement user feedback loops to validate real-world recommendation quality
3. **Expansion**: Add more cities and contextual features (weather, seasonality, crowd levels)

## Tech Stack

- **ML**: Python, scikit-learn, XGBoost, pandas, numpy
- **Visualization**: matplotlib, seaborn
- **Frontend (MVP)**: Lovable (see `wander-ai-plans-73-main.zip`)
- **Data Storage**: Excel-based for MVP simplicity

## 👤 Author

**Online Team 1**  
AIPI 590 - Duke University

## 📝 License

This project is for educational purposes as part of AIPI 590 coursework.
