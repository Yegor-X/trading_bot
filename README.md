# Social Media News-Driven Trading Bot

This project implements a **news-aware trading bot** that uses **machine learning** and **keyword detection** to analyze social media feeds and place trades for specific stocks.

## What It Does

- Collects **real-time social media news** from a simulated exchange (`optibook`).
- **Cleans and vectorizes** the news using **TF-IDF**.
- Trains a separate **Random Forest classifier** per stock to predict if the news will significantly impact that stock.
- Detects **keywords** related to specific companies for manual override if the ML model isn't confident.
- Places **limit orders** based on calculated theoretical prices and position size.

---


## How to Run

### 1. Prepare Data

Ensure you have a `training.csv` file with the following columns:

- `SocialMediaFeed`: Text of the news post
- One column per stock (e.g., `NVDA`, `ING`, `SAN`, `PFE`, `CSCO`) with numeric impact values

### 2. Install Dependencies

```bash
pip install -r requirements.txt
```

---

## Components

### 1. `train_optimized_classifier()`

Trains one **binary classifier per stock** using a labeled `training.csv` file. Each classifier predicts whether a piece of social media text is likely to cause a large market impact (top 20% of impact values).

- Uses `TF-IDF` for text vectorization
- Trains using `RandomForestClassifier`
- Applies thresholding to select only strong movements as targets

### 2. `improved_keyword_detection(text, stock)`

Uses **regex rules** to manually match stock-specific keywords (e.g., "NVIDIA", "GeForce", "Pfizer", etc.). Acts as a fallback or confirmation to ML predictions.

### 3. `predict_news_impact_improved(...)`

Given a news text:
- Applies vectorization and classification for each stock
- Combines **keyword match** and **ML confidence**
- Returns a list of affected stocks with confidence score and reason ("ml" or "keyword")

---

## `HackathonTradingBot` Class

Main bot class. Handles connection, logic, and trading.

### Key Parameters

- `QUOTED_VOLUME`: max order volume per tick
- `FIXED_CREDIT`: price spread added to improve chance of order execution
- `POSITION_LIMIT`: maximum position per instrument
- `TARGET_POSITION`: position size after which aggression is reduced
- `CONFIDENCE_THRESHOLD`: ML confidence required to react to news
- `PAUSE_DURATION`: cooldown after reacting to news

### Key Methods

- `setup()`: Connects to the exchange, loads instruments, trains ML models
- `check_for_news()`: Fetches new news and reacts if a stock is impacted
- `execute_trading_logic()`: Places limit orders based on theoretical price and position
- `check_position_limits()`: Stops trading if global position is too large
- `run()`: Main infinite loop (news checking + trading)
