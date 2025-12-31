# GG Moonshot v1d — Multi-Timeframe Trading Bot

An advanced multi-timeframe cryptocurrency trading bot with Fibonacci targets, ATR-based trailing stops, and intelligent position management.

## Overview

GG Moonshot is a sophisticated trading system that simultaneously monitors **8 different timeframes** (1m, 5m, 15m, 30m, 1h, 2h, 4h, 1d) to identify optimal entry and exit points across multiple cryptocurrency pairs on Binance.US.

### Key Features

- **Multi-Timeframe Analysis**: Scans 8 timeframes with optimized cadence
- **Dual Exit Strategies**:
  - 1h: Fibonacci extension targets with ATR trailing
  - Other TFs: R-multiple exits with ATR-based stops
- **Smart Position Management**: Max 2 concurrent positions with rotation
- **Rich Console UI**: Live wallet and position tracking
- **Persistent State**: Saves positions, trades, stats across restarts
- **Paper & Live Modes**: Safe testing before real trading
- **Risk Management**: Configurable risk fraction per trade

## Trading Strategy

### Entry Logic

**1-Hour Timeframe (Moonshot)**:
- Breakout detection: Close > swing high by configurable %
- Volume spike confirmation
- Fibonacci-based position sizing

**Other Timeframes** (1m/5m/15m/30m/2h/4h/1d):
- Timeframe-specific entry rules
- Momentum and volatility filters
- Adaptive to market conditions

### Exit Logic

**1-Hour Timeframe**:
- TP1: 1.0 Fibonacci extension
- TP2: -0.382 Fibonacci retracement
- TP3: -0.618 Fibonacci retracement
- ATR trailing stop after TP1

**Other Timeframes**:
- TP1: 1.0R (1x ATR from entry)
- TP2: 1.8R
- TP3: 2.6R
- ATR trailing stop after TP1

### Position Management

- **Maximum Positions**: 2 concurrent (bot-managed)
- **Rotation**: Exits losing positions when stronger signals appear
- **Portfolio View**: Separate wallet table (all holdings) and bot table (managed positions)

## Installation

### Requirements

```bash
pip install ccxt pandas numpy rich python-dotenv
```

### Dependencies
- **ccxt**: Exchange connectivity
- **pandas**: Data analysis
- **numpy**: Numerical operations
- **rich**: Beautiful console UI
- **python-dotenv**: Environment configuration

## Configuration

### Environment Variables

Set these in your environment or `.env` file:

```bash
GG_PAPER=0                # 1 for paper trading, 0 for live (default: live)
GG_SCAN_CAP=0             # Max pairs to scan (0=all)
GG_CHUNK=10               # API request chunk size
GG_BREAKOUT_PCT=2.0       # Breakout threshold percentage
GG_VOL_SPIKE=2.0          # Volume spike multiplier
GG_ATR_MULT=2.0           # ATR trailing stop multiplier
GG_RISK_FRAC=0.18         # Risk fraction per trade (18%)
GG_CYCLE_SLEEP=30         # Seconds between scan cycles
GG_DUST=1.0               # Minimum position size in USD
```

### First Run Setup

1. **API Keys**: On first run, the bot will prompt for:
   - Binance.US API Key
   - Binance.US API Secret
   - Keys are encrypted and saved to `.gg_keys.json`

2. **Paper Mode Testing** (Recommended):
```bash
export GG_PAPER=1
python gg_moonshot_mtf.py
```

3. **Live Trading** (After Testing):
```bash
export GG_PAPER=0
python gg_moonshot_mtf.py
```

## Usage

### Windows

```batch
Run_Moonshot_MTF.bat
```

### Linux/Mac

```bash
python gg_moonshot_mtf.py
```

### Console Interface

The bot displays:

**Header**:
- Session P&L
- Total P&L
- Current time

**Wallet Table**:
- All spot holdings
- Current values
- Total portfolio value

**Bot Table**:
- Active positions (max 2)
- Entry prices
- Current P&L
- Exit targets
- Trailing stops

## Multi-Timeframe Scanning

The bot optimizes API usage by scanning timeframes at different cadences:

| Timeframe | Scan Frequency |
|-----------|----------------|
| 1m        | Every cycle    |
| 5m        | Every cycle    |
| 15m       | Every cycle    |
| 30m       | Every 2 cycles |
| 1h        | Every cycle    |
| 2h        | Every 2 cycles |
| 4h        | Every 4 cycles |
| 1d        | Every 8 cycles |

## File Structure

```
goldengoose-moonshot/
├── gg_moonshot_mtf.py        # Main bot script
├── Run_Moonshot_MTF.bat      # Windows launcher
├── .gg_keys.json             # Encrypted API keys (created on first run)
├── positions.json            # Active positions state
├── trades.csv                # Trade history log
├── stats.json                # Total P&L stats
└── session.json              # Session P&L tracking
```

## Risk Management

### Built-in Safeguards

- **Position Limits**: Max 2 concurrent positions
- **Risk Per Trade**: Configurable (default 18% of capital)
- **ATR-Based Stops**: Dynamic stop losses based on volatility
- **Trailing Stops**: Lock in profits as price moves favorably
- **Rotation Logic**: Cuts losing positions for better opportunities

### Recommended Settings

**Conservative**:
```bash
GG_RISK_FRAC=0.10    # 10% per trade
GG_ATR_MULT=2.5      # Wider stops
GG_BREAKOUT_PCT=3.0  # Higher breakout threshold
```

**Balanced** (Default):
```bash
GG_RISK_FRAC=0.18    # 18% per trade
GG_ATR_MULT=2.0      # Standard stops
GG_BREAKOUT_PCT=2.0  # Standard breakout
```

**Aggressive**:
```bash
GG_RISK_FRAC=0.25    # 25% per trade
GG_ATR_MULT=1.5      # Tighter stops
GG_BREAKOUT_PCT=1.5  # Lower breakout threshold
```

## Trade Logging

All trades are logged to `trades.csv` with:
- Timestamp
- Symbol
- Side (BUY/SELL)
- Amount
- Price
- P&L
- Exit reason

## Performance Tracking

### Session Stats
- Tracked in `session.json`
- Resets each time bot starts
- Shows current session P&L

### Total Stats
- Tracked in `stats.json`
- Persists across restarts
- Shows lifetime P&L

## Troubleshooting

### API Time Skew Errors
The bot automatically retries on time synchronization issues. If persistent:
- Sync your system clock
- Check network latency to Binance

### No Signals Detected
- Verify pairs are trading on Binance.US
- Check `GG_BREAKOUT_PCT` - may be too high
- Review `GG_VOL_SPIKE` threshold

### High API Usage
- Increase `GG_CYCLE_SLEEP`
- Reduce `GG_SCAN_CAP` to scan fewer pairs
- Increase `GG_CHUNK` size

## Advanced Features

### Position Rotation
If both position slots are filled and a new strong signal appears while one position is losing, the bot will:
1. Exit the losing position
2. Enter the new opportunity
3. Maintain max 2 positions

### Fibonacci Targets (1h only)
Based on swing high/low:
- 1.0 extension
- -0.382 retracement
- -0.618 retracement (Golden Ratio)

### R-Multiple Exits (Other TFs)
Risk-reward based targets:
- 1R = Break even after fees
- 1.8R = Profitable exit
- 2.6R = Extended profit target

## ⚠️ Important Warnings

- **Live Trading Risk**: Default mode is LIVE. Set `GG_PAPER=1` for safe testing
- **Capital Requirements**: Ensure sufficient capital for position sizing
- **API Limits**: Respect Binance.US rate limits
- **Monitoring**: Check bot regularly, don't leave unattended long-term
- **Testing**: Always paper trade first with new settings

## Updates & Improvements

This is version 1d with MTF (Multi-Timeframe) support. Key improvements:
- Multi-timeframe scanning
- Optimized API cadence
- Enhanced exit logic
- Better position rotation
- Improved UI

## License

Provided as-is for personal use and learning.

---

**GG Moonshot - Because good trading is golden** 🌙✨
