#!/usr/bin/env python3
# GG Moonshot v1d (MTF) — Binance.US (long-only), Console
# - Scans many /USDT pairs with chunking & sleeps
# - Max 2 concurrent positions (bot-managed)
# - ENTRY:
#     * 1h: "Moonshot" breakout (close > swing-high by BREAKOUT_PCT and vol spike) — keeps Fib targets
#     * 1m/5m/15m/30m/2h/4h/1d: timeframe-specific rules (see rule engine)
# - EXIT:
#     * 1h: Fibonacci extension targets (1.0/-0.382/-0.618) + ATR trailing after TP1
#     * Others: R-multiple targets from ATR stop (TP1=1.0R, TP2=1.8R, TP3=2.6R) + trailing after TP1
# - UI: header with Session & Total PnL, wallet & bot tables
# - Persists keys, positions, stats, session; retries on time skew
# - LIVE by default; set GG_PAPER=1 to paper trade
import os, sys, time, math, csv, json
from pathlib import Path
from datetime import datetime, timezone
from statistics import mean

import ccxt
import pandas as pd
import numpy as np
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich import box

try:
    from zoneinfo import ZoneInfo
    LOCAL_TZ = ZoneInfo("America/Denver")
except Exception:
    LOCAL_TZ = None

APP = "GG Moonshot v1d — MTF"
BASE = Path(__file__).resolve().parent

# files
KEYS_FILE  = BASE / ".gg_keys.json"
POS_FILE   = BASE / "positions.json"
TRADES_CSV = BASE / "trades.csv"
STATS_FILE = BASE / "stats.json"
SESSION_FILE = BASE / "session.json"

# tunables
MAX_OPEN_POS = 2
DUST_USD = 1.0

BREAKOUT_PCT = float(os.environ.get("GG_BREAKOUT_PCT", "2.0"))
VOL_SPIKE    = float(os.environ.get("GG_VOL_SPIKE", "2.0"))
ATR_MULT     = float(os.environ.get("GG_ATR_MULT", "2.0"))  # trailing mult
RISK_FRAC    = float(os.environ.get("GG_RISK_FRAC", "0.18"))
SCAN_CAP     = int(os.environ.get("GG_SCAN_CAP", "0"))  # 0=all
CHUNK        = int(os.environ.get("GG_CHUNK", "10"))
CYCLE_SLEEP  = int(os.environ.get("GG_CYCLE_SLEEP", "30"))
PAPER        = (os.environ.get("GG_PAPER","0") == "1")

# MTF scan cadence (run heavier TFs less frequently to save API)
CADENCE = {
    "1m": 1,
    "5m": 1,
    "15m": 1,
    "30m": 2,
    "1h": 1,
    "2h": 2,
    "4h": 4,
    "1d": 8
}

console = Console()

# -------- utils ----------
def now_local():
    try:
        if LOCAL_TZ: return datetime.now(LOCAL_TZ).strftime("%Y-%m-%d %H:%M:%S %Z")
    except Exception: pass
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")

def beep():
    if os.name == "nt":
        try:
            import ctypes
            ctypes.windll.user32.MessageBeep(0xFFFFFFFF)
        except Exception:
            pass

def load_json(p: Path, default):
    try: return json.loads(p.read_text())
    except Exception: return default

def save_json(p: Path, obj):
    tmp = p.with_suffix(p.suffix + ".tmp")
    tmp.write_text(json.dumps(obj, indent=2))
    tmp.replace(p)

def safe_float(x, default=0.0):
    try: return float(x)
    except Exception: return default

def jget(d, *keys, default=None):
    cur = d
    for k in keys:
        if isinstance(cur, dict) and (k in cur):
            cur = cur[k]
        else:
            return default
    return cur

def ensure_trades_csv():
    if not TRADES_CSV.exists():
        with TRADES_CSV.open("w", newline="") as f:
            csv.writer(f).writerow(["ts","symbol","side","qty","price","reason","realized_pnl"])

# -------- exchange ----------
def read_keys():
    if KEYS_FILE.exists():
        return load_json(KEYS_FILE, {})
    console.print(Panel.fit("[bold]Enter Binance.US API Keys[/bold]\n(saved to .gg_keys.json)", style="cyan"))
    k = console.input("API Key: ").strip()
    s = console.input("API Secret: ").strip()
    obj = {"apiKey":k, "secret":s}
    save_json(KEYS_FILE, obj)
    return obj

def ex_connect(live=True):
    keys = read_keys()
    ex = ccxt.binanceus({
        "apiKey": keys["apiKey"],
        "secret": keys["secret"],
        "enableRateLimit": True,
        "timeout": 20000,
        "options": {
            "defaultType": "spot",
            "adjustForTimeDifference": True,
            "recvWindow": 10000
        }
    })
    try: ex.load_time_difference()
    except Exception: pass
    if not live:
        ex.set_sandbox_mode(True)
    return ex

def fetch_retry(func, *a, **kw):
    try:
        return func(*a, **kw)
    except ccxt.InvalidNonce:
        try:
            # if first arg is exchange instance
            if a and hasattr(a[0], "load_time_difference"):
                a[0].load_time_difference()
        except Exception:
            pass
        time.sleep(0.3)
        return func(*a, **kw)

# -------- markets / wallet ----------
def list_usdt_pairs(ex):
    mkts = ex.load_markets()
    pairs = [s for s,m in mkts.items() if m.get("quote")=="USDT" and m.get("spot")]
    pairs.sort()
    if SCAN_CAP>0:
        pairs = pairs[:SCAN_CAP]
    return pairs, mkts

def get_usdt_map(mkts):
    mp = {}
    for s,m in mkts.items():
        if m.get("quote")=="USDT" and m.get("spot"):
            mp[m["base"].upper()] = m["symbol"]
            if "baseId" in m:
                mp[str(m["baseId"]).upper()] = m["symbol"]
    return mp

def fetch_balance(ex):
    return fetch_retry(ex.fetch_balance)

def equity_est_usdt(ex, bal):
    total = 0.0
    try: tickers = fetch_retry(ex.fetch_tickers)
    except Exception: tickers = {}
    for ccy, amt in (bal.get("total") or {}).items():
        f = safe_float(amt, 0.0)
        if f<=0: continue
        if ccy in ("USDT","USD"):
            total += f
        else:
            sym = f"{ccy}/USDT"
            t = tickers.get(sym) or {}
            px = safe_float(t.get("last") or t.get("close"))
            if px>0: total += f*px
    return total

def backfill_avg_entry(ex, pair, lookback_ms=14*24*60*60*1000):
    since = ex.milliseconds() - lookback_ms
    try: fills = ex.fetch_my_trades(pair, since=since, limit=1000)
    except Exception: return None
    qty = 0.0; cost=0.0
    for t in fills:
        side = t.get("side")
        amount = safe_float(t.get("amount"))
        price  = safe_float(t.get("price") or (safe_float(t.get("cost"))/amount if amount else 0.0))
        fee = safe_float(jget(t, "fee","cost", default=0.0))
        if side == "buy":
            qty  += amount; cost += amount*price + fee
        elif side == "sell":
            qty  -= amount; cost -= amount*price
    if qty <= 0.0: return None
    return cost / max(qty, 1e-12)

def wallet_positions(ex, bal, usdt_map):
    positions = []
    totals = bal.get("total") or {}
    free   = bal.get("free") or {}
    last_cache = {}
    for asset, amt in totals.items():
        if not asset or asset.upper() in ("USDT","USD"): continue
        q = safe_float(amt, 0.0)
        if q<=0: continue
        pair = usdt_map.get(asset.upper())
        if not pair: continue
        if pair not in last_cache:
            try:
                t = fetch_retry(ex.fetch_ticker, pair)
                last_cache[pair] = safe_float(t.get("last") or t.get("close"))
            except Exception:
                last_cache[pair] = 0.0
        last = last_cache[pair]
        if last<=0: continue
        val = q*last
        if val < DUST_USD: continue
        avg = backfill_avg_entry(ex, pair)
        pnl = pnlp = None
        if avg and avg>0:
            pnl  = (last - avg) * q
            pnlp = (last/avg - 1.0) * 100.0
        positions.append({
            "pair": pair, "asset": asset.upper(), "qty": q, "last": last,
            "value_usd": val, "avg_entry": avg, "pnl": pnl, "pnl_pct": pnlp,
            "free": safe_float(free.get(asset, 0.0))
        })
    positions.sort(key=lambda r: r["value_usd"], reverse=True)
    return positions

# -------- indicators / OHLCV ----------
def ohlcv(ex, sym, tf="1h", limit=240):
    return fetch_retry(ex.fetch_ohlcv, sym, timeframe=tf, limit=limit)

def np_ema(arr, n):
    if len(arr) < n: return None
    alpha = 2/(n+1)
    out = np.empty_like(arr, dtype=float)
    out[:]=np.nan
    out[n-1] = np.nanmean(arr[:n])
    for i in range(n, len(arr)):
        prev = out[i-1] if i-1>=0 and not math.isnan(out[i-1]) else arr[i-1]
        out[i] = alpha*arr[i] + (1-alpha)*prev
    return out

def atr_from_df(df, n=14):
    h,l,c = df["high"].to_numpy(), df["low"].to_numpy(), df["close"].to_numpy()
    tr = np.maximum(h-l, np.maximum(np.abs(h-np.roll(c,1)), np.abs(l-np.roll(c,1))))
    tr[0] = h[0]-l[0]
    atr = pd.Series(tr).ewm(span=n, adjust=False).mean().to_numpy()
    return atr

def swing_hi_lo(df, lookback=120):
    if len(df) < lookback+2: return None, None
    sub = df.iloc[-lookback-2:-2]
    return float(sub["high"].max()), float(sub["low"].min())

# -------- strategy rules ----------
def rule_1h_moonshot(df):
    close = df["close"].to_numpy()
    vol   = df["volume"].to_numpy()
    if len(df) < 60: return None
    c  = float(close[-2])     # last closed
    v  = float(vol[-2])
    hi, lo = swing_hi_lo(df, 120)
    if not hi or not lo: return None
    vma20 = float(pd.Series(vol[-22:-2]).mean())
    if vma20 <= 0: return None
    break_ok = c >= hi*(1.0 + BREAKOUT_PCT/100.0)
    vol_ok   = v >= vma20*VOL_SPIKE
    if not (break_ok and vol_ok): return None
    rng = hi - lo
    strength = (c/hi - 1.0) * (v/max(vma20,1e-9)) * 100.0
    fib = {"tp1": hi, "tp2": hi + 0.382*rng, "tp3": hi + 0.618*rng}
    atr_vals = atr_from_df(df, 14)
    atr14 = float(atr_vals[-2]) if atr_vals is not None else 0.0
    return {"strength": strength, "atr": atr14, "targets": ("fib", fib)}

def rule_momentum_ema(df, fast=20, slow=50):
    close = df["close"].to_numpy()
    if len(close) < slow+5: return None
    ema_fast = np_ema(close, fast)
    ema_slow = np_ema(close, slow)
    if ema_fast is None or ema_slow is None: return None
    c = float(close[-2])
    ef = float(ema_fast[-2])
    es = float(ema_slow[-2])
    if math.isnan(ef) or math.isnan(es): return None
    if ef > es and c > es:
        mom = (c/es - 1.0) * 100.0
        atr_vals = atr_from_df(df, 14); a = float(atr_vals[-2]) if atr_vals is not None else 0.0
        return {"strength": mom, "atr": a, "targets": ("R", (1.0, 1.8, 2.6))}
    return None

def rule_donchian_break(df, n=20):
    if len(df) < n+5: return None
    highs = df["high"].to_numpy()
    close = df["close"].to_numpy()
    don_hi = float(pd.Series(highs[:-1]).rolling(n).max().iloc[-1])
    c = float(close[-2])
    if c > don_hi:
        mom = (c/don_hi - 1.0) * 100.0
        atr_vals = atr_from_df(df, 14); a = float(atr_vals[-2]) if atr_vals is not None else 0.0
        return {"strength": mom, "atr": a, "targets": ("R", (1.0, 1.8, 2.6))}
    return None

def rule_bb_squeeze_break(df, n=20, k=2.0):
    if len(df) < n+5: return None
    close = df["close"].to_numpy()
    s = pd.Series(close[:-1])
    ma = s.rolling(n).mean().iloc[-1]
    std = s.rolling(n).std().iloc[-1]
    if std is None or std==0: return None
    upper = ma + k*std
    c = float(close[-2])
    width = (k*std)/ma if ma else 0.0
    if c > upper and width < 0.05:
        strength = (c/upper - 1.0) * 100.0 + (0.05 - width)*100.0
        atr_vals = atr_from_df(df, 14); a = float(atr_vals[-2]) if atr_vals is not None else 0.0
        return {"strength": strength, "atr": a, "targets": ("R", (1.0, 1.8, 2.6))}
    return None

# which rule per timeframe
RULES = {
    "1m":  lambda df: rule_momentum_ema(df, 8, 21),
    "5m":  lambda df: rule_donchian_break(df, 20),
    "15m": lambda df: rule_momentum_ema(df, 20, 50),
    "30m": lambda df: rule_bb_squeeze_break(df, 20, 2.0),
    "1h":  lambda df: rule_1h_moonshot(df),
    "2h":  lambda df: rule_momentum_ema(df, 20, 50),
    "4h":  lambda df: rule_momentum_ema(df, 50, 200),
    "1d":  lambda df: rule_donchian_break(df, 55),
}

# -------- bot state ----------
def load_positions(): return load_json(POS_FILE, {})
def save_positions(p): save_json(POS_FILE, p)
def load_stats(): return load_json(STATS_FILE, {"total_realized":0.0, "wins":0, "losses":0, "trades":0})
def save_stats(s): save_json(STATS_FILE, s)
def load_session(): return load_json(SESSION_FILE, {"session_pnl":0.0})
def save_session(s): save_json(SESSION_FILE, s)

# -------- trading ----------
def last_price(ex, sym):
    t = fetch_retry(ex.fetch_ticker, sym)
    return safe_float(t.get("last") or t.get("close"))

def mkt_buy(ex, sym, usdt_alloc, mkts):
    px = last_price(ex, sym)
    m = mkts.get(sym, {})
    prec = m.get("precision",{}).get("amount", None)
    qty = usdt_alloc/px if px>0 else 0.0
    if prec is not None:
        qty = float(f"{qty:.{prec}f}")
    if qty <= 0: raise RuntimeError("qty too small")
    if PAPER:
        fill = px
    else:
        o = fetch_retry(ex.create_order, sym, "market", "buy", qty)
        fill = safe_float(o.get("average") or o.get("price") or px)
    beep(); return qty, fill

def mkt_sell(ex, sym, qty):
    px = last_price(ex, sym)
    if PAPER:
        fill = px
    else:
        o = fetch_retry(ex.create_order, sym, "market", "sell", qty)
        fill = safe_float(o.get("average") or o.get("price") or px)
    beep(); return fill

def record_trade(kind, sym, qty, price, reason, realized=None):
    ensure_trades_csv()
    with TRADES_CSV.open("a", newline="") as f:
        csv.writer(f).writerow([datetime.now(timezone.utc).isoformat(), sym, kind, f"{qty:.8f}", f"{price:.6f}", reason, "" if realized is None else f"{realized:.2f}"])

def update_unrl(ex, positions):
    for s,p in positions.items():
        lp = last_price(ex, s); p["last"] = lp
        p["pnl_usd"] = (lp - p["entry"]) * p["qty"]
        p["pnl_pct"] = (lp/p["entry"] - 1.0)*100.0 if p["entry"]>0 else 0.0

def try_exits(ex, positions):
    to_close = []
    for s,p in positions.items():
        lp = last_price(ex, s)
        # mark TP1 (for enabling trail)
        if p.get("tp_scheme") == "fib":
            if lp >= p["tp"]["tp1"]: p["tp1_hit"] = True
        else:
            if lp >= p["tp"]["tp1"]: p["tp1_hit"] = True

        # update trailing after TP1
        if p.get("tp1_hit"):
            rows = ohlcv(ex, s, p["tf"], 200)
            if rows:
                import pandas as pd
                df = pd.DataFrame(rows, columns=["ts","open","high","low","close","volume"])
                atrv = atr_from_df(df, 14)
                a = float(atrv[-2]) if atrv is not None and len(atrv)>2 else p.get("atr",0.0)
            else:
                a = p.get("atr", 0.0)
            trail = lp - ATR_MULT * a
            if p.get("stop") is None or trail > p["stop"]:
                p["stop"] = trail

        # stop exit
        stop = p.get("stop")
        if stop and lp <= stop:
            fill = mkt_sell(ex, s, p["qty"])
            realized = (fill - p["entry"]) * p["qty"]
            record_trade("SELL", s, p["qty"], fill, "STOP/TRAIL", realized)
            to_close.append((s, realized))
            continue

        # TP2/TP3 exits for R-based
        if p.get("tp_scheme") == "R":
            for key in ("tp2","tp3"):
                if not p.get(f"{key}_done") and lp >= p["tp"][key]:
                    qty = round(p["qty"]*0.5, 6) if key=="tp2" else round(p["qty"], 6)
                    if qty>0:
                        fill = mkt_sell(ex, s, qty)
                        realized = (fill - p["entry"]) * qty
                        record_trade("SELL", s, qty, fill, key.upper(), realized)
                        p["qty"] -= qty
                        p[f"{key}_done"] = True
                        if p["qty"] <= 0:
                            to_close.append((s, 0.0))

    return to_close

def rows_to_df(rows):
    import pandas as pd
    return pd.DataFrame(rows, columns=["ts","open","high","low","close","volume"])

def size_allocation(bal):
    usdt_free = safe_float(jget(bal,"free","USDT", default=0.0))
    usd_free  = safe_float(jget(bal,"free","USD",  default=0.0))
    return max(10.0, (usdt_free + usd_free) * RISK_FRAC)

def weakest(positions, ex):
    worst_s, worst = None, 0.0
    for s,p in positions.items():
        lp = last_price(ex, s)
        pnl = (lp - p["entry"]) * p["qty"]
        if (worst_s is None) or (pnl < worst):
            worst_s, worst = s, pnl
    return worst_s, worst

# -------- UI ----------
def header_panel(live, cycles, scanned, open_count, bal, equity, session_pnl, stats):
    usdt_free = safe_float(jget(bal,"free","USDT", default=0.0))
    usd_free  = safe_float(jget(bal,"free","USD",  default=0.0))
    total_pnl = float(stats.get("total_realized",0.0))
    wins, losses = int(stats.get("wins",0)), int(stats.get("losses",0))
    winrate = (wins / max(1, wins+losses) * 100.0)
    text = (
        f"[bold cyan]{APP}[/] — {'LIVE 🔒' if live and not PAPER else 'PAPER 🧪'}\n"
        f"[dim]{now_local()}[/dim] | Cycles: [bold]{cycles}[/] | Scanned: [bold]{scanned}[/] | Open (bot): [bold]{open_count}[/]\n"
        f"Buying Power (USDT+USD): [bold]{usdt_free+usd_free:,.2f}[/]   USDT Free: [bold]{usdt_free:,.2f}[/]   Total Equity: [bold]{equity:,.2f}[/]\n"
        f"Session PnL: [bold]{session_pnl:,.2f}[/]   Total PnL: [bold]{total_pnl:,.2f}[/]   WinRate (all-time): [bold]{winrate:.1f}%[/]"
    )
    return Panel(text, box=box.SQUARE, title="Status", style="bright_cyan")

def wallet_table(wallet_pos, bot_pos):
    botset = set(bot_pos.keys())
    t = Table(title="Open Positions (WALLET view)", box=box.MINIMAL_HEAVY_HEAD, expand=True, show_lines=True)
    t.add_column("Pair"); t.add_column("Qty", justify="right"); t.add_column("Last", justify="right")
    t.add_column("Value $", justify="right"); t.add_column("Avg Entry", justify="right"); t.add_column("PnL $", justify="right")
    t.add_column("PnL %", justify="right"); t.add_column("Managed", justify="center")
    for r in wallet_pos:
        managed = "BOT" if r["pair"] in botset else "—"
        pnl_txt = "-" if r["pnl"] is None else f"{r['pnl']:,.2f}"
        pct_txt = "-" if r["pnl_pct"] is None else f"{r['pnl_pct']:.2f}%"
        t.add_row(r["pair"].replace("/",""), f"{r['qty']:.6f}", f"{r['last']:.6f}", f"{r['value_usd']:,.2f}",
                  "-" if r["avg_entry"] is None else f"{r['avg_entry']:.6f}", pnl_txt, pct_txt, managed)
    if not wallet_pos:
        t.add_row("—","—","—","—","—","—","—","—")
    return t

def bot_table(bot_pos):
    t = Table(title="Bot-Managed Positions", box=box.MINIMAL, expand=True, show_lines=False)
    for c in ["Pair","TF","Qty","Entry","Last","PnL $","PnL %","TP1","TP2","TP3","Stop"]:
        t.add_column(c, justify="right" if c not in ("Pair","TF") else "left")
    for s,p in bot_pos.items():
        lp = p.get("last", p["entry"])
        pnl = (lp - p["entry"])*p["qty"]
        pnlp = (lp/p["entry"]-1.0)*100.0 if p["entry"]>0 else 0.0
        tp1, tp2, tp3 = p["tp"]["tp1"], p["tp"]["tp2"], p["tp"]["tp3"]
        t.add_row(s.replace("/",""), p["tf"], f"{p['qty']:.6f}", f"{p['entry']:.6f}", f"{lp:.6f}",
                  f"{pnl:,.2f}", f"{pnlp:.2f}%",
                  f"{tp1:.6f}", f"{tp2:.6f}", f"{tp3:.6f}", "-" if p.get("stop") is None else f"{p['stop']:.6f}")
    if not bot_pos:
        t.add_row("—","—","—","—","—","—","—","—","—","—","—")
    return t

# -------- main ----------
def main():
    live = not PAPER
    ex = ex_connect(live=live)
    positions = load_positions()
    stats = load_stats()
    session = load_session()
    cycles = 0

    while True:
        cycles += 1
        try:
            pairs, mkts = list_usdt_pairs(ex)
            bal = fetch_balance(ex)
            equity = equity_est_usdt(ex, bal)
            usdt_map = get_usdt_map(mkts)
            wallet_pos = wallet_positions(ex, bal, usdt_map)

            # exits & unrealized
            update_unrl(ex, positions)
            closes = try_exits(ex, positions)
            if closes:
                for s, realized in closes:
                    session["session_pnl"] = float(session.get("session_pnl",0.0)) + realized
                    stats["total_realized"] = float(stats.get("total_realized",0.0)) + realized
                    stats["trades"] = int(stats.get("trades",0)) + 1
                    if realized >= 0: stats["wins"] = int(stats.get("wins",0)) + 1
                    else: stats["losses"] = int(stats.get("losses",0)) + 1
                    positions.pop(s, None)
                save_session(session); save_stats(stats); save_positions(positions)

            # entries / rotation
            def consider_entry(sym, tf, sig):
                alloc = size_allocation(bal)
                qty, fill = mkt_buy(ex, sym, alloc, mkts)
                if sig["targets"][0] == "fib":
                    tp = sig["targets"][1]
                    stop = fill - ATR_MULT * (sig.get("atr", 0.0))
                    tp_scheme = "fib"
                else:  # R-multiples
                    R = ATR_MULT * (sig.get("atr", 0.0))
                    if R <= 0: return
                    tp = {"tp1": fill + 1.0*R, "tp2": fill + 1.8*R, "tp3": fill + 2.6*R}
                    stop = fill - 1.0*R
                    tp_scheme = "R"
                positions[sym] = {
                    "tf": tf, "qty": qty, "entry": fill, "atr": sig.get("atr",0.0),
                    "tp": tp, "tp_scheme": tp_scheme, "tp1_hit": False, "stop": stop, "last": fill
                }
                reason = "MTF "+tf+" "+("FIB" if tp_scheme=="fib" else "R")
                record_trade("BUY", sym, qty, fill, reason)

            if len(positions) < MAX_OPEN_POS or MAX_OPEN_POS == 0:
                signals = []
                for tf, mod in CADENCE.items():
                    if cycles % mod != 0: 
                        continue
                    for i in range(0, len(pairs), CHUNK):
                        batch = pairs[i:i+CHUNK]
                        for s in batch:
                            if s in positions: continue
                            rows = ohlcv(ex, s, tf, 240 if tf!="1m" else 500)
                            if not rows: continue
                            df = rows_to_df(rows)
                            rule = RULES.get(tf)
                            if not rule: continue
                            sig = rule(df)
                            if sig:
                                sig["pair"] = s; sig["tf"] = tf; sig["strength"] = sig.get("strength", 0.0)
                                signals.append(sig)
                        time.sleep(max(ex.rateLimit, 250)/1000.0 * 1.2)
                # pick strongest overall
                signals.sort(key=lambda x: -x.get("strength", 0.0))
                for sig in signals:
                    if len(positions) >= MAX_OPEN_POS: break
                    consider_entry(sig["pair"], sig["tf"], sig)
                save_positions(positions)
            else:
                # rotation
                best = None
                for tf, mod in CADENCE.items():
                    if cycles % mod != 0: continue
                    for i in range(0, len(pairs), CHUNK):
                        batch = pairs[i:i+CHUNK]
                        for s in batch:
                            if s in positions: continue
                            rows = ohlcv(ex, s, tf, 240 if tf!="1m" else 500)
                            if not rows: continue
                            df = rows_to_df(rows)
                            rule = RULES.get(tf)
                            if not rule: continue
                            sig = rule(df)
                            if sig and (best is None or sig["strength"] > best["strength"]):
                                sig["pair"] = s; sig["tf"] = tf; best = sig
                        time.sleep(max(ex.rateLimit,250)/1000.0 * 1.2)
                if best:
                    worst_s, worst_pnl = weakest(positions, ex)
                    if worst_s is not None and worst_pnl < 0:
                        fill = mkt_sell(ex, worst_s, positions[worst_s]["qty"])
                        realized = (fill - positions[worst_s]["entry"]) * positions[worst_s]["qty"]
                        record_trade("SELL", worst_s, positions[worst_s]["qty"], fill, "ROTATE", realized)
                        session["session_pnl"] = float(session.get("session_pnl",0.0)) + realized
                        stats["total_realized"] = float(stats.get("total_realized",0.0)) + realized
                        stats["trades"] = int(stats.get("trades",0)) + 1
                        if realized >= 0: stats["wins"] = int(stats.get("wins",0)) + 1
                        else: stats["losses"] = int(stats.get("losses",0)) + 1
                        positions.pop(worst_s, None)
                        consider_entry(best["pair"], best["tf"], best)
                        save_session(session); save_stats(stats); save_positions(positions)

            # UI
            console.clear()
            console.print(header_panel(live, cycles, len(pairs), len(positions), bal, equity, float(session.get("session_pnl",0.0)), stats))
            console.print(wallet_table(wallet_pos, positions))
            console.print(bot_table(positions))
            console.print(Panel(f"Ctrl+C to stop. Next scan in {CYCLE_SLEEP}s…", style="dim"))
            time.sleep(CYCLE_SLEEP)

        except KeyboardInterrupt:
            console.print("\n[bold yellow]Stopped by user.[/]")
            save_positions(positions)
            break
        except ccxt.InvalidNonce:
            try: ex.load_time_difference()
            except Exception: pass
            time.sleep(1.0)
        except Exception as e:
            console.print(f"[red]Error:[/] {e}")
            time.sleep(5)

if __name__ == "__main__":
    main()
