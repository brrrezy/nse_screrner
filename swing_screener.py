import argparse
import datetime as dt
import os
import tempfile
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import requests
import ta
import yfinance as yf
from tqdm import tqdm


# Use a writable cache location (Streamlit Cloud repo can be read-only)
_DEFAULT_CACHE_DIR = Path(tempfile.gettempdir()) / "nse_screener_cache"
CACHE_DIR = Path(os.environ.get("NSE_SCREENER_CACHE_DIR", str(_DEFAULT_CACHE_DIR)))
CACHE_DIR.mkdir(parents=True, exist_ok=True)
NSE_EQUITY_LIST_CACHE = CACHE_DIR / "EQUITY_L.csv"


# ============================================================
# 1) NSE UNIVERSE
# ============================================================


def get_nse_stocks(cache_ttl_hours: int = 24) -> List[str]:
    """
    Returns NSE equity symbols as Yahoo tickers (e.g., RELIANCE.NS).
    Uses NSE archive CSV with local cache.
    """
    url = "https://archives.nseindia.com/content/equities/EQUITY_L.csv"
    headers = {
        "User-Agent": "Mozilla/5.0",
        "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
        "Accept-Language": "en-US,en;q=0.5",
        "Connection": "keep-alive",
    }

    def cache_is_fresh() -> bool:
        if not NSE_EQUITY_LIST_CACHE.exists():
            return False
        age = dt.datetime.now() - dt.datetime.fromtimestamp(NSE_EQUITY_LIST_CACHE.stat().st_mtime)
        return age.total_seconds() < cache_ttl_hours * 3600

    if not cache_is_fresh():
        try:
            resp = requests.get(url, headers=headers, timeout=30)
            resp.raise_for_status()
            NSE_EQUITY_LIST_CACHE.write_text(resp.text, encoding="utf-8")
        except Exception:
            # If network fails, fall back to existing cache if present
            if not NSE_EQUITY_LIST_CACHE.exists():
                raise

    text = NSE_EQUITY_LIST_CACHE.read_text(encoding="utf-8", errors="ignore")
    lines = text.splitlines()
    if len(lines) < 2:
        return []

    symbols: List[str] = []
    for line in lines[1:]:
        if not line.strip():
            continue
        sym = line.split(",")[0].strip().strip('"')
        if sym:
            symbols.append(f"{sym}.NS")
    return symbols


# ============================================================
# 2) DATA NORMALIZATION (yfinance MultiIndex)
# ============================================================


def normalize_ohlcv_df(df: pd.DataFrame) -> pd.DataFrame:
    """
    yfinance may return MultiIndex columns even for a single ticker, e.g.
    ('Close','RELIANCE.NS'). Normalize to single-level columns: Open/High/Low/Close/Volume.
    """
    df = df.copy()

    if isinstance(df.columns, pd.MultiIndex):
        lvl0 = set(df.columns.get_level_values(0))
        lvl1 = set(df.columns.get_level_values(1))
        fields = {"Open", "High", "Low", "Close", "Adj Close", "Volume"}

        if lvl0.issubset(fields):
            df.columns = df.columns.get_level_values(0)
        elif lvl1.issubset(fields):
            df.columns = df.columns.get_level_values(1)
        else:
            df.columns = ["_".join(map(str, c)).strip() for c in df.columns.to_flat_index()]

    df.columns = [str(c) for c in df.columns]
    df = df.rename(columns={c: c.title() for c in df.columns})
    return df


# ============================================================
# 3) TECHNICAL ENGINE (Predicta V4-ish)
# ============================================================


def add_technical_indicators(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    df["ema8"] = ta.trend.ema_indicator(df["Close"], 8)
    df["ema21"] = ta.trend.ema_indicator(df["Close"], 21)
    df["ema55"] = ta.trend.ema_indicator(df["Close"], 55)
    df["ema144"] = ta.trend.ema_indicator(df["Close"], 144)
    df["ema50"] = ta.trend.ema_indicator(df["Close"], 50)
    df["ema200"] = ta.trend.ema_indicator(df["Close"], 200)

    df["rsi"] = ta.momentum.rsi(df["Close"], 14)

    macd = ta.trend.MACD(df["Close"])
    df["macd"] = macd.macd()
    df["macd_signal"] = macd.macd_signal()

    df["adx"] = ta.trend.adx(df["High"], df["Low"], df["Close"], 14)
    df["stoch"] = ta.momentum.stoch(df["High"], df["Low"], df["Close"], 14)

    df["vol_ma20"] = df["Volume"].rolling(20).mean()
    df["vol_mult"] = df["Volume"] / df["vol_ma20"]

    df["atr"] = ta.volatility.average_true_range(df["High"], df["Low"], df["Close"], 14)
    df["atr_ma20"] = df["atr"].rolling(20).mean()
    df["atr_high"] = df["atr"] > df["atr_ma20"]

    # Price + volume panel helpers
    df["adr20"] = (df["High"] - df["Low"]).rolling(20).mean()
    df["adrp20"] = (df["adr20"] / df["Close"]) * 100.0
    df["rvol50"] = df["Volume"] / df["Volume"].rolling(50).mean()

    # Delta proxy via CLV * volume
    hl_range = (df["High"] - df["Low"]).replace(0, np.nan)
    clv = ((df["Close"] - df["Low"]) - (df["High"] - df["Close"])) / hl_range
    df["clv"] = clv.fillna(0.0)
    df["delta_proxy"] = df["clv"] * df["Volume"]

    return df


def predicta_v4_confluence(latest: pd.Series) -> Tuple[int, Dict[str, bool]]:
    close = latest.get("Close", np.nan)
    ema50 = latest.get("ema50", np.nan)
    ema200 = latest.get("ema200", np.nan)
    if pd.isna(ema200):
        trend_ok = bool(close > ema50)
    else:
        trend_ok = bool(close > ema50 > ema200)

    signals = {
        "MACD": bool(latest.get("macd", np.nan) > latest.get("macd_signal", np.nan)),
        "RSI": bool(latest.get("rsi", np.nan) >= 55),
        "STOCH": bool(latest.get("stoch", np.nan) >= 60),
        "VOLUME": bool(latest.get("vol_mult", np.nan) >= 1.2),
        "DELTA": bool(latest.get("delta_proxy", 0.0) > 0),
        "TREND": trend_ok,
        "ADX": bool(latest.get("adx", np.nan) >= 20),
        "ATR": bool(bool(latest.get("atr_high", False))),
    }
    return int(sum(signals.values())), signals


def calculate_price_volume_metrics(df: pd.DataFrame) -> Dict[str, Any]:
    latest = df.iloc[-1]
    ema_distance = {
        "EMA8%": (latest["Close"] - latest["ema8"]) / latest["ema8"] * 100,
        "EMA21%": (latest["Close"] - latest["ema21"]) / latest["ema21"] * 100,
        "EMA55%": (latest["Close"] - latest["ema55"]) / latest["ema55"] * 100,
        "EMA144%": (latest["Close"] - latest["ema144"]) / latest["ema144"] * 100,
    }

    last30 = df.iloc[-30:]
    up_days = int((last30["Close"] > last30["Open"]).sum())
    down_days = int(len(last30) - up_days)

    high_52w = float(df["High"].tail(252).max()) if len(df) >= 252 else float(df["High"].max())
    dist_52w_high_pct = float((latest["Close"] / high_52w - 1.0) * 100.0) if high_52w else np.nan

    out: Dict[str, Any] = {}
    out.update(ema_distance)
    out.update(
        {
            "ADR": float(latest.get("adr20", np.nan)),
            "ADR%": float(latest.get("adrp20", np.nan)),
            "RVol": float(latest.get("rvol50", np.nan)),
            "U/D Days(30)": f"{up_days}/{down_days}",
            "Dist52WHigh%": dist_52w_high_pct,
        }
    )
    return out


# ============================================================
# 4) SETUPS (VCP / SFP / IPO BASE)
# ============================================================


def detect_vcp(df: pd.DataFrame, lookback: int = 60) -> Dict[str, Any]:
    if len(df) < lookback + 5:
        return {"VCP": False}

    w = df.tail(lookback).copy()
    w["range%"] = (w["High"] - w["Low"]) / w["Close"] * 100.0
    r1 = float(w["range%"].iloc[:20].mean())
    r2 = float(w["range%"].iloc[20:40].mean())
    r3 = float(w["range%"].iloc[40:60].mean())

    v1 = float(w["Volume"].iloc[:20].mean())
    v2 = float(w["Volume"].iloc[20:40].mean())
    v3 = float(w["Volume"].iloc[40:60].mean())

    return {"VCP": bool((r1 > r2 > r3) and (r3 < 6.0) and (v1 > v2 > v3))}


def detect_swing_failure(df: pd.DataFrame, lookback: int = 20) -> Dict[str, Any]:
    if len(df) < lookback + 2:
        return {"SFP": False}
    recent = df.iloc[-(lookback + 1) : -1]
    swing_low = float(recent["Low"].min())
    last = df.iloc[-1]
    sfp = bool((last["Low"] < swing_low) and (last["Close"] > swing_low) and (last["Close"] > last["Open"]))
    return {"SFP": sfp, "SFP_Level": swing_low}


def detect_ipo_base(symbol: str) -> Dict[str, Any]:
    try:
        info = yf.Ticker(symbol).info or {}
        first_trade = info.get("firstTradeDateEpochUtc")
        if not first_trade:
            return {"IPO_BASE": False, "IPO_Days": np.nan}
        ipo_date = dt.datetime.utcfromtimestamp(int(first_trade)).date()
        ipo_days = (dt.date.today() - ipo_date).days
        return {"IPO_BASE": bool(ipo_days <= 365), "IPO_Days": ipo_days}
    except Exception:
        return {"IPO_BASE": False, "IPO_Days": np.nan}


# ============================================================
# 5) FUNDAMENTALS (best-effort)
# ============================================================


def safe_get(df: Optional[pd.DataFrame], row_name: str, col_idx: int = 0) -> Optional[float]:
    if df is None or df.empty or row_name not in df.index:
        return None
    try:
        v = df.loc[row_name].iloc[col_idx]
        return None if pd.isna(v) else float(v)
    except Exception:
        return None


def compute_altman_z(info: Dict[str, Any], financials: Optional[pd.DataFrame], balance: Optional[pd.DataFrame]) -> Optional[float]:
    ta_ = safe_get(balance, "Total Assets", 0)
    tl_ = safe_get(balance, "Total Liab", 0) or safe_get(balance, "Total Liabilities Net Minority Interest", 0)
    ca_ = safe_get(balance, "Total Current Assets", 0)
    cl_ = safe_get(balance, "Total Current Liabilities", 0)
    re_ = safe_get(balance, "Retained Earnings", 0)
    ebit_ = safe_get(financials, "Ebit", 0) or safe_get(financials, "EBIT", 0)
    sales_ = safe_get(financials, "Total Revenue", 0)

    mcap = info.get("marketCap")
    try:
        mcap = float(mcap) if mcap is not None else None
    except Exception:
        mcap = None

    if any(x is None for x in [ta_, tl_, ca_, cl_, re_, ebit_, sales_, mcap]) or ta_ == 0 or tl_ == 0:
        return None
    wc = ca_ - cl_
    return float(1.2 * (wc / ta_) + 1.4 * (re_ / ta_) + 3.3 * (ebit_ / ta_) + 0.6 * (mcap / tl_) + 1.0 * (sales_ / ta_))


def compute_beneish_m(financials: Optional[pd.DataFrame], balance: Optional[pd.DataFrame]) -> Optional[float]:
    # Best-effort; often unavailable for NSE on Yahoo
    if financials is None or balance is None or financials.empty or balance.empty:
        return None
    if financials.shape[1] < 2 or balance.shape[1] < 2:
        return None

    sales_t = safe_get(financials, "Total Revenue", 0)
    sales_t1 = safe_get(financials, "Total Revenue", 1)
    cogs_t = safe_get(financials, "Cost Of Revenue", 0)
    cogs_t1 = safe_get(financials, "Cost Of Revenue", 1)
    ar_t = safe_get(balance, "Net Receivables", 0) or safe_get(balance, "Accounts Receivable", 0)
    ar_t1 = safe_get(balance, "Net Receivables", 1) or safe_get(balance, "Accounts Receivable", 1)
    ta_t = safe_get(balance, "Total Assets", 0)
    ta_t1 = safe_get(balance, "Total Assets", 1)
    ca_t = safe_get(balance, "Total Current Assets", 0)
    ca_t1 = safe_get(balance, "Total Current Assets", 1)
    ppe_t = safe_get(balance, "Property Plant Equipment", 0) or safe_get(balance, "Property Plant And Equipment Net", 0)
    ppe_t1 = safe_get(balance, "Property Plant Equipment", 1) or safe_get(balance, "Property Plant And Equipment Net", 1)
    cl_t = safe_get(balance, "Total Current Liabilities", 0)
    cl_t1 = safe_get(balance, "Total Current Liabilities", 1)
    ltd_t = safe_get(balance, "Long Term Debt", 0) or safe_get(balance, "Long Term Debt And Capital Lease Obligation", 0)
    ltd_t1 = safe_get(balance, "Long Term Debt", 1) or safe_get(balance, "Long Term Debt And Capital Lease Obligation", 1)
    dep_t = safe_get(financials, "Reconciled Depreciation", 0) or safe_get(financials, "Depreciation", 0)
    dep_t1 = safe_get(financials, "Reconciled Depreciation", 1) or safe_get(financials, "Depreciation", 1)
    sga_t = safe_get(financials, "Selling General Administrative", 0) or safe_get(financials, "Selling General And Administration", 0)
    sga_t1 = safe_get(financials, "Selling General Administrative", 1) or safe_get(financials, "Selling General And Administration", 1)

    if any(v is None for v in [sales_t, sales_t1, ta_t, ta_t1]) or sales_t == 0 or sales_t1 == 0:
        return None

    dsri = (ar_t / sales_t) / (ar_t1 / sales_t1) if all(v is not None and v != 0 for v in [ar_t, sales_t, ar_t1, sales_t1]) else None
    gmi = (
        ((sales_t1 - cogs_t1) / sales_t1) / ((sales_t - cogs_t) / sales_t)
        if all(v is not None and v != 0 for v in [sales_t, sales_t1, cogs_t, cogs_t1])
        else None
    )
    aqi = (
        (1 - ((ca_t + ppe_t) / ta_t)) / (1 - ((ca_t1 + ppe_t1) / ta_t1))
        if all(v is not None and v != 0 for v in [ca_t, ppe_t, ta_t, ca_t1, ppe_t1, ta_t1])
        else None
    )
    sgi = sales_t / sales_t1
    depi = (
        (dep_t1 / (dep_t1 + ppe_t1)) / (dep_t / (dep_t + ppe_t))
        if all(v is not None and v != 0 for v in [dep_t, dep_t1, ppe_t, ppe_t1]) and (dep_t + ppe_t) != 0 and (dep_t1 + ppe_t1) != 0
        else None
    )
    sgai = (
        (sga_t / sales_t) / (sga_t1 / sales_t1)
        if all(v is not None and v != 0 for v in [sga_t, sales_t, sga_t1, sales_t1])
        else None
    )
    lvgi = (
        ((cl_t + ltd_t) / ta_t) / ((cl_t1 + ltd_t1) / ta_t1)
        if all(v is not None and v != 0 for v in [cl_t, ltd_t, ta_t, cl_t1, ltd_t1, ta_t1])
        else None
    )

    needed = [dsri, gmi, aqi, sgi, depi, sgai, lvgi]
    if any(v is None for v in needed):
        return None

    tata = 0.0
    return float(-4.84 + 0.92 * dsri + 0.528 * gmi + 0.404 * aqi + 0.892 * sgi + 0.115 * depi - 0.172 * sgai + 4.679 * tata - 0.327 * lvgi)


def get_fundamentals(symbol: str) -> Dict[str, Any]:
    try:
        stock = yf.Ticker(symbol)
        info = stock.info or {}
        financials = getattr(stock, "financials", None)
        balance = getattr(stock, "balance_sheet", None)
        q_fin = getattr(stock, "quarterly_financials", None)

        out: Dict[str, Any] = {
            "PE": info.get("trailingPE"),
            "ROE": info.get("returnOnEquity"),
            "DebtToEquity": info.get("debtToEquity"),
            "ProfitMargin": info.get("profitMargins"),
            "RevenueGrowth": info.get("revenueGrowth"),
            "OperatingMargin": info.get("operatingMargins"),
            "EPS_Est_Growth%": (info.get("earningsGrowth") * 100.0) if info.get("earningsGrowth") is not None else None,
        }

        out["AltmanZ"] = compute_altman_z(info, financials, balance)
        out["BeneishM"] = compute_beneish_m(financials, balance)

        # Simple quality score (0-5)
        quality = 0
        if out.get("ROE") is not None and out["ROE"] >= 0.15:
            quality += 1
        if out.get("DebtToEquity") is not None and out["DebtToEquity"] <= 1.0:
            quality += 1
        if out.get("OperatingMargin") is not None and out["OperatingMargin"] >= 0.12:
            quality += 1
        if out.get("RevenueGrowth") is not None and out["RevenueGrowth"] >= 0.10:
            quality += 1
        if out.get("ProfitMargin") is not None and out["ProfitMargin"] >= 0.08:
            quality += 1
        out["FundamentalQualityScore"] = quality

        # QoQ / YoY revenue (last 6 quarters)
        if isinstance(q_fin, pd.DataFrame) and (not q_fin.empty) and ("Total Revenue" in q_fin.index):
            rev = q_fin.loc["Total Revenue"].astype(float).sort_index()
            qoq = rev.pct_change() * 100.0
            yoy = rev.pct_change(4) * 100.0
            for col in rev.index[-6:]:
                label = pd.to_datetime(col).date().isoformat() if hasattr(col, "date") else str(col)
                out[f"Rev_{label}"] = float(rev[col])
                out[f"Rev_QoQ%_{label}"] = float(qoq[col]) if not pd.isna(qoq[col]) else None
                out[f"Rev_YoY%_{label}"] = float(yoy[col]) if not pd.isna(yoy[col]) else None

        return out
    except Exception:
        return {"FundamentalQualityScore": 0}


# ============================================================
# 6) EXPORT
# ============================================================


def export_excel(top: pd.DataFrame, all_rows: pd.DataFrame, out_path: Path) -> None:
    try:
        with pd.ExcelWriter(out_path, engine="openpyxl") as writer:
            top.to_excel(writer, sheet_name="Top10", index=False)
            all_rows.to_excel(writer, sheet_name="AllScanned", index=False)
    except Exception:
        top.to_csv(out_path.with_suffix(".csv"), index=False)


# ============================================================
# 7) MAIN SCREENER (fallback always produces Top-N)
# ============================================================


def run_full_system(
    universe_limit: int = 500,
    min_confluence_score: int = 6,
    period: str = "2y",
    interval: str = "1d",
    top_n: int = 10,
    start_index: int = 0,
    fundamentals_top_k_multiplier: int = 5,
) -> pd.DataFrame:
    summary = {
        "universe_total": 0,
        "scanned": 0,
        "download_empty": 0,
        "missing_columns": 0,
        "too_short": 0,
        "errors": 0,
        "passed_filter": 0,
    }

    try:
        all_symbols = get_nse_stocks()
    except Exception as e:
        df = pd.DataFrame()
        df.attrs["error"] = f"Failed to fetch NSE universe: {type(e).__name__}: {e}"
        df.attrs["summary"] = summary
        return df

    summary["universe_total"] = len(all_symbols)
    start = max(0, int(start_index))
    if universe_limit and universe_limit > 0:
        symbols = all_symbols[start : start + universe_limit]
    else:
        symbols = all_symbols[start:]

    rows: List[Dict[str, Any]] = []

    for symbol in tqdm(symbols, disable=True):
        try:
            df = yf.download(
                symbol,
                period=period,
                interval=interval,
                progress=False,
                auto_adjust=False,
                threads=False,
                timeout=10,
            )
            if df is None or df.empty:
                summary["download_empty"] += 1
                continue

            df = normalize_ohlcv_df(df)
            needed = {"Open", "High", "Low", "Close", "Volume"}
            if not needed.issubset(df.columns):
                summary["missing_columns"] += 1
                continue

            # 90+ trading days supports VCP(60), RVOL(50), and most indicators.
            # EMA200 may be unavailable on short histories (handled in confluence logic).
            if len(df) < 90:
                summary["too_short"] += 1
                continue

            df = add_technical_indicators(df)
            latest = df.iloc[-1]
            confluence, conf_signals = predicta_v4_confluence(latest)
            passed = confluence >= min_confluence_score

            pv = calculate_price_volume_metrics(df)
            vcp = detect_vcp(df)
            sfp = detect_swing_failure(df)
            ipo = detect_ipo_base(symbol)

            setup_score = (2 if vcp.get("VCP") else 0) + (2 if sfp.get("SFP") else 0) + (1 if ipo.get("IPO_BASE") else 0)
            base_score = int(confluence + setup_score)

            rows.append(
                {
                    "Symbol": symbol,
                    "Score": base_score,
                    "ConfluenceScore": confluence,
                    "SetupScore": setup_score,
                    "PassedFilter": bool(passed),
                    "Price": float(latest["Close"]),
                    "RSI": float(latest.get("rsi", np.nan)),
                    "ADX": float(latest.get("adx", np.nan)),
                    "Stoch": float(latest.get("stoch", np.nan)),
                    "VolMult": float(latest.get("vol_mult", np.nan)),
                    "DeltaProxy": float(latest.get("delta_proxy", np.nan)),
                    **{f"C_{k}": v for k, v in conf_signals.items()},
                    **pv,
                    **vcp,
                    **sfp,
                    **ipo,
                }
            )
        except Exception:
            summary["errors"] += 1
            continue

    summary["scanned"] = len(rows)
    all_df = pd.DataFrame(rows)
    if all_df.empty:
        all_df.attrs["summary"] = summary
        return all_df

    passed_df = all_df[all_df["PassedFilter"]].copy()
    summary["passed_filter"] = int(len(passed_df))
    pool = passed_df if not passed_df.empty else all_df
    pool = pool.sort_values(by=["Score", "ConfluenceScore", "RVol"], ascending=False)

    top_k = min(len(pool), max(top_n * fundamentals_top_k_multiplier, top_n))
    pool_topk = pool.head(top_k).copy()

    fundamentals_rows: List[Dict[str, Any]] = []
    for sym in pool_topk["Symbol"].tolist():
        fundamentals_rows.append({"Symbol": sym, **get_fundamentals(sym)})
    fund_df = pd.DataFrame(fundamentals_rows).drop_duplicates(subset=["Symbol"])

    pool_topk = pool_topk.merge(fund_df, on="Symbol", how="left")
    pool_topk["FundamentalQualityScore"] = pool_topk.get("FundamentalQualityScore", 0).fillna(0).astype(int)
    pool_topk["Score"] = (pool_topk["ConfluenceScore"].astype(int) + pool_topk["SetupScore"].astype(int) + pool_topk["FundamentalQualityScore"]).astype(int)
    pool_topk = pool_topk.sort_values(by=["Score", "ConfluenceScore", "SetupScore", "RVol"], ascending=False)

    top = pool_topk.head(top_n).reset_index(drop=True)

    out_path = Path("Predicta_Top10.xlsx")
    export_excel(top=top, all_rows=all_df.sort_values(by=["Score", "ConfluenceScore"], ascending=False), out_path=out_path)

    top.attrs["summary"] = summary
    return top


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="NSE swing screener (confluence + setups + fundamentals)")
    p.add_argument("--universe-limit", type=int, default=500, help="How many NSE symbols to scan (0 = all)")
    p.add_argument("--start-index", type=int, default=0, help="Start index in NSE list (0-based)")
    p.add_argument("--min-score", type=int, default=6, help="Minimum confluence score (0-8). Fallback still returns Top-N.")
    p.add_argument("--period", type=str, default="2y", help="yfinance history period (e.g. 6mo, 1y, 2y)")
    p.add_argument("--interval", type=str, default="1d", help="yfinance interval (e.g. 1d)")
    p.add_argument("--top-n", type=int, default=10, help="Top N to export")
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    top_df = run_full_system(
        universe_limit=args.universe_limit,
        start_index=args.start_index,
        min_confluence_score=args.min_score,
        period=args.period,
        interval=args.interval,
        top_n=args.top_n,
    )
    summary = top_df.attrs.get("summary")
    if summary:
        print("Scan summary:", summary)
    if not top_df.empty:
        cols = [c for c in ["Symbol", "Score", "ConfluenceScore", "SetupScore", "FundamentalQualityScore", "Price", "RVol", "ADR%"] if c in top_df.columns]
        print(top_df[cols])
        print(f"Saved report to {Path('Predicta_Top10.xlsx').resolve()}")

# ---- Legacy duplicate code below (disabled) ----
'''
import argparse
import datetime as dt
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import requests
import ta
import yfinance as yf
from tqdm import tqdm


CACHE_DIR = Path(".cache")
CACHE_DIR.mkdir(exist_ok=True)
NSE_EQUITY_LIST_CACHE = CACHE_DIR / "EQUITY_L.csv"


# ============================================================
# 1) NSE UNIVERSE
# ============================================================


def get_nse_stocks(cache_ttl_hours: int = 24) -> List[str]:
    """Returns NSE symbols as Yahoo tickers (e.g., RELIANCE.NS)."""
    url = "https://archives.nseindia.com/content/equities/EQUITY_L.csv"
    headers = {
        "User-Agent": "Mozilla/5.0",
        "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
        "Accept-Language": "en-US,en;q=0.5",
        "Connection": "keep-alive",
    }

    def cache_is_fresh() -> bool:
        if not NSE_EQUITY_LIST_CACHE.exists():
            return False
        age = dt.datetime.now() - dt.datetime.fromtimestamp(NSE_EQUITY_LIST_CACHE.stat().st_mtime)
        return age.total_seconds() < cache_ttl_hours * 3600

    if not cache_is_fresh():
        resp = requests.get(url, headers=headers, timeout=30)
        resp.raise_for_status()
        NSE_EQUITY_LIST_CACHE.write_text(resp.text, encoding="utf-8")

    text = NSE_EQUITY_LIST_CACHE.read_text(encoding="utf-8", errors="ignore")
    lines = text.splitlines()
    if not lines:
        return []

    symbols: List[str] = []
    for line in lines[1:]:
        if not line.strip():
            continue
        sym = line.split(",")[0].strip().strip('"')
        if sym:
            symbols.append(f"{sym}.NS")
    return symbols


# ============================================================
# 2) TECHNICAL ENGINE (Predicta V4-ish)
# ============================================================


def add_technical_indicators(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    df["ema8"] = ta.trend.ema_indicator(df["Close"], 8)
    df["ema21"] = ta.trend.ema_indicator(df["Close"], 21)
    df["ema55"] = ta.trend.ema_indicator(df["Close"], 55)
    df["ema144"] = ta.trend.ema_indicator(df["Close"], 144)
    df["ema50"] = ta.trend.ema_indicator(df["Close"], 50)
    df["ema200"] = ta.trend.ema_indicator(df["Close"], 200)

    df["rsi"] = ta.momentum.rsi(df["Close"], 14)

    macd = ta.trend.MACD(df["Close"])
    df["macd"] = macd.macd()
    df["macd_signal"] = macd.macd_signal()

    df["adx"] = ta.trend.adx(df["High"], df["Low"], df["Close"], 14)
    df["stoch"] = ta.momentum.stoch(df["High"], df["Low"], df["Close"], 14)

    df["vol_ma20"] = df["Volume"].rolling(20).mean()
    df["vol_mult"] = df["Volume"] / df["vol_ma20"]

    df["atr"] = ta.volatility.average_true_range(df["High"], df["Low"], df["Close"], 14)
    df["atr_ma20"] = df["atr"].rolling(20).mean()
    df["atr_high"] = df["atr"] > df["atr_ma20"]

    # Price + Volume helpers
    df["adr20"] = (df["High"] - df["Low"]).rolling(20).mean()
    df["adrp20"] = (df["adr20"] / df["Close"]) * 100.0
    df["rvol50"] = df["Volume"] / df["Volume"].rolling(50).mean()

    # Daily delta proxy via CLV * volume
    hl_range = (df["High"] - df["Low"]).replace(0, np.nan)
    clv = ((df["Close"] - df["Low"]) - (df["High"] - df["Close"])) / hl_range
    df["clv"] = clv.fillna(0.0)
    df["delta_proxy"] = df["clv"] * df["Volume"]

    return df


def predicta_v4_confluence(latest: pd.Series) -> Tuple[int, Dict[str, bool]]:
    signals = {
        "MACD": bool(latest.get("macd", np.nan) > latest.get("macd_signal", np.nan)),
        "RSI": bool(latest.get("rsi", np.nan) >= 55),
        "STOCH": bool(latest.get("stoch", np.nan) >= 60),
        "VOLUME": bool(latest.get("vol_mult", np.nan) >= 1.2),
        "DELTA": bool(latest.get("delta_proxy", 0.0) > 0),
        "TREND": bool(latest.get("Close", np.nan) > latest.get("ema50", np.nan) > latest.get("ema200", np.nan)),
        "ADX": bool(latest.get("adx", np.nan) >= 20),
        "ATR": bool(bool(latest.get("atr_high", False))),
    }
    return int(sum(signals.values())), signals


# ============================================================
# 3) PRICE + VOLUME STRENGTH PANEL
# ============================================================


def calculate_price_volume_metrics(df: pd.DataFrame) -> Dict[str, Any]:
    latest = df.iloc[-1]
    ema_distance = {
        "EMA8%": (latest["Close"] - latest["ema8"]) / latest["ema8"] * 100,
        "EMA21%": (latest["Close"] - latest["ema21"]) / latest["ema21"] * 100,
        "EMA55%": (latest["Close"] - latest["ema55"]) / latest["ema55"] * 100,
        "EMA144%": (latest["Close"] - latest["ema144"]) / latest["ema144"] * 100,
    }

    last30 = df.iloc[-30:]
    up_days = int((last30["Close"] > last30["Open"]).sum())
    down_days = int(len(last30) - up_days)

    high_52w = float(df["High"].tail(252).max()) if len(df) >= 252 else float(df["High"].max())
    dist_52w_high_pct = float((latest["Close"] / high_52w - 1.0) * 100.0) if high_52w else np.nan

    out: Dict[str, Any] = {}
    out.update(ema_distance)
    out.update(
        {
            "ADR": float(latest.get("adr20", np.nan)),
            "ADR%": float(latest.get("adrp20", np.nan)),
            "RVol": float(latest.get("rvol50", np.nan)),
            "U/D Days(30)": f"{up_days}/{down_days}",
            "Dist52WHigh%": dist_52w_high_pct,
        }
    )
    return out


# ============================================================
# 4) SETUP DETECTORS (VCP / IPO BASE / SWING FAILURE)
# ============================================================


def detect_vcp(df: pd.DataFrame, lookback: int = 60) -> Dict[str, Any]:
    if len(df) < lookback + 5:
        return {"VCP": False, "VCP_Reason": "not_enough_data"}

    w = df.tail(lookback).copy()
    w["range%"] = (w["High"] - w["Low"]) / w["Close"] * 100.0
    r1 = float(w["range%"].iloc[:20].mean())
    r2 = float(w["range%"].iloc[20:40].mean())
    r3 = float(w["range%"].iloc[40:60].mean())

    v1 = float(w["Volume"].iloc[:20].mean())
    v2 = float(w["Volume"].iloc[20:40].mean())
    v3 = float(w["Volume"].iloc[40:60].mean())

    range_contracting = (r1 > r2 > r3) and (r3 < 6.0)
    volume_contracting = (v1 > v2 > v3)

    return {
        "VCP": bool(range_contracting and volume_contracting),
        "VCP_R1": r1,
        "VCP_R2": r2,
        "VCP_R3": r3,
    }


def detect_swing_failure(df: pd.DataFrame, lookback: int = 20) -> Dict[str, Any]:
    if len(df) < lookback + 2:
        return {"SFP": False, "SFP_Level": np.nan}

    recent = df.iloc[-(lookback + 1) : -1]
    swing_low = float(recent["Low"].min())
    last = df.iloc[-1]
    sfp = bool((last["Low"] < swing_low) and (last["Close"] > swing_low) and (last["Close"] > last["Open"]))
    return {"SFP": sfp, "SFP_Level": swing_low}


def detect_ipo_base(symbol: str) -> Dict[str, Any]:
    try:
        info = yf.Ticker(symbol).info or {}
        first_trade = info.get("firstTradeDateEpochUtc")
        if not first_trade:
            return {"IPO_BASE": False, "IPO_Days": np.nan}
        ipo_date = dt.datetime.utcfromtimestamp(int(first_trade)).date()
        ipo_days = (dt.date.today() - ipo_date).days
        return {"IPO_BASE": bool(ipo_days <= 365), "IPO_Days": ipo_days}
    except Exception:
        return {"IPO_BASE": False, "IPO_Days": np.nan}


# ============================================================
# 5) FUNDAMENTALS (QUALITY + ALTMAN Z + BENEISH M + QoQ/YoY)
# ============================================================


def safe_get(df: Optional[pd.DataFrame], row_name: str, col_idx: int = 0) -> Optional[float]:
    if df is None or df.empty or row_name not in df.index:
        return None
    try:
        v = df.loc[row_name].iloc[col_idx]
        if pd.isna(v):
            return None
        return float(v)
    except Exception:
        return None


def compute_altman_z(info: Dict[str, Any], financials: Optional[pd.DataFrame], balance: Optional[pd.DataFrame]) -> Optional[float]:
    ta_ = safe_get(balance, "Total Assets", 0)
    tl_ = safe_get(balance, "Total Liab", 0) or safe_get(balance, "Total Liabilities Net Minority Interest", 0)
    ca_ = safe_get(balance, "Total Current Assets", 0)
    cl_ = safe_get(balance, "Total Current Liabilities", 0)
    re_ = safe_get(balance, "Retained Earnings", 0)
    ebit_ = safe_get(financials, "Ebit", 0) or safe_get(financials, "EBIT", 0)
    sales_ = safe_get(financials, "Total Revenue", 0)

    mcap = info.get("marketCap")
    try:
        mcap = float(mcap) if mcap is not None else None
    except Exception:
        mcap = None

    if any(x is None for x in [ta_, tl_, ca_, cl_, re_, ebit_, sales_, mcap]) or ta_ == 0 or tl_ == 0:
        return None

    wc = ca_ - cl_
    z = 1.2 * (wc / ta_) + 1.4 * (re_ / ta_) + 3.3 * (ebit_ / ta_) + 0.6 * (mcap / tl_) + 1.0 * (sales_ / ta_)
    return float(z)


def compute_beneish_m(financials: Optional[pd.DataFrame], balance: Optional[pd.DataFrame]) -> Optional[float]:
    if financials is None or balance is None or financials.empty or balance.empty:
        return None
    if financials.shape[1] < 2 or balance.shape[1] < 2:
        return None

    def g(row: str, idx: int) -> Optional[float]:
        return safe_get(financials, row, idx)

    def b(row: str, idx: int) -> Optional[float]:
        return safe_get(balance, row, idx)

    sales_t = g("Total Revenue", 0)
    sales_t1 = g("Total Revenue", 1)
    cogs_t = g("Cost Of Revenue", 0)
    cogs_t1 = g("Cost Of Revenue", 1)

    ar_t = b("Net Receivables", 0) or b("Accounts Receivable", 0)
    ar_t1 = b("Net Receivables", 1) or b("Accounts Receivable", 1)
    ta_t = b("Total Assets", 0)
    ta_t1 = b("Total Assets", 1)
    ppe_t = b("Property Plant Equipment", 0) or b("Property Plant And Equipment Net", 0)
    ppe_t1 = b("Property Plant Equipment", 1) or b("Property Plant And Equipment Net", 1)
    ca_t = b("Total Current Assets", 0)
    ca_t1 = b("Total Current Assets", 1)
    cl_t = b("Total Current Liabilities", 0)
    cl_t1 = b("Total Current Liabilities", 1)
    ltd_t = b("Long Term Debt", 0) or b("Long Term Debt And Capital Lease Obligation", 0)
    ltd_t1 = b("Long Term Debt", 1) or b("Long Term Debt And Capital Lease Obligation", 1)

    dep_t = g("Reconciled Depreciation", 0) or g("Depreciation", 0)
    dep_t1 = g("Reconciled Depreciation", 1) or g("Depreciation", 1)

    sga_t = g("Selling General Administrative", 0) or g("Selling General And Administration", 0)
    sga_t1 = g("Selling General Administrative", 1) or g("Selling General And Administration", 1)

    if any(v is None for v in [sales_t, sales_t1, ta_t, ta_t1]) or sales_t == 0 or sales_t1 == 0:
        return None

    dsri = (ar_t / sales_t) / (ar_t1 / sales_t1) if all(v is not None and v != 0 for v in [ar_t, sales_t, ar_t1, sales_t1]) else None
    gmi = (
        ((sales_t1 - cogs_t1) / sales_t1) / ((sales_t - cogs_t) / sales_t)
        if all(v is not None and v != 0 for v in [sales_t, sales_t1, cogs_t, cogs_t1])
        else None
    )
    aqi = (
        (1 - ((ca_t + ppe_t) / ta_t)) / (1 - ((ca_t1 + ppe_t1) / ta_t1))
        if all(v is not None and v != 0 for v in [ca_t, ppe_t, ta_t, ca_t1, ppe_t1, ta_t1])
        else None
    )
    sgi = sales_t / sales_t1
    depi = (
        (dep_t1 / (dep_t1 + ppe_t1)) / (dep_t / (dep_t + ppe_t))
        if all(v is not None and v != 0 for v in [dep_t, dep_t1, ppe_t, ppe_t1]) and (dep_t + ppe_t) != 0 and (dep_t1 + ppe_t1) != 0
        else None
    )
    sgai = (
        (sga_t / sales_t) / (sga_t1 / sales_t1)
        if all(v is not None and v != 0 for v in [sga_t, sales_t, sga_t1, sales_t1])
        else None
    )
    lvgi = (
        ((cl_t + ltd_t) / ta_t) / ((cl_t1 + ltd_t1) / ta_t1)
        if all(v is not None and v != 0 for v in [cl_t, ltd_t, ta_t, cl_t1, ltd_t1, ta_t1])
        else None
    )

    tata = 0.0
    needed = [dsri, gmi, aqi, sgi, depi, sgai, lvgi]
    if any(v is None for v in needed):
        return None

    m = -4.84 + 0.92 * dsri + 0.528 * gmi + 0.404 * aqi + 0.892 * sgi + 0.115 * depi - 0.172 * sgai + 4.679 * tata - 0.327 * lvgi
    return float(m)


def get_fundamentals(symbol: str) -> Dict[str, Any]:
    """
    Best-effort Yahoo fundamentals. Always returns a dict (may be empty/None values).
    """
    try:
        stock = yf.Ticker(symbol)
        info = stock.info or {}
        financials = getattr(stock, "financials", None)
        balance = getattr(stock, "balance_sheet", None)
        q_fin = getattr(stock, "quarterly_financials", None)

        out: Dict[str, Any] = {
            "PE": info.get("trailingPE"),
            "ROE": info.get("returnOnEquity"),
            "ROA": info.get("returnOnAssets"),
            "DebtToEquity": info.get("debtToEquity"),
            "ProfitMargin": info.get("profitMargins"),
            "RevenueGrowth": info.get("revenueGrowth"),
            "GrossMargin": info.get("grossMargins"),
            "OperatingMargin": info.get("operatingMargins"),
            "EPS_Est_Growth%": (info.get("earningsGrowth") * 100.0) if info.get("earningsGrowth") is not None else None,
        }

        out["AltmanZ"] = compute_altman_z(info, financials, balance)
        out["BeneishM"] = compute_beneish_m(financials, balance)

        # Revenue QoQ/YoY (last 6 quarters)
        if isinstance(q_fin, pd.DataFrame) and (not q_fin.empty) and ("Total Revenue" in q_fin.index):
            rev = q_fin.loc["Total Revenue"].astype(float).sort_index()
            qoq = rev.pct_change() * 100.0
            yoy = rev.pct_change(4) * 100.0
            for col in rev.index[-6:]:
                label = pd.to_datetime(col).date().isoformat() if hasattr(col, "date") else str(col)
                out[f"Rev_{label}"] = float(rev[col])
                out[f"Rev_QoQ%_{label}"] = float(qoq[col]) if not pd.isna(qoq[col]) else None
                out[f"Rev_YoY%_{label}"] = float(yoy[col]) if not pd.isna(yoy[col]) else None

        # Net income QoQ/YoY (last 6 quarters)
        if isinstance(q_fin, pd.DataFrame) and (not q_fin.empty) and ("Net Income" in q_fin.index):
            ni = q_fin.loc["Net Income"].astype(float).sort_index()
            ni_qoq = ni.pct_change() * 100.0
            ni_yoy = ni.pct_change(4) * 100.0
            for col in ni.index[-6:]:
                label = pd.to_datetime(col).date().isoformat() if hasattr(col, "date") else str(col)
                out[f"NI_{label}"] = float(ni[col])
                out[f"NI_QoQ%_{label}"] = float(ni_qoq[col]) if not pd.isna(ni_qoq[col]) else None
                out[f"NI_YoY%_{label}"] = float(ni_yoy[col]) if not pd.isna(ni_yoy[col]) else None

        # Fundamental quality score (v1)
        quality = 0
        if out.get("ROE") is not None and out["ROE"] >= 0.15:
            quality += 1
        if out.get("DebtToEquity") is not None and out["DebtToEquity"] <= 1.0:
            quality += 1
        if out.get("OperatingMargin") is not None and out["OperatingMargin"] >= 0.12:
            quality += 1
        if out.get("RevenueGrowth") is not None and out["RevenueGrowth"] >= 0.10:
            quality += 1
        if out.get("ProfitMargin") is not None and out["ProfitMargin"] >= 0.08:
            quality += 1
        out["FundamentalQualityScore"] = quality
        return out
    except Exception:
        return {"FundamentalQualityScore": 0}


# ============================================================
# 6) EXPORT
# ============================================================


def export_excel(top: pd.DataFrame, all_rows: pd.DataFrame, out_path: Path) -> None:
    try:
        with pd.ExcelWriter(out_path, engine="openpyxl") as writer:
            top.to_excel(writer, sheet_name="Top10", index=False)
            all_rows.to_excel(writer, sheet_name="AllScanned", index=False)
    except Exception:
        out_csv = out_path.with_suffix(".csv")
        top.to_csv(out_csv, index=False)
        print(f"Excel export unavailable; saved CSV to {out_csv.resolve()}")


# ============================================================
# 7) MAIN SCREENER (ALWAYS RETURNS TOP-N)
# ============================================================

def normalize_ohlcv_df(df: pd.DataFrame) -> pd.DataFrame:
    """
    yfinance v1.2+ may return MultiIndex columns even for a single ticker.
    This normalizes to single-level columns with standard OHLCV names.
    """
    df = df.copy()

    if isinstance(df.columns, pd.MultiIndex):
        lvl0 = set(df.columns.get_level_values(0))
        lvl1 = set(df.columns.get_level_values(1))
        fields = {"Open", "High", "Low", "Close", "Adj Close", "Volume", "Dividends", "Stock Splits", "Capital Gains"}

        if lvl0.issubset(fields):
            df.columns = df.columns.get_level_values(0)
        elif lvl1.issubset(fields):
            df.columns = df.columns.get_level_values(1)
        else:
            df.columns = ["_".join(map(str, c)).strip() for c in df.columns.to_flat_index()]

    # Ensure string columns and standard capitalization
    df.columns = [str(c) for c in df.columns]
    df = df.rename(columns={c: c.title() for c in df.columns})
    return df


def run_full_system(
    universe_limit: int = 500,
    min_confluence_score: int = 6,
    period: str = "2y",
    interval: str = "1d",
    top_n: int = 10,
    fundamentals_top_k_multiplier: int = 5,
    start_index: int = 0,
) -> pd.DataFrame:
    all_stocks = get_nse_stocks()
    total_universe = len(all_stocks)

    start = max(0, int(start_index))
    if start >= total_universe:
        stocks: List[str] = []
    else:
        if universe_limit and universe_limit > 0:
            end = min(start + universe_limit, total_universe)
            stocks = all_stocks[start:end]
        else:
            stocks = all_stocks[start:]

    counters = {
        "download_empty": 0,
        "missing_columns": 0,
        "too_short": 0,
        "ok": 0,
        "errors": 0,
        "fundamentals_errors": 0,
    }

    rows: List[Dict[str, Any]] = []
    print(f"Scanning {len(stocks)} NSE stocks (indices {start} to {start + max(len(stocks) - 1, 0)} of {total_universe})...")

    for symbol in tqdm(stocks):
        try:
            df = yf.download(symbol, period=period, interval=interval, progress=False, auto_adjust=False)
            if df is None or df.empty:
                counters["download_empty"] += 1
                continue

            df = normalize_ohlcv_df(df)
            needed = {"Open", "High", "Low", "Close", "Volume"}
            if not needed.issubset(df.columns):
                counters["missing_columns"] += 1
                continue

            if len(df) < 220:
                counters["too_short"] += 1
                continue

            df = add_technical_indicators(df)
            latest = df.iloc[-1]

            confluence_score, confluence_signals = predicta_v4_confluence(latest)
            passed = bool(confluence_score >= min_confluence_score)

            pv = calculate_price_volume_metrics(df)
            vcp = detect_vcp(df)
            sfp = detect_swing_failure(df)

            # IPO base can fail due to Yahoo .info; don't let it kill candidate
            try:
                ipo = detect_ipo_base(symbol)
            except Exception:
                ipo = {"IPO_BASE": False, "IPO_Days": np.nan}

            setup_score = (2 if vcp.get("VCP") else 0) + (2 if sfp.get("SFP") else 0) + (1 if ipo.get("IPO_BASE") else 0)
            base_score = int(confluence_score + setup_score)

            row: Dict[str, Any] = {
                "Symbol": symbol,
                "Score": base_score,  # will be upgraded after fundamentals (top-K)
                "ConfluenceScore": confluence_score,
                "SetupScore": setup_score,
                "PassedFilter": passed,
                "Price": float(latest["Close"]),
                "RSI": float(latest.get("rsi", np.nan)),
                "ADX": float(latest.get("adx", np.nan)),
                "Stoch": float(latest.get("stoch", np.nan)),
                "VolMult": float(latest.get("vol_mult", np.nan)),
                "ATR": float(latest.get("atr", np.nan)),
                "DeltaProxy": float(latest.get("delta_proxy", np.nan)),
                **{f"C_{k}": v for k, v in confluence_signals.items()},
                **pv,
                **vcp,
                **sfp,
                **ipo,
                "FundamentalQualityScore": np.nan,
                "AltmanZ": np.nan,
                "BeneishM": np.nan,
                "EPS_Est_Growth%": np.nan,
            }
            rows.append(row)
            counters["ok"] += 1
        except Exception:
            counters["errors"] += 1
            continue

    if not rows:
        print("No rows scanned successfully. Summary:", counters)
        return pd.DataFrame()

    all_df = pd.DataFrame(rows)

    pool = all_df[all_df["PassedFilter"]].copy()
    if pool.empty:
        print(
            f"No symbols met min confluence score >= {min_confluence_score}. "
            f"Falling back to best available Top-{top_n} by score."
        )
        pool = all_df.copy()

    pool = pool.sort_values(by=["Score", "ConfluenceScore", "RVol"], ascending=False)

    # Pull fundamentals only for top-K (speed + reliability)
    top_k = min(len(pool), max(top_n * fundamentals_top_k_multiplier, top_n))
    pool_topk = pool.head(top_k).copy()

    fundamentals_rows: List[Dict[str, Any]] = []
    for sym in tqdm(pool_topk["Symbol"].tolist(), desc="Fundamentals (top-K)"):
        try:
            fundamentals_rows.append({"Symbol": sym, **get_fundamentals(sym)})
        except Exception:
            counters["fundamentals_errors"] += 1
            fundamentals_rows.append({"Symbol": sym, "FundamentalQualityScore": 0})

    fund_df = pd.DataFrame(fundamentals_rows).drop_duplicates(subset=["Symbol"])
    pool_topk = pool_topk.merge(fund_df, on="Symbol", how="left", suffixes=("", "_fund"))

    # Upgrade score with fundamentals bonus
    fq = pool_topk.get("FundamentalQualityScore")
    if fq is None:
        pool_topk["FundamentalQualityScore"] = 0
    pool_topk["FundamentalQualityScore"] = pool_topk["FundamentalQualityScore"].fillna(0).astype(int)
    pool_topk["Score"] = (pool_topk["ConfluenceScore"].astype(int) + pool_topk["SetupScore"].astype(int) + pool_topk["FundamentalQualityScore"].astype(int)).astype(int)

    pool_topk = pool_topk.sort_values(by=["Score", "ConfluenceScore", "SetupScore", "RVol"], ascending=False)
    top = pool_topk.head(top_n).reset_index(drop=True)

    out_path = Path("Predicta_Top10.xlsx")
    export_excel(top=top, all_rows=all_df.sort_values(by=["Score", "ConfluenceScore"], ascending=False), out_path=out_path)

    print("\nTOP SWING CANDIDATES (next day watchlist)\n")
    cols = [
        "Symbol",
        "Score",
        "ConfluenceScore",
        "SetupScore",
        "FundamentalQualityScore",
        "Price",
        "VCP",
        "SFP",
        "IPO_BASE",
        "AltmanZ",
        "BeneishM",
        "EPS_Est_Growth%",
        "RVol",
        "ADR%",
    ]
    cols = [c for c in cols if c in top.columns]
    with pd.option_context("display.max_columns", 200, "display.width", 240):
        print(top[cols])

    print("\nScan summary:", counters)
    print(f"Saved report to {out_path.resolve()}")
    return top


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="NSE swing screener (confluence + setups + fundamentals)")
    p.add_argument("--universe-limit", type=int, default=500, help="How many NSE symbols to scan (0 = all)")
    p.add_argument("--min-score", type=int, default=6, help="Minimum confluence score (0-8). If none pass, fallback Top-N is produced.")
    p.add_argument("--period", type=str, default="2y", help="yfinance history period (e.g. 6mo, 1y, 2y)")
    p.add_argument("--interval", type=str, default="1d", help="yfinance interval (e.g. 1d)")
    p.add_argument("--top-n", type=int, default=10, help="Top N to export")
    p.add_argument("--start-index", type=int, default=0, help="Start index in NSE list for chunking (0-based)")
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run_full_system(
        universe_limit=args.universe_limit,
        min_confluence_score=args.min_score,
        period=args.period,
        interval=args.interval,
        top_n=args.top_n,
        start_index=args.start_index,
    )
    raise SystemExit(0)

import argparse
import datetime as dt
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import requests
import ta
import yfinance as yf
from tqdm import tqdm


CACHE_DIR = Path(".cache")
CACHE_DIR.mkdir(exist_ok=True)
NSE_EQUITY_LIST_CACHE = CACHE_DIR / "EQUITY_L.csv"


# ============================================================
# 1) NSE UNIVERSE
# ============================================================


def get_nse_stocks(cache_ttl_hours: int = 24) -> List[str]:
    """
    Returns NSE equity symbols as Yahoo tickers (e.g., RELIANCE.NS).
    Uses NSE archive CSV with a small local cache to reduce failures.
    """
    url = "https://archives.nseindia.com/content/equities/EQUITY_L.csv"
    headers = {
        "User-Agent": "Mozilla/5.0",
        "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
        "Accept-Language": "en-US,en;q=0.5",
        "Connection": "keep-alive",
    }

    def cache_is_fresh() -> bool:
        if not NSE_EQUITY_LIST_CACHE.exists():
            return False
        age = dt.datetime.now() - dt.datetime.fromtimestamp(NSE_EQUITY_LIST_CACHE.stat().st_mtime)
        return age.total_seconds() < cache_ttl_hours * 3600

    if not cache_is_fresh():
        resp = requests.get(url, headers=headers, timeout=30)
        resp.raise_for_status()
        NSE_EQUITY_LIST_CACHE.write_text(resp.text, encoding="utf-8")

    text = NSE_EQUITY_LIST_CACHE.read_text(encoding="utf-8", errors="ignore")
    lines = text.splitlines()
    if not lines:
        return []

    symbols: List[str] = []
    for line in lines[1:]:
        if not line.strip():
            continue
        sym = line.split(",")[0].strip().strip('"')
        if sym:
            symbols.append(f"{sym}.NS")
    return symbols


# ============================================================
# 2) TECHNICAL ENGINE (Predicta V4-ish)
# ============================================================


def add_technical_indicators(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    df["ema8"] = ta.trend.ema_indicator(df["Close"], 8)
    df["ema21"] = ta.trend.ema_indicator(df["Close"], 21)
    df["ema55"] = ta.trend.ema_indicator(df["Close"], 55)
    df["ema144"] = ta.trend.ema_indicator(df["Close"], 144)
    df["ema50"] = ta.trend.ema_indicator(df["Close"], 50)
    df["ema200"] = ta.trend.ema_indicator(df["Close"], 200)

    df["rsi"] = ta.momentum.rsi(df["Close"], 14)

    macd = ta.trend.MACD(df["Close"])
    df["macd"] = macd.macd()
    df["macd_signal"] = macd.macd_signal()

    df["adx"] = ta.trend.adx(df["High"], df["Low"], df["Close"], 14)
    df["stoch"] = ta.momentum.stoch(df["High"], df["Low"], df["Close"], 14)

    df["vol_ma20"] = df["Volume"].rolling(20).mean()
    df["vol_mult"] = df["Volume"] / df["vol_ma20"]

    df["atr"] = ta.volatility.average_true_range(df["High"], df["Low"], df["Close"], 14)
    df["atr_ma20"] = df["atr"].rolling(20).mean()
    df["atr_high"] = df["atr"] > df["atr_ma20"]

    # Price + Volume panel helpers
    df["adr20"] = (df["High"] - df["Low"]).rolling(20).mean()
    df["adrp20"] = (df["adr20"] / df["Close"]) * 100.0
    df["rvol50"] = df["Volume"] / df["Volume"].rolling(50).mean()
    df["obv"] = ta.volume.on_balance_volume(df["Close"], df["Volume"])

    # Daily "delta" approximation via CLV * volume
    hl_range = (df["High"] - df["Low"]).replace(0, np.nan)
    clv = ((df["Close"] - df["Low"]) - (df["High"] - df["Close"])) / hl_range
    df["clv"] = clv.fillna(0.0)
    df["delta_proxy"] = df["clv"] * df["Volume"]

    return df


def predicta_v4_confluence(latest: pd.Series) -> Tuple[int, Dict[str, Any]]:
    """
    Returns (subscore, signal_details) for:
    MACD, RSI, STOCH, VOLUME, DELTA, TREND, ADX, ATR
    """
    signals = {
        "MACD": bool(latest.get("macd", np.nan) > latest.get("macd_signal", np.nan)),
        "RSI": bool(latest.get("rsi", np.nan) >= 55),
        "STOCH": bool(latest.get("stoch", np.nan) >= 60),
        "VOLUME": bool(latest.get("vol_mult", np.nan) >= 1.2),
        "DELTA": bool(latest.get("delta_proxy", 0.0) > 0),
        "TREND": bool(latest.get("Close", np.nan) > latest.get("ema50", np.nan) > latest.get("ema200", np.nan)),
        "ADX": bool(latest.get("adx", np.nan) >= 20),
        "ATR": bool(bool(latest.get("atr_high", False))),
    }
    score = int(sum(1 for v in signals.values() if v))
    return score, signals


# ============================================================
# 3) PRICE + VOLUME STRENGTH PANEL
# ============================================================


def calculate_price_volume_metrics(df: pd.DataFrame) -> Dict[str, Any]:
    latest = df.iloc[-1]
    ema_distance = {
        "EMA8%": (latest["Close"] - latest["ema8"]) / latest["ema8"] * 100,
        "EMA21%": (latest["Close"] - latest["ema21"]) / latest["ema21"] * 100,
        "EMA55%": (latest["Close"] - latest["ema55"]) / latest["ema55"] * 100,
        "EMA144%": (latest["Close"] - latest["ema144"]) / latest["ema144"] * 100,
    }

    last30 = df.iloc[-30:]
    up_days = int((last30["Close"] > last30["Open"]).sum())
    down_days = int(len(last30) - up_days)

    high_52w = float(df["High"].tail(252).max()) if len(df) >= 252 else float(df["High"].max())
    dist_52w_high_pct = float((latest["Close"] / high_52w - 1.0) * 100.0) if high_52w else np.nan

    out: Dict[str, Any] = {}
    out.update(ema_distance)
    out.update(
        {
            "ADR": float(latest.get("adr20", np.nan)),
            "ADR%": float(latest.get("adrp20", np.nan)),
            "RVol": float(latest.get("rvol50", np.nan)),
            "U/D Days(30)": f"{up_days}/{down_days}",
            "Dist52WHigh%": dist_52w_high_pct,
        }
    )
    return out


# ============================================================
# 4) SETUP DETECTORS (VCP / IPO BASE / SWING FAILURE)
# ============================================================


def detect_vcp(df: pd.DataFrame, lookback: int = 60) -> Dict[str, Any]:
    """
    VCP (daily approximation):
    - average range% contracts across 3 segments
    - volume contracts across the same segments
    """
    if len(df) < lookback + 5:
        return {"VCP": False, "VCP_Reason": "not_enough_data"}

    w = df.tail(lookback).copy()
    w["range%"] = (w["High"] - w["Low"]) / w["Close"] * 100.0
    r1 = float(w["range%"].iloc[:20].mean())
    r2 = float(w["range%"].iloc[20:40].mean())
    r3 = float(w["range%"].iloc[40:60].mean())

    v1 = float(w["Volume"].iloc[:20].mean())
    v2 = float(w["Volume"].iloc[20:40].mean())
    v3 = float(w["Volume"].iloc[40:60].mean())

    range_contracting = (r1 > r2 > r3) and (r3 < 6.0)
    volume_contracting = (v1 > v2 > v3)

    return {
        "VCP": bool(range_contracting and volume_contracting),
        "VCP_R1": r1,
        "VCP_R2": r2,
        "VCP_R3": r3,
        "VCP_V1": v1,
        "VCP_V2": v2,
        "VCP_V3": v3,
    }


def detect_swing_failure(df: pd.DataFrame, lookback: int = 20) -> Dict[str, Any]:
    """
    Bullish SFP (daily approximation):
    - last candle makes lower low vs recent swing low
    - but closes back above that swing low
    """
    if len(df) < lookback + 2:
        return {"SFP": False, "SFP_Level": np.nan}

    recent = df.iloc[-(lookback + 1) : -1]
    swing_low = float(recent["Low"].min())
    last = df.iloc[-1]

    sfp = bool((last["Low"] < swing_low) and (last["Close"] > swing_low) and (last["Close"] > last["Open"]))
    return {"SFP": sfp, "SFP_Level": swing_low}


def detect_ipo_base(symbol: str) -> Dict[str, Any]:
    """
    IPO base needs listing date; Yahoo sometimes provides firstTradeDateEpochUtc.
    """
    try:
        info = yf.Ticker(symbol).info or {}
        first_trade = info.get("firstTradeDateEpochUtc")
        if not first_trade:
            return {"IPO_BASE": False, "IPO_Days": np.nan, "IPO_Reason": "missing_first_trade_date"}
        ipo_date = dt.datetime.utcfromtimestamp(int(first_trade)).date()
        ipo_days = (dt.date.today() - ipo_date).days
        return {"IPO_BASE": bool(ipo_days <= 365), "IPO_Days": ipo_days}
    except Exception:
        return {"IPO_BASE": False, "IPO_Days": np.nan, "IPO_Reason": "info_error"}


# ============================================================
# 5) FUNDAMENTALS (QUALITY + ALTMAN Z + BENEISH M + QoQ revenue)
# ============================================================


def safe_get(df: Optional[pd.DataFrame], row_name: str, col_idx: int = 0) -> Optional[float]:
    if df is None or df.empty or row_name not in df.index:
        return None
    try:
        v = df.loc[row_name].iloc[col_idx]
        if pd.isna(v):
            return None
        return float(v)
    except Exception:
        return None


def compute_altman_z(info: Dict[str, Any], financials: Optional[pd.DataFrame], balance: Optional[pd.DataFrame]) -> Optional[float]:
    """
    Altman Z-score (public manufacturing form).
    Returns None if statements are missing/incomplete.
    """
    ta_ = safe_get(balance, "Total Assets", 0)
    tl_ = safe_get(balance, "Total Liab", 0) or safe_get(balance, "Total Liabilities Net Minority Interest", 0)
    ca_ = safe_get(balance, "Total Current Assets", 0)
    cl_ = safe_get(balance, "Total Current Liabilities", 0)
    re_ = safe_get(balance, "Retained Earnings", 0)
    ebit_ = safe_get(financials, "Ebit", 0) or safe_get(financials, "EBIT", 0)
    sales_ = safe_get(financials, "Total Revenue", 0)

    mcap = info.get("marketCap")
    try:
        mcap = float(mcap) if mcap is not None else None
    except Exception:
        mcap = None

    if any(x is None for x in [ta_, tl_, ca_, cl_, re_, ebit_, sales_, mcap]):
        return None
    if ta_ == 0 or tl_ == 0:
        return None

    wc = ca_ - cl_
    z = (
        1.2 * (wc / ta_)
        + 1.4 * (re_ / ta_)
        + 3.3 * (ebit_ / ta_)
        + 0.6 * (mcap / tl_)
        + 1.0 * (sales_ / ta_)
    )
    return float(z)


def compute_beneish_m(financials: Optional[pd.DataFrame], balance: Optional[pd.DataFrame]) -> Optional[float]:
    """
    Beneish M-score (best-effort). Needs two annual periods.
    Returns None when required lines are missing.
    """
    if financials is None or balance is None or financials.empty or balance.empty:
        return None
    if financials.shape[1] < 2 or balance.shape[1] < 2:
        return None

    def g(row: str, idx: int) -> Optional[float]:
        return safe_get(financials, row, idx)

    def b(row: str, idx: int) -> Optional[float]:
        return safe_get(balance, row, idx)

    sales_t = g("Total Revenue", 0)
    sales_t1 = g("Total Revenue", 1)
    cogs_t = g("Cost Of Revenue", 0)
    cogs_t1 = g("Cost Of Revenue", 1)

    ar_t = b("Net Receivables", 0) or b("Accounts Receivable", 0)
    ar_t1 = b("Net Receivables", 1) or b("Accounts Receivable", 1)
    ta_t = b("Total Assets", 0)
    ta_t1 = b("Total Assets", 1)
    ppe_t = b("Property Plant Equipment", 0) or b("Property Plant And Equipment Net", 0)
    ppe_t1 = b("Property Plant Equipment", 1) or b("Property Plant And Equipment Net", 1)
    ca_t = b("Total Current Assets", 0)
    ca_t1 = b("Total Current Assets", 1)
    cl_t = b("Total Current Liabilities", 0)
    cl_t1 = b("Total Current Liabilities", 1)
    ltd_t = b("Long Term Debt", 0) or b("Long Term Debt And Capital Lease Obligation", 0)
    ltd_t1 = b("Long Term Debt", 1) or b("Long Term Debt And Capital Lease Obligation", 1)

    dep_t = g("Reconciled Depreciation", 0) or g("Depreciation", 0)
    dep_t1 = g("Reconciled Depreciation", 1) or g("Depreciation", 1)

    sga_t = g("Selling General Administrative", 0) or g("Selling General And Administration", 0)
    sga_t1 = g("Selling General Administrative", 1) or g("Selling General And Administration", 1)

    if any(v is None for v in [sales_t, sales_t1, ta_t, ta_t1]) or sales_t == 0 or sales_t1 == 0:
        return None

    dsri = (ar_t / sales_t) / (ar_t1 / sales_t1) if all(v is not None and v != 0 for v in [ar_t, sales_t, ar_t1, sales_t1]) else None
    gmi = (
        ((sales_t1 - cogs_t1) / sales_t1) / ((sales_t - cogs_t) / sales_t)
        if all(v is not None and v != 0 for v in [sales_t, sales_t1, cogs_t, cogs_t1])
        else None
    )
    aqi = (
        (1 - ((ca_t + ppe_t) / ta_t)) / (1 - ((ca_t1 + ppe_t1) / ta_t1))
        if all(v is not None and v != 0 for v in [ca_t, ppe_t, ta_t, ca_t1, ppe_t1, ta_t1])
        else None
    )
    sgi = sales_t / sales_t1
    depi = (
        (dep_t1 / (dep_t1 + ppe_t1)) / (dep_t / (dep_t + ppe_t))
        if all(v is not None and v != 0 for v in [dep_t, dep_t1, ppe_t, ppe_t1]) and (dep_t + ppe_t) != 0 and (dep_t1 + ppe_t1) != 0
        else None
    )
    sgai = (
        (sga_t / sales_t) / (sga_t1 / sales_t1)
        if all(v is not None and v != 0 for v in [sga_t, sales_t, sga_t1, sales_t1])
        else None
    )
    lvgi = (
        ((cl_t + ltd_t) / ta_t) / ((cl_t1 + ltd_t1) / ta_t1)
        if all(v is not None and v != 0 for v in [cl_t, ltd_t, ta_t, cl_t1, ltd_t1, ta_t1])
        else None
    )

    tata = 0.0  # best-effort; needs cashflow data for true value

    needed = [dsri, gmi, aqi, sgi, depi, sgai, lvgi]
    if any(v is None for v in needed):
        return None

    m = -4.84 + 0.92 * dsri + 0.528 * gmi + 0.404 * aqi + 0.892 * sgi + 0.115 * depi - 0.172 * sgai + 4.679 * tata - 0.327 * lvgi
    return float(m)


def get_fundamentals(symbol: str) -> Dict[str, Any]:
    stock = yf.Ticker(symbol)
    info = stock.info or {}

    financials = getattr(stock, "financials", None)
    balance = getattr(stock, "balance_sheet", None)
    q_fin = getattr(stock, "quarterly_financials", None)

    fundamentals: Dict[str, Any] = {
        "PE": info.get("trailingPE"),
        "ROE": info.get("returnOnEquity"),
        "ROA": info.get("returnOnAssets"),
        "DebtToEquity": info.get("debtToEquity"),
        "ProfitMargin": info.get("profitMargins"),
        "RevenueGrowth": info.get("revenueGrowth"),
        "GrossMargin": info.get("grossMargins"),
        "OperatingMargin": info.get("operatingMargins"),
        "EPS_Est_Growth%": (info.get("earningsGrowth") * 100.0) if info.get("earningsGrowth") is not None else None,
    }

    fundamentals["AltmanZ"] = compute_altman_z(info, financials, balance)
    fundamentals["BeneishM"] = compute_beneish_m(financials, balance)

    # Detailed QoQ Revenue % per quarter (keep last 6 quarters)
    if isinstance(q_fin, pd.DataFrame) and (not q_fin.empty) and ("Total Revenue" in q_fin.index):
        rev = q_fin.loc["Total Revenue"].astype(float)
        rev = rev.sort_index()
        qoq = rev.pct_change() * 100.0
        yoy = rev.pct_change(4) * 100.0
        for col in rev.index[-6:]:
            label = pd.to_datetime(col).date().isoformat() if hasattr(col, "date") else str(col)
            fundamentals[f"Rev_{label}"] = float(rev[col])
            fundamentals[f"QoQ%_{label}"] = float(qoq[col]) if not pd.isna(qoq[col]) else None
            fundamentals[f"YoY%_{label}"] = float(yoy[col]) if not pd.isna(yoy[col]) else None

    # Quarterly Earnings (Net Income) QoQ & YoY (best-effort)
    if isinstance(q_fin, pd.DataFrame) and (not q_fin.empty) and ("Net Income" in q_fin.index):
        ni = q_fin.loc["Net Income"].astype(float)
        ni = ni.sort_index()
        ni_qoq = ni.pct_change() * 100.0
        ni_yoy = ni.pct_change(4) * 100.0
        for col in ni.index[-6:]:
            label = pd.to_datetime(col).date().isoformat() if hasattr(col, "date") else str(col)
            fundamentals[f"NI_{label}"] = float(ni[col])
            fundamentals[f"NI_QoQ%_{label}"] = float(ni_qoq[col]) if not pd.isna(ni_qoq[col]) else None
            fundamentals[f"NI_YoY%_{label}"] = float(ni_yoy[col]) if not pd.isna(ni_yoy[col]) else None

    # Fundamental Quality Scorecard (v1)
    quality = 0
    if fundamentals.get("ROE") is not None and fundamentals["ROE"] >= 0.15:
        quality += 1
    if fundamentals.get("DebtToEquity") is not None and fundamentals["DebtToEquity"] <= 1.0:
        quality += 1
    if fundamentals.get("OperatingMargin") is not None and fundamentals["OperatingMargin"] >= 0.12:
        quality += 1
    if fundamentals.get("RevenueGrowth") is not None and fundamentals["RevenueGrowth"] >= 0.10:
        quality += 1
    if fundamentals.get("ProfitMargin") is not None and fundamentals["ProfitMargin"] >= 0.08:
        quality += 1
    fundamentals["FundamentalQualityScore"] = quality

    return fundamentals


# ============================================================
# 6) RUNNER + REPORT
# ============================================================


@dataclass
class Candidate:
    symbol: str
    score: int
    price: float
    details: Dict[str, Any]


def export_excel(top: pd.DataFrame, all_candidates: pd.DataFrame, out_path: Path) -> None:
    """
    Writes an Excel report if openpyxl is installed; otherwise falls back to CSV.
    """
    try:
        with pd.ExcelWriter(out_path, engine="openpyxl") as writer:
            top.to_excel(writer, sheet_name="Top10", index=False)
            all_candidates.to_excel(writer, sheet_name="AllCandidates", index=False)
    except Exception:
        out_csv = out_path.with_suffix(".csv")
        top.to_csv(out_csv, index=False)
        print(f"Excel export unavailable; saved CSV to {out_csv.resolve()}")


def run_full_system(
    universe_limit: int = 500,
    min_confluence_score: int = 6,
    period: str = "1y",
    interval: str = "1d",
    top_n: int = 10,
) -> pd.DataFrame:
    stocks = get_nse_stocks()
    if universe_limit and universe_limit > 0:
        stocks = stocks[:universe_limit]

    candidates: List[Candidate] = []
    print(f"Scanning {len(stocks)} NSE stocks...")

    for symbol in tqdm(stocks):
        try:
            df = yf.download(symbol, period=period, interval=interval, progress=False, auto_adjust=False)
            if df is None or df.empty:
                continue

            df = df.rename(columns={c: c.title() for c in df.columns})
            needed = {"Open", "High", "Low", "Close", "Volume"}
            if not needed.issubset(df.columns):
                continue

            if len(df) < 220:
                continue

            df = add_technical_indicators(df)
            latest = df.iloc[-1]

            confluence_score, confluence_signals = predicta_v4_confluence(latest)
            if confluence_score < min_confluence_score:
                continue

            pv = calculate_price_volume_metrics(df)
            vcp = detect_vcp(df)
            sfp = detect_swing_failure(df)
            ipo = detect_ipo_base(symbol)
            fundamentals = get_fundamentals(symbol)

            setup_score = (2 if vcp.get("VCP") else 0) + (2 if sfp.get("SFP") else 0) + (1 if ipo.get("IPO_BASE") else 0)
            fund_bonus = int(fundamentals.get("FundamentalQualityScore") or 0)
            total_score = int(confluence_score + setup_score + fund_bonus)

            row: Dict[str, Any] = {
                "Symbol": symbol,
                "Score": total_score,
                "ConfluenceScore": confluence_score,
                "SetupScore": setup_score,
                "FundBonus": fund_bonus,
                "Price": float(latest["Close"]),
                "RSI": float(latest.get("rsi", np.nan)),
                "ADX": float(latest.get("adx", np.nan)),
                "Stoch": float(latest.get("stoch", np.nan)),
                "VolMult": float(latest.get("vol_mult", np.nan)),
                "ATR": float(latest.get("atr", np.nan)),
                "DeltaProxy": float(latest.get("delta_proxy", np.nan)),
                **{f"C_{k}": v for k, v in confluence_signals.items()},
                **pv,
                **vcp,
                **sfp,
                **ipo,
                **fundamentals,
            }

            candidates.append(Candidate(symbol=symbol, score=total_score, price=row["Price"], details=row))
        except Exception:
            continue

    if not candidates:
        print("No candidates found with current thresholds.")
        return pd.DataFrame()

    df_results = pd.DataFrame([c.details for c in candidates])
    df_results = df_results.sort_values(by=["Score", "ConfluenceScore", "SetupScore", "RVol"], ascending=False)
    top = df_results.head(top_n).reset_index(drop=True)

    out_path = Path("Predicta_Top10.xlsx")
    export_excel(top=top, all_candidates=df_results, out_path=out_path)

    print("\nTOP 10 SWING CANDIDATES (next day watchlist)\n")
    cols = [
        "Symbol",
        "Score",
        "ConfluenceScore",
        "SetupScore",
        "FundBonus",
        "Price",
        "VCP",
        "SFP",
        "IPO_BASE",
        "FundamentalQualityScore",
        "AltmanZ",
        "BeneishM",
        "EPS_Est_Growth%",
        "RVol",
        "ADR%",
    ]
    cols = [c for c in cols if c in top.columns]
    with pd.option_context("display.max_columns", 200, "display.width", 240):
        print(top[cols])

    print(f"\nSaved report to {out_path.resolve()}")
    return top


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="NSE swing screener (confluence + setups + fundamentals)")
    p.add_argument("--universe-limit", type=int, default=500, help="How many NSE symbols to scan (0 = all)")
    p.add_argument("--min-score", type=int, default=6, help="Minimum confluence score (0-8)")
    p.add_argument("--period", type=str, default="1y", help="yfinance history period (e.g. 6mo, 1y, 2y)")
    p.add_argument("--interval", type=str, default="1d", help="yfinance interval (e.g. 1d)")
    p.add_argument("--top-n", type=int, default=10, help="Top N to export")
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run_full_system(
        universe_limit=args.universe_limit,
        min_confluence_score=args.min_score,
        period=args.period,
        interval=args.interval,
        top_n=args.top_n,
    )
    raise SystemExit(0)

import argparse
import datetime as dt
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import requests
import ta
import yfinance as yf
from tqdm import tqdm


CACHE_DIR = Path(".cache")
CACHE_DIR.mkdir(exist_ok=True)
NSE_EQUITY_LIST_CACHE = CACHE_DIR / "EQUITY_L.csv"


# ============================================================
# 1) NSE UNIVERSE
# ============================================================


def get_nse_stocks(cache_ttl_hours: int = 24) -> List[str]:
    """
    Returns NSE equity symbols as Yahoo tickers (e.g., RELIANCE.NS).
    Uses NSE archive CSV with a small local cache to reduce failures.
    """
    url = "https://archives.nseindia.com/content/equities/EQUITY_L.csv"
    headers = {
        "User-Agent": "Mozilla/5.0",
        "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
        "Accept-Language": "en-US,en;q=0.5",
        "Connection": "keep-alive",
    }

    def cache_is_fresh() -> bool:
        if not NSE_EQUITY_LIST_CACHE.exists():
            return False
        age = dt.datetime.now() - dt.datetime.fromtimestamp(NSE_EQUITY_LIST_CACHE.stat().st_mtime)
        return age.total_seconds() < cache_ttl_hours * 3600

    if not cache_is_fresh():
        resp = requests.get(url, headers=headers, timeout=30)
        resp.raise_for_status()
        NSE_EQUITY_LIST_CACHE.write_text(resp.text, encoding="utf-8")

    text = NSE_EQUITY_LIST_CACHE.read_text(encoding="utf-8", errors="ignore")
    lines = text.splitlines()
    if not lines:
        return []

    symbols: List[str] = []
    for line in lines[1:]:
        if not line.strip():
            continue
        sym = line.split(",")[0].strip().strip('"')
        if sym:
            symbols.append(f"{sym}.NS")
    return symbols


# ============================================================
# 2) TECHNICAL ENGINE (Predicta V4-ish)
# ============================================================


def add_technical_indicators(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    df["ema8"] = ta.trend.ema_indicator(df["Close"], 8)
    df["ema21"] = ta.trend.ema_indicator(df["Close"], 21)
    df["ema55"] = ta.trend.ema_indicator(df["Close"], 55)
    df["ema144"] = ta.trend.ema_indicator(df["Close"], 144)
    df["ema50"] = ta.trend.ema_indicator(df["Close"], 50)
    df["ema200"] = ta.trend.ema_indicator(df["Close"], 200)

    df["rsi"] = ta.momentum.rsi(df["Close"], 14)

    macd = ta.trend.MACD(df["Close"])
    df["macd"] = macd.macd()
    df["macd_signal"] = macd.macd_signal()

    df["adx"] = ta.trend.adx(df["High"], df["Low"], df["Close"], 14)
    df["stoch"] = ta.momentum.stoch(df["High"], df["Low"], df["Close"], 14)

    df["vol_ma20"] = df["Volume"].rolling(20).mean()
    df["vol_mult"] = df["Volume"] / df["vol_ma20"]

    df["atr"] = ta.volatility.average_true_range(df["High"], df["Low"], df["Close"], 14)
    df["atr_ma20"] = df["atr"].rolling(20).mean()
    df["atr_high"] = df["atr"] > df["atr_ma20"]

    # Price + Volume panel helpers
    df["adr20"] = (df["High"] - df["Low"]).rolling(20).mean()
    df["adrp20"] = (df["adr20"] / df["Close"]) * 100.0
    df["rvol50"] = df["Volume"] / df["Volume"].rolling(50).mean()
    df["obv"] = ta.volume.on_balance_volume(df["Close"], df["Volume"])

    # Daily "delta" approximation via CLV * volume
    hl_range = (df["High"] - df["Low"]).replace(0, np.nan)
    clv = ((df["Close"] - df["Low"]) - (df["High"] - df["Close"])) / hl_range
    df["clv"] = clv.fillna(0.0)
    df["delta_proxy"] = df["clv"] * df["Volume"]

    return df


def predicta_v4_confluence(latest: pd.Series) -> Tuple[int, Dict[str, Any]]:
    """
    Returns (subscore, signal_details) for:
    MACD, RSI, STOCH, VOLUME, DELTA, TREND, ADX, ATR
    """
    signals = {
        "MACD": bool(latest.get("macd", np.nan) > latest.get("macd_signal", np.nan)),
        "RSI": bool(latest.get("rsi", np.nan) >= 55),
        "STOCH": bool(latest.get("stoch", np.nan) >= 60),
        "VOLUME": bool(latest.get("vol_mult", np.nan) >= 1.2),
        "DELTA": bool(latest.get("delta_proxy", 0.0) > 0),
        "TREND": bool(latest.get("Close", np.nan) > latest.get("ema50", np.nan) > latest.get("ema200", np.nan)),
        "ADX": bool(latest.get("adx", np.nan) >= 20),
        "ATR": bool(bool(latest.get("atr_high", False))),
    }
    score = int(sum(1 for v in signals.values() if v))
    return score, signals


# ============================================================
# 3) PRICE + VOLUME STRENGTH PANEL
# ============================================================


def calculate_price_volume_metrics(df: pd.DataFrame) -> Dict[str, Any]:
    latest = df.iloc[-1]
    ema_distance = {
        "EMA8%": (latest["Close"] - latest["ema8"]) / latest["ema8"] * 100,
        "EMA21%": (latest["Close"] - latest["ema21"]) / latest["ema21"] * 100,
        "EMA55%": (latest["Close"] - latest["ema55"]) / latest["ema55"] * 100,
        "EMA144%": (latest["Close"] - latest["ema144"]) / latest["ema144"] * 100,
    }

    last30 = df.iloc[-30:]
    up_days = int((last30["Close"] > last30["Open"]).sum())
    down_days = int(len(last30) - up_days)

    high_52w = float(df["High"].tail(252).max()) if len(df) >= 252 else float(df["High"].max())
    dist_52w_high_pct = float((latest["Close"] / high_52w - 1.0) * 100.0) if high_52w else np.nan

    out: Dict[str, Any] = {}
    out.update(ema_distance)
    out.update(
        {
            "ADR": float(latest.get("adr20", np.nan)),
            "ADR%": float(latest.get("adrp20", np.nan)),
            "RVol": float(latest.get("rvol50", np.nan)),
            "U/D Days(30)": f"{up_days}/{down_days}",
            "Dist52WHigh%": dist_52w_high_pct,
        }
    )
    return out


# ============================================================
# 4) SETUP DETECTORS (VCP / IPO BASE / SWING FAILURE)
# ============================================================


def detect_vcp(df: pd.DataFrame, lookback: int = 60) -> Dict[str, Any]:
    """
    VCP (daily approximation):
    - average range% contracts across 3 segments
    - volume contracts across the same segments
    """
    if len(df) < lookback + 5:
        return {"VCP": False, "VCP_Reason": "not_enough_data"}

    w = df.tail(lookback).copy()
    w["range%"] = (w["High"] - w["Low"]) / w["Close"] * 100.0
    r1 = float(w["range%"].iloc[:20].mean())
    r2 = float(w["range%"].iloc[20:40].mean())
    r3 = float(w["range%"].iloc[40:60].mean())

    v1 = float(w["Volume"].iloc[:20].mean())
    v2 = float(w["Volume"].iloc[20:40].mean())
    v3 = float(w["Volume"].iloc[40:60].mean())

    range_contracting = (r1 > r2 > r3) and (r3 < 6.0)
    volume_contracting = (v1 > v2 > v3)

    return {
        "VCP": bool(range_contracting and volume_contracting),
        "VCP_R1": r1,
        "VCP_R2": r2,
        "VCP_R3": r3,
        "VCP_V1": v1,
        "VCP_V2": v2,
        "VCP_V3": v3,
    }


def detect_swing_failure(df: pd.DataFrame, lookback: int = 20) -> Dict[str, Any]:
    """
    Bullish SFP (daily approximation):
    - last candle makes lower low vs recent swing low
    - but closes back above that swing low
    """
    if len(df) < lookback + 2:
        return {"SFP": False, "SFP_Level": np.nan}

    recent = df.iloc[-(lookback + 1) : -1]
    swing_low = float(recent["Low"].min())
    last = df.iloc[-1]

    sfp = bool((last["Low"] < swing_low) and (last["Close"] > swing_low) and (last["Close"] > last["Open"]))
    return {"SFP": sfp, "SFP_Level": swing_low}


def detect_ipo_base(symbol: str) -> Dict[str, Any]:
    """
    IPO base needs listing date; Yahoo sometimes provides firstTradeDateEpochUtc.
    """
    try:
        info = yf.Ticker(symbol).info or {}
        first_trade = info.get("firstTradeDateEpochUtc")
        if not first_trade:
            return {"IPO_BASE": False, "IPO_Days": np.nan, "IPO_Reason": "missing_first_trade_date"}
        ipo_date = dt.datetime.utcfromtimestamp(int(first_trade)).date()
        ipo_days = (dt.date.today() - ipo_date).days
        return {"IPO_BASE": bool(ipo_days <= 365), "IPO_Days": ipo_days}
    except Exception:
        return {"IPO_BASE": False, "IPO_Days": np.nan, "IPO_Reason": "info_error"}


# ============================================================
# 5) FUNDAMENTALS (QUALITY + ALTMAN Z + BENEISH M + QoQ revenue)
# ============================================================


def safe_get(df: Optional[pd.DataFrame], row_name: str, col_idx: int = 0) -> Optional[float]:
    if df is None or df.empty or row_name not in df.index:
        return None
    try:
        v = df.loc[row_name].iloc[col_idx]
        if pd.isna(v):
            return None
        return float(v)
    except Exception:
        return None


def compute_altman_z(info: Dict[str, Any], financials: Optional[pd.DataFrame], balance: Optional[pd.DataFrame]) -> Optional[float]:
    """
    Altman Z-score (public manufacturing form).
    Returns None if statements are missing/incomplete.
    """
    ta_ = safe_get(balance, "Total Assets", 0)
    tl_ = safe_get(balance, "Total Liab", 0) or safe_get(balance, "Total Liabilities Net Minority Interest", 0)
    ca_ = safe_get(balance, "Total Current Assets", 0)
    cl_ = safe_get(balance, "Total Current Liabilities", 0)
    re_ = safe_get(balance, "Retained Earnings", 0)
    ebit_ = safe_get(financials, "Ebit", 0) or safe_get(financials, "EBIT", 0)
    sales_ = safe_get(financials, "Total Revenue", 0)

    mcap = info.get("marketCap")
    try:
        mcap = float(mcap) if mcap is not None else None
    except Exception:
        mcap = None

    if any(x is None for x in [ta_, tl_, ca_, cl_, re_, ebit_, sales_, mcap]):
        return None
    if ta_ == 0 or tl_ == 0:
        return None

    wc = ca_ - cl_
    z = (
        1.2 * (wc / ta_)
        + 1.4 * (re_ / ta_)
        + 3.3 * (ebit_ / ta_)
        + 0.6 * (mcap / tl_)
        + 1.0 * (sales_ / ta_)
    )
    return float(z)


def compute_beneish_m(financials: Optional[pd.DataFrame], balance: Optional[pd.DataFrame]) -> Optional[float]:
    """
    Beneish M-score (best-effort). Needs two annual periods.
    Returns None when required lines are missing.
    """
    if financials is None or balance is None or financials.empty or balance.empty:
        return None
    if financials.shape[1] < 2 or balance.shape[1] < 2:
        return None

    def g(row: str, idx: int) -> Optional[float]:
        return safe_get(financials, row, idx)

    def b(row: str, idx: int) -> Optional[float]:
        return safe_get(balance, row, idx)

    sales_t = g("Total Revenue", 0)
    sales_t1 = g("Total Revenue", 1)
    cogs_t = g("Cost Of Revenue", 0)
    cogs_t1 = g("Cost Of Revenue", 1)

    ar_t = b("Net Receivables", 0) or b("Accounts Receivable", 0)
    ar_t1 = b("Net Receivables", 1) or b("Accounts Receivable", 1)
    ta_t = b("Total Assets", 0)
    ta_t1 = b("Total Assets", 1)
    ppe_t = b("Property Plant Equipment", 0) or b("Property Plant And Equipment Net", 0)
    ppe_t1 = b("Property Plant Equipment", 1) or b("Property Plant And Equipment Net", 1)
    ca_t = b("Total Current Assets", 0)
    ca_t1 = b("Total Current Assets", 1)
    cl_t = b("Total Current Liabilities", 0)
    cl_t1 = b("Total Current Liabilities", 1)
    ltd_t = b("Long Term Debt", 0) or b("Long Term Debt And Capital Lease Obligation", 0)
    ltd_t1 = b("Long Term Debt", 1) or b("Long Term Debt And Capital Lease Obligation", 1)

    dep_t = g("Reconciled Depreciation", 0) or g("Depreciation", 0)
    dep_t1 = g("Reconciled Depreciation", 1) or g("Depreciation", 1)

    sga_t = g("Selling General Administrative", 0) or g("Selling General And Administration", 0)
    sga_t1 = g("Selling General Administrative", 1) or g("Selling General And Administration", 1)

    if any(v is None for v in [sales_t, sales_t1, ta_t, ta_t1]) or sales_t == 0 or sales_t1 == 0:
        return None

    dsri = (ar_t / sales_t) / (ar_t1 / sales_t1) if all(v is not None and v != 0 for v in [ar_t, sales_t, ar_t1, sales_t1]) else None
    gmi = (
        ((sales_t1 - cogs_t1) / sales_t1) / ((sales_t - cogs_t) / sales_t)
        if all(v is not None and v != 0 for v in [sales_t, sales_t1, cogs_t, cogs_t1])
        else None
    )
    aqi = (
        (1 - ((ca_t + ppe_t) / ta_t)) / (1 - ((ca_t1 + ppe_t1) / ta_t1))
        if all(v is not None and v != 0 for v in [ca_t, ppe_t, ta_t, ca_t1, ppe_t1, ta_t1])
        else None
    )
    sgi = sales_t / sales_t1
    depi = (
        (dep_t1 / (dep_t1 + ppe_t1)) / (dep_t / (dep_t + ppe_t))
        if all(v is not None and v != 0 for v in [dep_t, dep_t1, ppe_t, ppe_t1]) and (dep_t + ppe_t) != 0 and (dep_t1 + ppe_t1) != 0
        else None
    )
    sgai = (
        (sga_t / sales_t) / (sga_t1 / sales_t1)
        if all(v is not None and v != 0 for v in [sga_t, sales_t, sga_t1, sales_t1])
        else None
    )
    lvgi = (
        ((cl_t + ltd_t) / ta_t) / ((cl_t1 + ltd_t1) / ta_t1)
        if all(v is not None and v != 0 for v in [cl_t, ltd_t, ta_t, cl_t1, ltd_t1, ta_t1])
        else None
    )

    tata = 0.0  # best-effort; needs cashflow data for true value

    needed = [dsri, gmi, aqi, sgi, depi, sgai, lvgi]
    if any(v is None for v in needed):
        return None

    m = -4.84 + 0.92 * dsri + 0.528 * gmi + 0.404 * aqi + 0.892 * sgi + 0.115 * depi - 0.172 * sgai + 4.679 * tata - 0.327 * lvgi
    return float(m)


def get_fundamentals(symbol: str) -> Dict[str, Any]:
    stock = yf.Ticker(symbol)
    info = stock.info or {}

    financials = getattr(stock, "financials", None)
    balance = getattr(stock, "balance_sheet", None)
    q_fin = getattr(stock, "quarterly_financials", None)

    fundamentals: Dict[str, Any] = {
        "PE": info.get("trailingPE"),
        "ROE": info.get("returnOnEquity"),
        "ROA": info.get("returnOnAssets"),
        "DebtToEquity": info.get("debtToEquity"),
        "ProfitMargin": info.get("profitMargins"),
        "RevenueGrowth": info.get("revenueGrowth"),
        "GrossMargin": info.get("grossMargins"),
        "OperatingMargin": info.get("operatingMargins"),
        "EPS_Est_Growth%": (info.get("earningsGrowth") * 100.0) if info.get("earningsGrowth") is not None else None,
    }

    fundamentals["AltmanZ"] = compute_altman_z(info, financials, balance)
    fundamentals["BeneishM"] = compute_beneish_m(financials, balance)

    # Detailed QoQ Revenue % per quarter (keep last 6 quarters)
    if isinstance(q_fin, pd.DataFrame) and (not q_fin.empty) and ("Total Revenue" in q_fin.index):
        rev = q_fin.loc["Total Revenue"].astype(float)
        rev = rev.sort_index()
        qoq = rev.pct_change() * 100.0
        for col in rev.index[-6:]:
            label = pd.to_datetime(col).date().isoformat() if hasattr(col, "date") else str(col)
            fundamentals[f"Rev_{label}"] = float(rev[col])
            fundamentals[f"QoQ%_{label}"] = float(qoq[col]) if not pd.isna(qoq[col]) else None

    # Fundamental Quality Scorecard (v1)
    quality = 0
    if fundamentals.get("ROE") is not None and fundamentals["ROE"] >= 0.15:
        quality += 1
    if fundamentals.get("DebtToEquity") is not None and fundamentals["DebtToEquity"] <= 1.0:
        quality += 1
    if fundamentals.get("OperatingMargin") is not None and fundamentals["OperatingMargin"] >= 0.12:
        quality += 1
    if fundamentals.get("RevenueGrowth") is not None and fundamentals["RevenueGrowth"] >= 0.10:
        quality += 1
    if fundamentals.get("ProfitMargin") is not None and fundamentals["ProfitMargin"] >= 0.08:
        quality += 1
    fundamentals["FundamentalQualityScore"] = quality

    return fundamentals


# ============================================================
# 6) RUNNER + REPORT
# ============================================================


@dataclass
class Candidate:
    symbol: str
    score: int
    price: float
    details: Dict[str, Any]


def export_excel(top: pd.DataFrame, all_candidates: pd.DataFrame, out_path: Path) -> None:
    """
    Writes an Excel report if openpyxl is installed; otherwise falls back to CSV.
    """
    try:
        with pd.ExcelWriter(out_path, engine="openpyxl") as writer:
            top.to_excel(writer, sheet_name="Top10", index=False)
            all_candidates.to_excel(writer, sheet_name="AllCandidates", index=False)
    except Exception:
        out_csv = out_path.with_suffix(".csv")
        top.to_csv(out_csv, index=False)
        print(f"Excel export unavailable; saved CSV to {out_csv.resolve()}")


def run_full_system(
    universe_limit: int = 500,
    min_confluence_score: int = 6,
    period: str = "1y",
    interval: str = "1d",
    top_n: int = 10,
) -> pd.DataFrame:
    stocks = get_nse_stocks()
    if universe_limit and universe_limit > 0:
        stocks = stocks[:universe_limit]

    candidates: List[Candidate] = []
    print(f"Scanning {len(stocks)} NSE stocks...")

    for symbol in tqdm(stocks):
        try:
            df = yf.download(symbol, period=period, interval=interval, progress=False, auto_adjust=False)
            if df is None or df.empty:
                continue

            df = df.rename(columns={c: c.title() for c in df.columns})
            needed = {"Open", "High", "Low", "Close", "Volume"}
            if not needed.issubset(df.columns):
                continue

            if len(df) < 220:
                continue

            df = add_technical_indicators(df)
            latest = df.iloc[-1]

            confluence_score, confluence_signals = predicta_v4_confluence(latest)
            if confluence_score < min_confluence_score:
                continue

            pv = calculate_price_volume_metrics(df)
            vcp = detect_vcp(df)
            sfp = detect_swing_failure(df)
            ipo = detect_ipo_base(symbol)
            fundamentals = get_fundamentals(symbol)

            row: Dict[str, Any] = {
                "Symbol": symbol,
                "Score": confluence_score,
                "Price": float(latest["Close"]),
                "RSI": float(latest.get("rsi", np.nan)),
                "ADX": float(latest.get("adx", np.nan)),
                "Stoch": float(latest.get("stoch", np.nan)),
                "VolMult": float(latest.get("vol_mult", np.nan)),
                "ATR": float(latest.get("atr", np.nan)),
                "DeltaProxy": float(latest.get("delta_proxy", np.nan)),
                **{f"C_{k}": v for k, v in confluence_signals.items()},
                **pv,
                **vcp,
                **sfp,
                **ipo,
                **fundamentals,
            }

            candidates.append(Candidate(symbol=symbol, score=confluence_score, price=row["Price"], details=row))
        except Exception:
            continue

    if not candidates:
        print("No candidates found with current thresholds.")
        return pd.DataFrame()

    df_results = pd.DataFrame([c.details for c in candidates])
    df_results = df_results.sort_values(by=["Score", "FundamentalQualityScore", "RVol"], ascending=False)
    top = df_results.head(top_n).reset_index(drop=True)

    out_path = Path("Predicta_Top10.xlsx")
    export_excel(top=top, all_candidates=df_results, out_path=out_path)

    print("\nTOP 10 SWING CANDIDATES (next day watchlist)\n")
    cols = [
        "Symbol",
        "Score",
        "Price",
        "VCP",
        "SFP",
        "IPO_BASE",
        "FundamentalQualityScore",
        "AltmanZ",
        "BeneishM",
        "EPS_Est_Growth%",
        "RVol",
        "ADR%",
    ]
    cols = [c for c in cols if c in top.columns]
    with pd.option_context("display.max_columns", 200, "display.width", 240):
        print(top[cols])

    print(f"\nSaved report to {out_path.resolve()}")
    return top


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="NSE swing screener (confluence + setups + fundamentals)")
    p.add_argument("--universe-limit", type=int, default=500, help="How many NSE symbols to scan (0 = all)")
    p.add_argument("--min-score", type=int, default=6, help="Minimum confluence score (0-8)")
    p.add_argument("--period", type=str, default="1y", help="yfinance history period (e.g. 6mo, 1y, 2y)")
    p.add_argument("--interval", type=str, default="1d", help="yfinance interval (e.g. 1d)")
    p.add_argument("--top-n", type=int, default=10, help="Top N to export")
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run_full_system(
        universe_limit=args.universe_limit,
        min_confluence_score=args.min_score,
        period=args.period,
        interval=args.interval,
        top_n=args.top_n,
    )

import argparse
import datetime as dt
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import requests
import ta
import yfinance as yf
from tqdm import tqdm


CACHE_DIR = Path(".cache")
CACHE_DIR.mkdir(exist_ok=True)
NSE_EQUITY_LIST_CACHE = CACHE_DIR / "EQUITY_L.csv"


# ============================================================
# 1) NSE UNIVERSE
# ============================================================


def get_nse_stocks(cache_ttl_hours: int = 24) -> List[str]:
    """
    Returns NSE equity symbols as Yahoo tickers (e.g., RELIANCE.NS).
    Uses NSE archive CSV with a small local cache to reduce failures.
    """
    url = "https://archives.nseindia.com/content/equities/EQUITY_L.csv"
    headers = {
        "User-Agent": "Mozilla/5.0",
        "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
        "Accept-Language": "en-US,en;q=0.5",
        "Connection": "keep-alive",
    }

    def cache_is_fresh() -> bool:
        if not NSE_EQUITY_LIST_CACHE.exists():
            return False
        age = dt.datetime.now() - dt.datetime.fromtimestamp(NSE_EQUITY_LIST_CACHE.stat().st_mtime)
        return age.total_seconds() < cache_ttl_hours * 3600

    if not cache_is_fresh():
        resp = requests.get(url, headers=headers, timeout=30)
        resp.raise_for_status()
        NSE_EQUITY_LIST_CACHE.write_text(resp.text, encoding="utf-8")

    text = NSE_EQUITY_LIST_CACHE.read_text(encoding="utf-8", errors="ignore")
    lines = text.splitlines()
    if not lines:
        return []

    # CSV header: SYMBOL,NAME OF COMPANY, SERIES, ...
    symbols: List[str] = []
    for line in lines[1:]:
        if not line.strip():
            continue
        sym = line.split(",")[0].strip().strip('"')
        if not sym:
            continue
        symbols.append(f"{sym}.NS")
    return symbols


# ============================================================
# 2) TECHNICAL ENGINE (Predicta V4-ish)
# ============================================================


def add_technical_indicators(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    df["ema8"] = ta.trend.ema_indicator(df["Close"], 8)
    df["ema21"] = ta.trend.ema_indicator(df["Close"], 21)
    df["ema55"] = ta.trend.ema_indicator(df["Close"], 55)
    df["ema144"] = ta.trend.ema_indicator(df["Close"], 144)
    df["ema50"] = ta.trend.ema_indicator(df["Close"], 50)
    df["ema200"] = ta.trend.ema_indicator(df["Close"], 200)

    df["rsi"] = ta.momentum.rsi(df["Close"], 14)

    macd = ta.trend.MACD(df["Close"])
    df["macd"] = macd.macd()
    df["macd_signal"] = macd.macd_signal()

    df["adx"] = ta.trend.adx(df["High"], df["Low"], df["Close"], 14)
    df["stoch"] = ta.momentum.stoch(df["High"], df["Low"], df["Close"], 14)

    df["vol_ma20"] = df["Volume"].rolling(20).mean()
    df["vol_mult"] = df["Volume"] / df["vol_ma20"]

    df["atr"] = ta.volatility.average_true_range(df["High"], df["Low"], df["Close"], 14)
    df["atr_ma20"] = df["atr"].rolling(20).mean()
    df["atr_high"] = df["atr"] > df["atr_ma20"]

    # Extra helpers for panels
    df["adr20"] = (df["High"] - df["Low"]).rolling(20).mean()
    df["adrp20"] = (df["adr20"] / df["Close"]) * 100.0
    df["rvol50"] = df["Volume"] / df["Volume"].rolling(50).mean()
    df["obv"] = ta.volume.on_balance_volume(df["Close"], df["Volume"])

    # Proxy "delta" with CLV * volume (daily approximation)
    hl_range = (df["High"] - df["Low"]).replace(0, np.nan)
    clv = ((df["Close"] - df["Low"]) - (df["High"] - df["Close"])) / hl_range
    df["clv"] = clv.fillna(0.0)
    df["delta_proxy"] = df["clv"] * df["Volume"]

    return df


def predicta_v4_confluence(latest: pd.Series) -> Tuple[int, Dict[str, Any]]:
    """
    Returns (subscore, signal_details).
    Interprets your "Predicta V4 Confluence" as 8 boolean checks.
    """
    signals = {
        "MACD": bool(latest.get("macd", np.nan) > latest.get("macd_signal", np.nan)),
        "RSI": bool(latest.get("rsi", np.nan) >= 55),
        "STOCH": bool(latest.get("stoch", np.nan) >= 60),
        "VOLUME": bool(latest.get("vol_mult", np.nan) >= 1.2),
        "DELTA": bool(latest.get("delta_proxy", 0.0) > 0),
        "TREND": bool(latest.get("Close", np.nan) > latest.get("ema50", np.nan) > latest.get("ema200", np.nan)),
        "ADX": bool(latest.get("adx", np.nan) >= 20),
        "ATR": bool(bool(latest.get("atr_high", False))),
    }
    score = int(sum(1 for v in signals.values() if v))
    return score, signals


# ============================================================
# 3) PRICE + VOLUME STRENGTH PANEL
# ============================================================


def calculate_price_volume_metrics(df: pd.DataFrame) -> Dict[str, Any]:
    latest = df.iloc[-1]
    ema_distance = {
        "EMA8%": (latest["Close"] - latest["ema8"]) / latest["ema8"] * 100,
        "EMA21%": (latest["Close"] - latest["ema21"]) / latest["ema21"] * 100,
        "EMA55%": (latest["Close"] - latest["ema55"]) / latest["ema55"] * 100,
        "EMA144%": (latest["Close"] - latest["ema144"]) / latest["ema144"] * 100,
    }

    rvol = float(latest.get("rvol50", np.nan))
    adr = float(latest.get("adr20", np.nan))
    adrp = float(latest.get("adrp20", np.nan))

    last30 = df.iloc[-30:]
    up_days = int((last30["Close"] > last30["Open"]).sum())
    down_days = int(len(last30) - up_days)

    high_52w = float(df["High"].tail(252).max()) if len(df) >= 252 else float(df["High"].max())
    dist_52w_high_pct = float((latest["Close"] / high_52w - 1.0) * 100.0) if high_52w else np.nan

    out: Dict[str, Any] = {}
    out.update(ema_distance)
    out.update(
        {
            "ADR": adr,
            "ADR%": adrp,
            "RVol": rvol,
            "U/D Days(30)": f"{up_days}/{down_days}",
            "Dist52WHigh%": dist_52w_high_pct,
        }
    )
    return out


# ============================================================
# 4) SETUP DETECTORS (VCP / IPO BASE / SWING FAILURE)
# ============================================================


def detect_vcp(df: pd.DataFrame, lookback: int = 60) -> Dict[str, Any]:
    """
    Volatility Contraction Pattern (daily approximation):
    - last 3 swing ranges contract (high-low) and volume trends down
    Returns flags and some supporting stats.
    """
    if len(df) < lookback + 5:
        return {"VCP": False, "VCP_Reason": "not_enough_data"}

    w = df.tail(lookback).copy()
    w["range"] = (w["High"] - w["Low"]) / w["Close"] * 100.0
    r1 = w["range"].iloc[:20].mean()
    r2 = w["range"].iloc[20:40].mean()
    r3 = w["range"].iloc[40:60].mean()

    v1 = w["Volume"].iloc[:20].mean()
    v2 = w["Volume"].iloc[20:40].mean()
    v3 = w["Volume"].iloc[40:60].mean()

    range_contracting = (r1 > r2 > r3) and (r3 < 6.0)  # customizable
    volume_contracting = (v1 > v2 > v3)

    return {
        "VCP": bool(range_contracting and volume_contracting),
        "VCP_R1": float(r1),
        "VCP_R2": float(r2),
        "VCP_R3": float(r3),
        "VCP_V1": float(v1),
        "VCP_V2": float(v2),
        "VCP_V3": float(v3),
    }


def detect_swing_failure(df: pd.DataFrame, lookback: int = 20) -> Dict[str, Any]:
    """
    Swing Failure Pattern (SFP) bullish (daily approximation):
    - today makes a lower low vs recent swing low, but closes back above it.
    """
    if len(df) < lookback + 2:
        return {"SFP": False, "SFP_Level": np.nan}

    recent = df.iloc[-(lookback + 1) : -1]
    swing_low = float(recent["Low"].min())
    last = df.iloc[-1]

    sfp = bool((last["Low"] < swing_low) and (last["Close"] > swing_low) and (last["Close"] > last["Open"]))
    return {"SFP": sfp, "SFP_Level": swing_low}


def detect_ipo_base(symbol: str) -> Dict[str, Any]:
    """
    IPO base requires IPO/listing date. Yahoo sometimes provides firstTradeDateEpochUtc.
    If available: IPO_BASE if listed <= 365d and price base (tight range last 20d).
    """
    try:
        info = yf.Ticker(symbol).info
        first_trade = info.get("firstTradeDateEpochUtc")
        if not first_trade:
            return {"IPO_BASE": False, "IPO_Days": np.nan, "IPO_Reason": "missing_first_trade_date"}
        ipo_date = dt.datetime.utcfromtimestamp(int(first_trade)).date()
        ipo_days = (dt.date.today() - ipo_date).days
        return {"IPO_BASE": bool(ipo_days <= 365), "IPO_Days": ipo_days}
    except Exception:
        return {"IPO_BASE": False, "IPO_Days": np.nan, "IPO_Reason": "info_error"}


# ============================================================
# 5) FUNDAMENTALS (QUALITY + ALTMAN Z + BENEISH M + QoQ revenue)
# ============================================================


def safe_get(df: pd.DataFrame, row_name: str, col_idx: int = 0) -> Optional[float]:
    if df is None or df.empty:
        return None
    if row_name not in df.index:
        return None
    try:
        v = df.loc[row_name].iloc[col_idx]
        if pd.isna(v):
            return None
        return float(v)
    except Exception:
        return None


def compute_altman_z(info: Dict[str, Any], financials: pd.DataFrame, balance: pd.DataFrame) -> Optional[float]:
    """
    Classic Altman Z-score (public manufacturing form) using annual statements if present:
    Z = 1.2*(WC/TA) + 1.4*(RE/TA) + 3.3*(EBIT/TA) + 0.6*(MVE/TL) + 1.0*(Sales/TA)
    Yahoo fields can be inconsistent; returns None if missing.
    """
    ta_ = safe_get(balance, "Total Assets", 0)
    tl_ = safe_get(balance, "Total Liab", 0) or safe_get(balance, "Total Liabilities Net Minority Interest", 0)
    ca_ = safe_get(balance, "Total Current Assets", 0)
    cl_ = safe_get(balance, "Total Current Liabilities", 0)
    re_ = safe_get(balance, "Retained Earnings", 0)
    ebit_ = safe_get(financials, "Ebit", 0) or safe_get(financials, "EBIT", 0)
    sales_ = safe_get(financials, "Total Revenue", 0)

    mcap = info.get("marketCap")
    if mcap is not None:
        try:
            mcap = float(mcap)
        except Exception:
            mcap = None

    if any(x is None for x in [ta_, tl_, ca_, cl_, re_, ebit_, sales_, mcap]):
        return None
    if ta_ == 0 or tl_ == 0:
        return None

    wc = ca_ - cl_
    z = (
        1.2 * (wc / ta_)
        + 1.4 * (re_ / ta_)
        + 3.3 * (ebit_ / ta_)
        + 0.6 * (mcap / tl_)
        + 1.0 * (sales_ / ta_)
    )
    return float(z)


def compute_beneish_m(financials: pd.DataFrame, balance: pd.DataFrame) -> Optional[float]:
    """
    Beneish M-score requires 2 periods of data.
    Uses best-effort mapping from Yahoo's statement rows. Returns None if missing.
    """
    if financials is None or balance is None or financials.empty or balance.empty:
        return None
    if financials.shape[1] < 2 or balance.shape[1] < 2:
        return None

    def g(row: str, idx: int) -> Optional[float]:
        return safe_get(financials, row, idx)

    def b(row: str, idx: int) -> Optional[float]:
        return safe_get(balance, row, idx)

    # Two most recent annual columns: 0 (latest), 1 (prior)
    sales_t = g("Total Revenue", 0)
    sales_t1 = g("Total Revenue", 1)
    cogs_t = g("Cost Of Revenue", 0)
    cogs_t1 = g("Cost Of Revenue", 1)
    # Net income / cashflow are used for TATA; Yahoo often lacks consistent cashflow rows for NSE.
    # We treat TATA as 0.0 when unavailable (best-effort).

    ar_t = b("Net Receivables", 0) or b("Accounts Receivable", 0)
    ar_t1 = b("Net Receivables", 1) or b("Accounts Receivable", 1)
    ta_t = b("Total Assets", 0)
    ta_t1 = b("Total Assets", 1)
    ppe_t = b("Property Plant Equipment", 0) or b("Property Plant And Equipment Net", 0)
    ppe_t1 = b("Property Plant Equipment", 1) or b("Property Plant And Equipment Net", 1)
    ca_t = b("Total Current Assets", 0)
    ca_t1 = b("Total Current Assets", 1)
    cl_t = b("Total Current Liabilities", 0)
    cl_t1 = b("Total Current Liabilities", 1)
    ltd_t = b("Long Term Debt", 0) or b("Long Term Debt And Capital Lease Obligation", 0)
    ltd_t1 = b("Long Term Debt", 1) or b("Long Term Debt And Capital Lease Obligation", 1)

    # Depreciation is often missing in Yahoo annual financials; DEPI may be unavailable.
    dep_t = g("Reconciled Depreciation", 0) or g("Depreciation", 0)
    dep_t1 = g("Reconciled Depreciation", 1) or g("Depreciation", 1)

    sga_t = g("Selling General Administrative", 0) or g("Selling General And Administration", 0)
    sga_t1 = g("Selling General Administrative", 1) or g("Selling General And Administration", 1)

    if any(x is None for x in [sales_t, sales_t1, ta_t, ta_t1]) or sales_t1 == 0 or sales_t == 0:
        return None

    # Indices
    dsri = (ar_t / sales_t) / (ar_t1 / sales_t1) if all(v is not None and v != 0 for v in [ar_t, sales_t, ar_t1, sales_t1]) else None
    gmi = (
        ((sales_t1 - cogs_t1) / sales_t1) / ((sales_t - cogs_t) / sales_t)
        if all(v is not None and v != 0 for v in [sales_t, sales_t1, cogs_t, cogs_t1])
        else None
    )
    aqi = (
        (1 - ((ca_t + ppe_t) / ta_t)) / (1 - ((ca_t1 + ppe_t1) / ta_t1))
        if all(v is not None and v != 0 for v in [ca_t, ppe_t, ta_t, ca_t1, ppe_t1, ta_t1])
        else None
    )
    sgi = sales_t / sales_t1
    depi = (
        (dep_t1 / (dep_t1 + ppe_t1)) / (dep_t / (dep_t + ppe_t))
        if all(v is not None and v != 0 for v in [dep_t, dep_t1, ppe_t, ppe_t1]) and (dep_t + ppe_t) != 0 and (dep_t1 + ppe_t1) != 0
        else None
    )
    sgai = (
        (sga_t / sales_t) / (sga_t1 / sales_t1)
        if all(v is not None and v != 0 for v in [sga_t, sales_t, sga_t1, sales_t1])
        else None
    )
    lvgi = (
        ((cl_t + ltd_t) / ta_t) / ((cl_t1 + ltd_t1) / ta_t1)
        if all(v is not None and v != 0 for v in [cl_t, ltd_t, ta_t, cl_t1, ltd_t1, ta_t1])
        else None
    )

    # TATA needs operating cashflow; skip if unknown
    tata = 0.0

    needed = [dsri, gmi, aqi, sgi, depi, sgai, lvgi]
    if any(v is None for v in needed):
        return None

    m = (
        -4.84
        + 0.92 * dsri
        + 0.528 * gmi
        + 0.404 * aqi
        + 0.892 * sgi
        + 0.115 * depi
        - 0.172 * sgai
        + 4.679 * (tata if tata is not None else 0.0)
        - 0.327 * lvgi
    )
    return float(m)


def get_fundamentals(symbol: str) -> Dict[str, Any]:
    """
    Pulls best-effort fundamentals from Yahoo.
    Notes:
    - Many NSE tickers have partial statement coverage on Yahoo.
    - Altman/Beneish will be None when inputs are missing.
    """
    stock = yf.Ticker(symbol)
    info = stock.info or {}

    financials = getattr(stock, "financials", None)
    balance = getattr(stock, "balance_sheet", None)
    q_fin = getattr(stock, "quarterly_financials", None)

    fundamentals: Dict[str, Any] = {
        "PE": info.get("trailingPE"),
        "ROE": info.get("returnOnEquity"),
        "ROA": info.get("returnOnAssets"),
        "DebtToEquity": info.get("debtToEquity"),
        "ProfitMargin": info.get("profitMargins"),
        "RevenueGrowth": info.get("revenueGrowth"),
        "GrossMargin": info.get("grossMargins"),
        "OperatingMargin": info.get("operatingMargins"),
        # EPS estimate growth is not consistently available; use earningsGrowth when present
        "EPS_Est_Growth%": (info.get("earningsGrowth") * 100.0) if info.get("earningsGrowth") is not None else None,
    }

    fundamentals["AltmanZ"] = compute_altman_z(info, financials, balance) if isinstance(financials, pd.DataFrame) and isinstance(balance, pd.DataFrame) else None
    fundamentals["BeneishM"] = compute_beneish_m(financials, balance) if isinstance(financials, pd.DataFrame) and isinstance(balance, pd.DataFrame) else None

    # QoQ Revenue % per quarter (cleanly structured)
    qoq_rows: Dict[str, Any] = {}
    if isinstance(q_fin, pd.DataFrame) and (not q_fin.empty) and ("Total Revenue" in q_fin.index):
        rev = q_fin.loc["Total Revenue"].sort_index()  # chronological by column label
        rev = rev.astype(float)
        qoq = rev.pct_change() * 100.0
        # Keep last 6 quarters to avoid very wide outputs
        for col in rev.index[-6:]:
            label = pd.to_datetime(col).date().isoformat() if hasattr(col, "date") else str(col)
            qoq_rows[f"Rev_{label}"] = float(rev[col])
            qoq_rows[f"QoQ%_{label}"] = float(qoq[col]) if not pd.isna(qoq[col]) else None

    fundamentals.update(qoq_rows)

    # Fundamental quality scorecard (simple v1 score)
    quality = 0
    if fundamentals.get("ROE") is not None and fundamentals["ROE"] >= 0.15:
        quality += 1
    if fundamentals.get("DebtToEquity") is not None and fundamentals["DebtToEquity"] <= 1.0:
        quality += 1
    if fundamentals.get("OperatingMargin") is not None and fundamentals["OperatingMargin"] >= 0.12:
        quality += 1
    if fundamentals.get("RevenueGrowth") is not None and fundamentals["RevenueGrowth"] >= 0.10:
        quality += 1
    if fundamentals.get("ProfitMargin") is not None and fundamentals["ProfitMargin"] >= 0.08:
        quality += 1
    fundamentals["FundamentalQualityScore"] = quality

    return fundamentals


# ============================================================
# 6) MASTER SCORING + REPORT
# ============================================================


@dataclass
class Candidate:
    symbol: str
    score: int
    price: float
    details: Dict[str, Any]


def run_full_system(
    universe_limit: int = 500,
    min_confluence_score: int = 6,
    period: str = "1y",
    interval: str = "1d",
    top_n: int = 10,
    fundamentals_for_candidates_only: bool = True,
) -> pd.DataFrame:
    stocks = get_nse_stocks()
    if universe_limit and universe_limit > 0:
        stocks = stocks[:universe_limit]

    candidates: List[Candidate] = []
    print(f"Scanning {len(stocks)} NSE stocks...")

    for symbol in tqdm(stocks):
        try:
            df = yf.download(symbol, period=period, interval=interval, progress=False, auto_adjust=False)
            if df is None or df.empty:
                continue

            # yfinance sometimes returns columns with lowercase names for some endpoints; normalize
            df = df.rename(columns={c: c.title() for c in df.columns})
            needed = {"Open", "High", "Low", "Close", "Volume"}
            if not needed.issubset(df.columns):
                continue

            if len(df) < 220:
                continue

            df = add_technical_indicators(df)
            latest = df.iloc[-1]
            confluence_score, confluence_signals = predicta_v4_confluence(latest)

            if confluence_score < min_confluence_score:
                continue

            pv = calculate_price_volume_metrics(df)
            vcp = detect_vcp(df)
            sfp = detect_swing_failure(df)
            ipo = detect_ipo_base(symbol)

            row: Dict[str, Any] = {
                "Symbol": symbol,
                "Score": confluence_score,
                "Price": float(latest["Close"]),
                "RSI": float(latest.get("rsi", np.nan)),
                "ADX": float(latest.get("adx", np.nan)),
                "Stoch": float(latest.get("stoch", np.nan)),
                "VolMult": float(latest.get("vol_mult", np.nan)),
                "ATR": float(latest.get("atr", np.nan)),
                "DeltaProxy": float(latest.get("delta_proxy", np.nan)),
                **{f"C_{k}": v for k, v in confluence_signals.items()},
                **pv,
                **vcp,
                **sfp,
                **ipo,
            }

            # Pull fundamentals only after passing technical gate (speed)
            if fundamentals_for_candidates_only:
                fundamentals = get_fundamentals(symbol)
                row.update(fundamentals)

            candidates.append(Candidate(symbol=symbol, score=confluence_score, price=row["Price"], details=row))
        except Exception:
            continue

    if not candidates:
        return pd.DataFrame()

    df_results = pd.DataFrame([c.details for c in candidates])
    df_results = df_results.sort_values(by=["Score", "FundamentalQualityScore", "RVol"], ascending=False)

    top = df_results.head(top_n).reset_index(drop=True)

    # Write a multi-sheet Excel report
    out_path = Path("Predicta_Top10.xlsx")
    with pd.ExcelWriter(out_path, engine="openpyxl") as writer:
        top.to_excel(writer, sheet_name="Top10", index=False)
        df_results.to_excel(writer, sheet_name="AllCandidates", index=False)

    print("\nTOP 10 SWING CANDIDATES (next day watchlist)\n")
    with pd.option_context("display.max_columns", 200, "display.width", 240):
        print(top[["Symbol", "Score", "Price", "VCP", "SFP", "IPO_BASE", "FundamentalQualityScore", "AltmanZ", "BeneishM", "RVol", "ADR%"]])
    print(f"\nSaved report to {out_path.resolve()}")

    return top


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="NSE swing screener (technical confluence + setups + fundamentals)")
    p.add_argument("--universe-limit", type=int, default=500, help="How many NSE symbols to scan (0 = all)")
    p.add_argument("--min-score", type=int, default=6, help="Minimum Predicta V4 confluence score")
    p.add_argument("--period", type=str, default="1y", help="yfinance history period (e.g. 6mo, 1y, 2y)")
    p.add_argument("--interval", type=str, default="1d", help="yfinance interval (e.g. 1d)")
    p.add_argument("--top-n", type=int, default=10, help="Top N to export")
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run_full_system(
        universe_limit=args.universe_limit,
        min_confluence_score=args.min_score,
        period=args.period,
        interval=args.interval,
        top_n=args.top_n,
    )

'''

