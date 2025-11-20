"""
indiana_tornado_scraper.py

Scrapes tornado data from https://www.weather.gov/ind/tornadostats for all counties,
extracts EF ratings, converts to Wind_mph, keeps date, injuries, fatalities,
and outputs a clean CSV ready for EVT modeling.

Requirements:
    pip install requests beautifulsoup4 pandas lxml python-dateutil
"""

import time
import re
from urllib.parse import urljoin
import requests
from bs4 import BeautifulSoup
import pandas as pd
from dateutil import parser as dateparser

BASE = "https://www.weather.gov/ind/tornadostats"
OUTFILE = "indiana_all_tornadoes.csv"
HEADERS = {
    "User-Agent": "Mozilla/5.0 (compatible; TornadoScraper/1.0; +https://example.com)"
}

EF_TO_MPH = {
    0: (65, 85),
    1: (86, 110),
    2: (111, 135),
    3: (136, 165),
    4: (166, 200),
    5: (201, 300),
}
def ef_to_mph_val(ef):
    try:
        ef = int(ef)
    except Exception:
        return None
    if ef in EF_TO_MPH:
        lo, hi = EF_TO_MPH[ef]
        return (lo + hi) / 2.0
    return None

def safe_get(url, session=None, max_tries=3, backoff=1.0):
    s = session or requests
    for i in range(max_tries):
        try:
            r = s.get(url, headers=HEADERS, timeout=20)
            r.raise_for_status()
            return r.text
        except Exception:
            time.sleep(backoff * (2 ** i))
    raise RuntimeError(f"Failed to fetch {url}")

def find_county_links(html):
    soup = BeautifulSoup(html, "lxml")
    links = set()
    for a in soup.find_all("a", href=True):
        href = a["href"]
        if "tornado" in href.lower() or "tornadostats" in href.lower():
            full = urljoin(BASE, href)
            links.add(full)
    return sorted(links)

def parse_table(df, source_url):
    """
    Normalize one table:
    - Ensure Date
    - EF/Wind_mph
    - Injuries/Fatalities if present
    """
    df = df.dropna(how='all').dropna(axis=1, how='all')
    df.columns = [str(c).strip() for c in df.columns]

    # Find columns
    date_col = next((c for c in df.columns if re.search(r'date|day|begin', c, re.I)), None)
    ef_col = next((c for c in df.columns if re.search(r'ef|f-scale|f rating', c, re.I)), None)
    inj_col = next((c for c in df.columns if re.search(r'injur', c, re.I)), None)
    fat_col = next((c for c in df.columns if re.search(r'fat', c, re.I)), None)
    county_col = next((c for c in df.columns if re.search(r'county', c, re.I)), None)
    location_col = next((c for c in df.columns if re.search(r'location|city|town', c, re.I)), None)

    df_out = pd.DataFrame()
    # Date parsing
    if date_col:
        df_out["Date"] = pd.to_datetime(df[date_col], errors='coerce')
    else:
        df_out["Date"] = pd.NaT

    # EF -> Wind
    if ef_col:
        df_out["EF"] = pd.to_numeric(df[ef_col], errors="coerce")
        df_out["Wind_mph"] = df_out["EF"].apply(lambda x: ef_to_mph_val(x))
    else:
        df_out["EF"] = pd.NA
        df_out["Wind_mph"] = pd.NA

    # Injuries/fatalities
    df_out["Injuries"] = pd.to_numeric(df[inj_col], errors="coerce") if inj_col else pd.NA
    df_out["Fatalities"] = pd.to_numeric(df[fat_col], errors="coerce") if fat_col else pd.NA

    # County/location
    df_out["County"] = df[county_col] if county_col else pd.NA
    df_out["Location"] = df[location_col] if location_col else pd.NA

    df_out["Source Page"] = source_url

    # Keep only rows with either Date or Wind_mph
    df_out = df_out[df_out["Date"].notna() | df_out["Wind_mph"].notna()]

    return df_out

def scrape_all_counties():
    session = requests.Session()
    main_html = safe_get(BASE, session=session)
    county_links = find_county_links(main_html)

    all_dfs = []
    for link in county_links:
        try:
            html = safe_get(link, session=session)
        except Exception as e:
            print(f"Failed {link}: {e}")
            continue

        try:
            tables = pd.read_html(html)
        except Exception:
            continue

        for t in tables:
            df_clean = parse_table(t, link)
            if not df_clean.empty:
                all_dfs.append(df_clean)
        time.sleep(0.8)

    if not all_dfs:
        raise RuntimeError("No tables extracted!")

    full = pd.concat(all_dfs, ignore_index=True)
    full.to_csv(OUTFILE, index=False)
    print(f"Saved {OUTFILE} with {len(full)} rows")
    if full["Date"].notna().sum() > 0:
        print("Date range:", full["Date"].min(), "to", full["Date"].max())
    return full

if __name__ == "__main__":
    scrape_all_counties()
