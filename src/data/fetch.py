"""
Data Ingestion and Parsing Module for CPCB / CECB Air Quality Data.
Supports parsing standard 15-minute average Excel exports (.xlsx) and CSV files.
Implements DP-1 (Ingestion) and DP-2 (Parsing).
"""

from dataclasses import dataclass, field
import logging
from pathlib import Path
import re
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


# Standard parameter name normalization map
PARAM_NORMALIZATION = {
    r"^pm\s*2\.?5.*": "PM2.5",
    r"^pm\s*10.*": "PM10",
    r"^no\b(?!2).*": "NO",
    r"^no2.*": "NO2",
    r"^so2.*": "SO2",
    r"^co\b.*": "CO",
    r"^ozone.*|^o3.*": "OZONE",
    r"^rh.*|^relative\s*humidity.*": "RH",
    r"^at.*|^temp.*|^ambient\s*temperature.*": "AT",
    r"^ws.*|^wind\s*speed.*": "WS",
    r"^wd.*|^wind\s*dir.*": "WD",
    r"^sr.*|^solar\s*rad.*": "SR",
    r"^nh3.*|^ammonia.*": "NH3",
    r"^rf.*|^rain.*": "RF",
    r"^bp.*|^barometric.*": "BP",
    r"^mh.*": "MH",
}


@dataclass
class StationMetadata:
    """Metadata extracted from CPCB / CECB file headers."""
    station_name: str
    city: str = "Korba"
    state: str = "Chhattisgarh"
    avg_period: str = "15 Minute"
    raw_file_path: Optional[str] = None
    parameters_found: List[str] = field(default_factory=list)


def normalize_column_name(col_name: str) -> str:
    """Normalize messy portal column headers into clean standardized feature names."""
    cleaned = str(col_name).strip()
    # Check against known regex patterns
    for pattern, std_name in PARAM_NORMALIZATION.items():
        if re.search(pattern, cleaned, re.IGNORECASE):
            return std_name
    return cleaned


def parse_cpcb_header(file_path: Union[str, Path]) -> Tuple[StationMetadata, int]:
    """
    Inspect the top rows of a CPCB/CECB Excel or CSV file to extract metadata
    and determine the row index where actual tabular data begins.
    """
    path = Path(file_path)
    station_name = path.stem
    city = "Korba"
    state = "Chhattisgarh"
    avg_period = "15 Minute"
    data_header_row = 0

    # Read first 25 rows as strings
    if path.suffix.lower() in [".xlsx", ".xls"]:
        preview_df = pd.read_excel(path, nrows=25, header=None)
    else:
        preview_df = pd.read_csv(path, nrows=25, header=None)

    for idx, row in preview_df.iterrows():
        row_str = " ".join([str(val) for val in row.values if pd.notna(val)])
        
        # Extract Station Name
        stn_match = re.search(r"(?:Station|Station Name)\s*[:=-]\s*([^,\n\r]+(?:,\s*[^,\n\r]+)?)", row_str, re.IGNORECASE)
        if stn_match:
            station_name = stn_match.group(1).strip()
            
        # Extract City
        city_match = re.search(r"City\s*[:=-]\s*([a-zA-Z\s]+)", row_str, re.IGNORECASE)
        if city_match:
            city = city_match.group(1).strip()
            
        # Extract State
        state_match = re.search(r"State\s*[:=-]\s*([a-zA-Z\s]+)", row_str, re.IGNORECASE)
        if state_match:
            state = state_match.group(1).strip()
            
        # Extract Avg Period
        avg_match = re.search(r"(?:AvgPeriod|Average Period|Interval)\s*[:=-]\s*([^\s,]+(?:\s+[^\s,]+)?)", row_str, re.IGNORECASE)
        if avg_match:
            avg_period = avg_match.group(1).strip()

        # Identify data header row (typically contains 'From Date' and 'To Date')
        if any(re.search(r"from\s*date", str(v), re.IGNORECASE) for v in row.values if pd.notna(v)):
            data_header_row = int(idx)
            break

    meta = StationMetadata(
        station_name=station_name,
        city=city,
        state=state,
        avg_period=avg_period,
        raw_file_path=str(path),
    )
    return meta, data_header_row


def load_cpcb_file(file_path: Union[str, Path]) -> Tuple[pd.DataFrame, StationMetadata]:
    """
    Ingests and parses a single CPCB / CECB .xlsx or .csv export.
    Returns:
        df: Cleaned 15-minute DataFrame with DatetimeIndex (IST), sorted and deduplicated.
        meta: StationMetadata object.
    """
    path = Path(file_path)
    if not path.exists():
        raise FileNotFoundError(f"File not found: {path}")

    meta, header_row = parse_cpcb_header(path)
    logger.info(f"Loading '{path.name}' | Station: '{meta.station_name}' | Header Row: {header_row}")

    if path.suffix.lower() in [".xlsx", ".xls"]:
        df_raw = pd.read_excel(path, skiprows=header_row)
    else:
        df_raw = pd.read_csv(path, skiprows=header_row)

    if df_raw.empty:
        raise ValueError(f"File {path.name} contains no tabular data below header row {header_row}.")

    # Identify timestamp columns
    from_col = None
    to_col = None
    for col in df_raw.columns:
        col_str = str(col).strip()
        if re.search(r"^from\s*date", col_str, re.IGNORECASE):
            from_col = col
        elif re.search(r"^to\s*date", col_str, re.IGNORECASE):
            to_col = col

    if from_col is None:
        # Fallback: check first column
        first_col = df_raw.columns[0]
        logger.warning(f"No explicit 'From Date' found; falling back to first column '{first_col}' as timestamp.")
        from_col = first_col

    # Parse timestamps: CPCB standard format is dd-mm-yyyy HH:MM or dd/mm/yyyy HH:MM
    df_raw["timestamp"] = pd.to_datetime(
        df_raw[from_col],
        format="%d-%m-%Y %H:%M",
        errors="coerce",
        dayfirst=True,
    )
    # If format mismatch, attempt flexible parsing
    if df_raw["timestamp"].isna().all():
        df_raw["timestamp"] = pd.to_datetime(df_raw[from_col], errors="coerce", dayfirst=True)

    # Drop rows where timestamp could not be parsed (e.g. trailing summary/empty rows)
    df_clean = df_raw.dropna(subset=["timestamp"]).copy()

    # Drop raw timestamp columns
    cols_to_drop = [c for c in [from_col, to_col] if c is not None and c in df_clean.columns]
    df_clean = df_clean.drop(columns=cols_to_drop)

    # Normalize remaining column names
    col_mapping = {}
    for col in df_clean.columns:
        if col != "timestamp":
            col_mapping[col] = normalize_column_name(str(col))
    df_clean = df_clean.rename(columns=col_mapping)

    # Convert all pollutant and weather columns to numeric
    numeric_cols = [c for c in df_clean.columns if c != "timestamp"]
    for c in numeric_cols:
        df_clean[c] = pd.to_numeric(df_clean[c], errors="coerce")

    # Deduplicate timestamps, keeping last reading
    df_clean = df_clean.drop_duplicates(subset=["timestamp"], keep="last")
    df_clean = df_clean.sort_values(by="timestamp").reset_index(drop=True)
    df_clean = df_clean.set_index("timestamp")

    # Trim empty padding rows at start and end
    # If all numeric columns are NaN, drop those rows
    valid_mask = df_clean[numeric_cols].notna().any(axis=1)
    if valid_mask.any():
        first_valid_idx = valid_mask.idxmax()
        last_valid_idx = valid_mask[::-1].idxmax()
        df_clean = df_clean.loc[first_valid_idx:last_valid_idx]

    meta.parameters_found = list(numeric_cols)
    return df_clean, meta


def load_all_stations(raw_dir: Union[str, Path]) -> Dict[str, pd.DataFrame]:
    """
    Scans raw directory and ingests all .xlsx and .csv files.
    Merges multiple files belonging to the same station across chunks.
    Returns:
        dict: {station_name: station_dataframe}
    """
    raw_path = Path(raw_dir)
    files = list(raw_path.glob("*.xlsx")) + list(raw_path.glob("*.xls")) + list(raw_path.glob("*.csv"))
    
    stations_data: Dict[str, List[pd.DataFrame]] = {}
    
    for f in sorted(files):
        try:
            df, meta = load_cpcb_file(f)
            stn = meta.station_name
            if stn not in stations_data:
                stations_data[stn] = []
            stations_data[stn].append(df)
            logger.info(f"Loaded {len(df)} rows from '{f.name}' for station '{stn}'")
        except Exception as e:
            logger.error(f"Error parsing file '{f.name}': {e}")

    merged_stations: Dict[str, pd.DataFrame] = {}
    for stn, dfs in stations_data.items():
        combined = pd.concat(dfs, axis=0)
        combined = combined[~combined.index.duplicated(keep="last")]
        combined = combined.sort_index()
        merged_stations[stn] = combined
        logger.info(f"Station '{stn}': Total {len(combined)} 15-minute readings across {len(dfs)} file(s).")

    return merged_stations
