"""
load_imd_rainfall.py
====================
Reads India Meteorological Department (IMD) daily gridded rainfall data
from raw binary (.grd) files written in C/Fortran float32 format.

Grid specification (from IMD C-code documentation):
  - 129 latitudes  : 6.5°N  → 38.5°N  (0.25° spacing)
  - 135 longitudes : 66.5°E → 100.0°E (0.25° spacing)
  - Data type      : 32-bit IEEE float (little-endian)
  - Missing value  : -999.0
"""

import numpy as np
import pandas as pd
import xarray as xr


# ---------------------------------------------------------------------------
# Grid constants (fixed by IMD specification)
# ---------------------------------------------------------------------------
NLAT = 129          # number of latitude  grid points
NLON = 135          # number of longitude grid points
GRID_SIZE = NLAT * NLON  # 17 415 values per day

LAT_START, LAT_END = 6.5, 38.5    # degrees North
LON_START, LON_END = 66.5, 100.0  # degrees East

MISSING_VALUE = -999.0  # IMD placeholder for no-data


def load_imd_rainfall(filepath: str, year: int) -> xr.Dataset:
    """Read an IMD daily gridded rainfall binary file into an xarray Dataset.

    Parameters
    ----------
    filepath : str
        Path to the ``.grd`` binary file (one full year of daily grids).
    year : int
        Calendar year of the data (used to build the time coordinate and
        to verify leap-year consistency).

    Returns
    -------
    xarray.Dataset
        Dataset with variable ``rainfall`` on dimensions
        ``('time', 'lat', 'lon')``.  Units are mm/day.
        Missing values (-999.0) are replaced with ``np.nan``.

    Raises
    ------
    ValueError
        If the file size is not an exact multiple of the grid size
        (NLAT × NLON), or if the inferred number of days doesn't match
        the calendar year.

    Examples
    --------
    >>> ds = load_imd_rainfall("Rainfall_ind2023_rfp25.grd", 2023)
    >>> ds
    <xarray.Dataset>
    Dimensions:   (time: 365, lat: 129, lon: 135)
    ...
    """

    # ------------------------------------------------------------------
    # 1. Read the flat binary data
    # ------------------------------------------------------------------
    raw = np.fromfile(filepath, dtype="float32")

    # ------------------------------------------------------------------
    # 2. Determine the number of days from the array length
    # ------------------------------------------------------------------
    if raw.size % GRID_SIZE != 0:
        raise ValueError(
            f"File size ({raw.size} floats) is not an exact multiple of "
            f"the IMD grid size ({NLAT}×{NLON} = {GRID_SIZE})."
        )

    ndays = raw.size // GRID_SIZE

    # Sanity-check against the calendar year
    import calendar
    expected_days = 366 if calendar.isleap(year) else 365
    if ndays != expected_days:
        raise ValueError(
            f"Inferred {ndays} days from the file, but year {year} has "
            f"{expected_days} days.  Check the file or the year argument."
        )

    # ------------------------------------------------------------------
    # 3. Reshape into (days, lat, lon)
    # ------------------------------------------------------------------
    data = raw.reshape((ndays, NLAT, NLON))

    # ------------------------------------------------------------------
    # 4. Replace IMD missing-value placeholder with NaN
    # ------------------------------------------------------------------
    data = np.where(np.isclose(data, MISSING_VALUE), np.nan, data)

    # ------------------------------------------------------------------
    # 5. Build coordinate arrays
    #    Using linspace avoids cumulative floating-point drift from arange.
    # ------------------------------------------------------------------
    latitudes  = np.linspace(LAT_START, LAT_END, NLAT)   # 6.50, 6.75, … 38.50
    longitudes = np.linspace(LON_START, LON_END, NLON)   # 66.50, 66.75, … 100.00
    times      = pd.date_range(start=f"{year}-01-01", periods=ndays, freq="D")

    # ------------------------------------------------------------------
    # 6. Package into an xarray Dataset
    # ------------------------------------------------------------------
    ds = xr.Dataset(
        data_vars={
            "rainfall": (
                ["time", "lat", "lon"],
                data,
                {
                    "units": "mm/day",
                    "long_name": "IMD Gridded Daily Rainfall",
                    "missing_value": MISSING_VALUE,
                },
            )
        },
        coords={
            "time": ("time", times),
            "lat":  ("lat",  latitudes,  {"units": "degrees_north", "long_name": "Latitude"}),
            "lon":  ("lon",  longitudes, {"units": "degrees_east",  "long_name": "Longitude"}),
        },
        attrs={
            "title": "IMD Daily Gridded Rainfall (0.25° × 0.25°)",
            "source": "India Meteorological Department",
            "grid_resolution": "0.25 degrees",
            "conventions": "CF-1.8",
        },
    )

    return ds


# ---------------------------------------------------------------------------
# Quick CLI usage
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import sys

    if len(sys.argv) != 3:
        print("Usage: python load_imd_rainfall.py <filepath.grd> <year>")
        sys.exit(1)

    fpath = sys.argv[1]
    yr    = int(sys.argv[2])

    ds = load_imd_rainfall(fpath, yr)
    print(ds)
    print("\n— Sample statistics —")
    print(ds["rainfall"].mean(dim="time").to_series().describe())
