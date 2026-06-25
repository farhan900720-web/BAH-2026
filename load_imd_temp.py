"""
load_imd_temp.py
================
Reads India Meteorological Department (IMD) daily gridded temperature data
(Minimum or Maximum) from raw binary (.grd) files written in float32 format.

Grid specification (from IMD documentation):
  - 31 latitudes  : 7.5°N  → 37.5°N  (1.0° spacing)
  - 31 longitudes : 67.5°E → 97.5°E  (1.0° spacing)
  - Data type     : 32-bit IEEE float (little-endian)
  - Missing value : 99.90
"""

import numpy as np
import pandas as pd
import xarray as xr


# ---------------------------------------------------------------------------
# Grid constants (fixed by IMD specification)
# ---------------------------------------------------------------------------
NLAT = 31           # number of latitude  grid points
NLON = 31           # number of longitude grid points
GRID_SIZE = NLAT * NLON  # 961 values per day

LAT_START, LAT_END = 7.5, 37.5    # degrees North
LON_START, LON_END = 67.5, 97.5   # degrees East

MISSING_VALUE = 99.90  # IMD placeholder for no-data


def load_imd_temp(filepath: str, year: int, temp_type: str = "tmax") -> xr.Dataset:
    """Read an IMD daily gridded temperature binary file into an xarray Dataset.

    Parameters
    ----------
    filepath : str
        Path to the ``.grd`` binary file (one full year of daily grids).
    year : int
        Calendar year of the data (used to build the time coordinate and
        to verify leap-year consistency).
    temp_type : str, optional
        Name for the data variable — typically ``'tmax'`` or ``'tmin'``.
        Defaults to ``'tmax'``.

    Returns
    -------
    xarray.Dataset
        Dataset with variable *temp_type* on dimensions
        ``('time', 'lat', 'lon')``.  Units are °C.
        Missing values (99.90) are replaced with ``np.nan``.

    Raises
    ------
    ValueError
        If the file size is not an exact multiple of the grid size
        (NLAT × NLON), or if the inferred number of days doesn't match
        the calendar year.

    Examples
    --------
    >>> ds = load_imd_temp("Maxtemp_MaxT_2023.grd", 2023, "tmax")
    >>> ds
    <xarray.Dataset>
    Dimensions:  (time: 365, lat: 31, lon: 31)
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
    latitudes  = np.linspace(LAT_START, LAT_END, NLAT)   # 7.5, 8.5, … 37.5
    longitudes = np.linspace(LON_START, LON_END, NLON)   # 67.5, 68.5, … 97.5
    times      = pd.date_range(start=f"{year}-01-01", periods=ndays, freq="D")

    # ------------------------------------------------------------------
    # 6. Package into an xarray Dataset
    # ------------------------------------------------------------------
    long_name = "Daily Maximum Temperature" if temp_type == "tmax" else "Daily Minimum Temperature"

    ds = xr.Dataset(
        data_vars={
            temp_type: (
                ["time", "lat", "lon"],
                data,
                {
                    "units": "°C",
                    "long_name": long_name,
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
            "title": f"IMD Daily Gridded {long_name} (1.0° × 1.0°)",
            "source": "India Meteorological Department",
            "grid_resolution": "1.0 degrees",
            "conventions": "CF-1.8",
        },
    )

    return ds


# ---------------------------------------------------------------------------
# Quick CLI usage
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import sys

    if len(sys.argv) != 4:
        print("Usage: python load_imd_temp.py <filepath.grd> <year> <tmax|tmin>")
        sys.exit(1)

    fpath     = sys.argv[1]
    yr        = int(sys.argv[2])
    ttype     = sys.argv[3]

    ds = load_imd_temp(fpath, yr, ttype)
    print(ds)
    print("\n— Sample statistics —")
    print(ds[ttype].mean(dim="time").to_series().describe())
