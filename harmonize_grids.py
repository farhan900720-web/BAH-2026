"""
harmonize_grids.py
==================
Spatially regrids IMD temperature data (1.0° grid) and ISRO soil moisture
data (~0.05° grid) onto the higher-resolution IMD rainfall grid (0.25°)
using bilinear interpolation, then merges all datasets into a single,
analysis-ready xarray.Dataset.

Usage
-----
    python harmonize_grids.py 2023
    → Produces  data/merged_climate_2023.nc
"""

import os

import numpy as np
import xarray as xr


def load_soil_moisture(filepath: str, rain_ds: xr.Dataset) -> xr.Dataset:
    """Load an ISRO Root Zone Soil Moisture NetCDF and regrid to rainfall grid.

    Parameters
    ----------
    filepath : str
        Path to the RZSM NetCDF file (e.g. ``data/RZSM_2022.nc``).
    rain_ds : xr.Dataset
        Rainfall dataset whose ``(lat, lon)`` coordinates define the
        target 0.25° grid.

    Returns
    -------
    xr.Dataset
        Dataset with a single variable ``soil_moisture`` on the rainfall
        grid's ``(time, lat, lon)`` coordinates.

    Notes
    -----
    * Uses ``chunks={'time': 30}`` for lazy / chunked loading to avoid
      MemoryErrors on the ~495 MB source files.
    * The ISRO grid (~0.05°) is much finer than the IMD 0.25° target,
      so ``interp_like`` effectively performs spatial averaging.
    """
    print(f"Loading soil moisture from {filepath} ...")
    try:
        # Prefer chunked/lazy loading if dask is available
        sm_ds = xr.open_dataset(filepath, chunks={"time": 30})
    except ImportError:
        # Fallback: eager loading (works without dask, uses more RAM)
        print("  [INFO] dask not installed — loading eagerly (may use ~500 MB RAM)")
        sm_ds = xr.open_dataset(filepath)

    # Rename the variable for clarity
    sm_ds = sm_ds.rename({"rzsm": "soil_moisture"})

    # Subset to the time range present in rain_ds
    common_times = np.intersect1d(sm_ds["time"].values, rain_ds["time"].values)
    if common_times.size == 0:
        raise ValueError(
            "Soil moisture and rainfall datasets share no common dates."
        )
    sm_ds = sm_ds.sel(time=common_times)

    # Regrid soil moisture onto the rainfall 0.25° grid
    print(
        f"  Regridding soil moisture from "
        f"({sm_ds.sizes['lat']}x{sm_ds.sizes['lon']}) -> "
        f"({rain_ds.sizes['lat']}x{rain_ds.sizes['lon']}) ..."
    )
    sm_regridded = sm_ds.interp_like(
        rain_ds,
        method="linear",
        kwargs={"fill_value": None},
    )

    # Force computation from lazy chunks so downstream ops work cleanly
    sm_regridded = sm_regridded.compute()

    print(f"  [OK] Soil moisture regridded — {common_times.size} time steps")
    return sm_regridded


def harmonize_grids(
    temp_ds: xr.Dataset,
    rain_ds: xr.Dataset,
    soil_ds: xr.Dataset | None = None,
) -> xr.Dataset:
    """Regrid temperature data to match rainfall coordinates and merge.

    Parameters
    ----------
    temp_ds : xr.Dataset
        Temperature dataset on a coarse grid (e.g. 1.0° × 1.0°).
        Expected dimensions: ``(time, lat, lon)``.
    rain_ds : xr.Dataset
        Rainfall dataset on a fine grid (e.g. 0.25° × 0.25°).
        Expected dimensions: ``(time, lat, lon)``.
    soil_ds : xr.Dataset or None, optional
        Soil moisture dataset already regridded to the rainfall grid.
        If provided, it is merged into the final dataset.

    Returns
    -------
    xr.Dataset
        Unified dataset where every variable shares the same
        ``(time, lat, lon)`` coordinates (those of *rain_ds*).

    Notes
    -----
    * Interpolation uses ``method='linear'`` (bilinear in 2-D).
    * Grid points in ``rain_ds`` that fall outside the spatial extent of
      ``temp_ds`` will be filled with ``NaN`` (no extrapolation).
    * The time dimensions must already be aligned (same dates); a check
      is performed and a ``ValueError`` raised if they differ.
    """

    # ------------------------------------------------------------------
    # 1. Validate temporal alignment
    # ------------------------------------------------------------------
    temp_times = temp_ds["time"].values
    rain_times = rain_ds["time"].values

    if not np.array_equal(temp_times, rain_times):
        # Find the common overlap and warn, or raise if there's none
        common = np.intersect1d(temp_times, rain_times)
        if common.size == 0:
            raise ValueError(
                "Temperature and rainfall datasets share no common dates. "
                "Cannot merge."
            )
        print(
            f"[WARN] Time axes differ - subsetting to {common.size} common dates "
            f"(temp has {temp_times.size}, rain has {rain_times.size})."
        )
        temp_ds = temp_ds.sel(time=common)
        rain_ds = rain_ds.sel(time=common)

    # ------------------------------------------------------------------
    # 2. Spatially interpolate temp_ds onto the rain_ds grid
    #    interp_like matches lat/lon from rain_ds using bilinear interp.
    # ------------------------------------------------------------------
    temp_regridded = temp_ds.interp_like(
        rain_ds,
        method="linear",
        kwargs={"fill_value": None},   # NaN outside original domain
    )

    print(
        f"[OK] Regridded temperature from "
        f"({temp_ds.sizes['lat']}x{temp_ds.sizes['lon']}) -> "
        f"({temp_regridded.sizes['lat']}x{temp_regridded.sizes['lon']})"
    )

    # ------------------------------------------------------------------
    # 3. Merge regridded temperature + original rainfall (+ soil moisture)
    # ------------------------------------------------------------------
    datasets_to_merge = [temp_regridded, rain_ds]
    if soil_ds is not None:
        datasets_to_merge.append(soil_ds)

    merged = xr.merge(
        datasets_to_merge,
        compat="override",      # keep rain_ds coords as the authority
        join="inner",           # only keep overlapping times
    )

    # Tidy up global attributes
    merged.attrs = {
        "title": "IMD + ISRO Merged Daily Climate Dataset (0.25 x 0.25 deg)",
        "source": "India Meteorological Department / ISRO Soil Moisture",
        "grid_resolution": "0.25 degrees",
        "variables": ", ".join(sorted(merged.data_vars)),
        "history": (
            "Temperature regridded from 1.0 deg to 0.25 deg; "
            "Soil moisture regridded from ~0.05 deg to 0.25 deg; "
            "merged with native 0.25 deg rainfall via bilinear interpolation."
        ),
        "conventions": "CF-1.8",
    }

    print(f"[OK] Merged dataset: {dict(merged.sizes)}")
    print(f"     Variables: {list(merged.data_vars)}")

    return merged


# -----------------------------------------------------------------------
# Execution block — load, harmonize, save
# -----------------------------------------------------------------------
if __name__ == "__main__":
    import argparse
    from load_imd_rainfall import load_imd_rainfall
    from load_imd_temp import load_imd_temp

    parser = argparse.ArgumentParser(description="Harmonize IMD grids for a given year.")
    parser.add_argument("year", type=int, help="Year to process (e.g. 2022, 2023, 2024)")
    args = parser.parse_args()

    YEAR = args.year
    OUTPUT = f"data/merged_climate_{YEAR}.nc"

    # ── Load datasets ──────────────────────────────────────────────────
    print("Loading rainfall …")
    rain_ds = load_imd_rainfall(f"data/Rainfall_ind{YEAR}_rfp25.grd", YEAR)

    print("Loading max temperature …")
    tmax_ds = load_imd_temp(f"data/Maxtemp_MaxT_{YEAR}.GRD", YEAR, "tmax")

    print("Loading min temperature …")
    tmin_ds = load_imd_temp(f"data/Mintemp_MinT_{YEAR}.GRD", YEAR, "tmin")

    # Combine tmax + tmin into one temperature dataset before regridding
    temp_ds = xr.merge([tmax_ds, tmin_ds])
    print(f"Combined temperature dataset: {list(temp_ds.data_vars)}\n")

    # ── Load ISRO soil moisture ────────────────────────────────────────
    soil_path = f"data/RZSM_{YEAR}.nc"
    soil_ds = None
    if os.path.isfile(soil_path):
        soil_ds = load_soil_moisture(soil_path, rain_ds)
    else:
        print(f"[WARN] Soil moisture file not found: {soil_path} — skipping.")

    # ── Harmonize & merge ──────────────────────────────────────────────
    merged = harmonize_grids(temp_ds, rain_ds, soil_ds=soil_ds)
    print(f"\n{merged}\n")

    # ── Save to NetCDF ─────────────────────────────────────────────────
    print(f"Saving to {OUTPUT} ...")
    merged.to_netcdf(
        OUTPUT,
        engine="netcdf4",
        encoding={
            var: {"dtype": "float32", "zlib": True, "complevel": 4}
            for var in merged.data_vars
        },
    )
    print(f"[OK] Saved -> {OUTPUT}")
