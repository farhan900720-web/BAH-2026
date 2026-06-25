"""
harmonize_grids.py
==================
Spatially regrids IMD temperature data (1.0° grid) onto the higher-resolution
IMD rainfall grid (0.25°) using bilinear interpolation, then merges both
datasets into a single, analysis-ready xarray.Dataset.

Usage
-----
    python harmonize_grids.py
    → Produces  data/merged_climate_2023.nc
"""

import numpy as np
import xarray as xr


def harmonize_grids(temp_ds: xr.Dataset, rain_ds: xr.Dataset) -> xr.Dataset:
    """Regrid temperature data to match rainfall coordinates and merge.

    Parameters
    ----------
    temp_ds : xr.Dataset
        Temperature dataset on a coarse grid (e.g. 1.0° × 1.0°).
        Expected dimensions: ``(time, lat, lon)``.
    rain_ds : xr.Dataset
        Rainfall dataset on a fine grid (e.g. 0.25° × 0.25°).
        Expected dimensions: ``(time, lat, lon)``.

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
    # 3. Merge regridded temperature + original rainfall
    # ------------------------------------------------------------------
    merged = xr.merge(
        [temp_regridded, rain_ds],
        compat="override",      # keep rain_ds coords as the authority
        join="inner",           # only keep overlapping times
    )

    # Tidy up global attributes
    merged.attrs = {
        "title": "IMD Merged Daily Climate Dataset (0.25 x 0.25 deg)",
        "source": "India Meteorological Department",
        "grid_resolution": "0.25 degrees",
        "variables": ", ".join(sorted(merged.data_vars)),
        "history": (
            "Temperature regridded from 1.0 deg to 0.25 deg via bilinear "
            "interpolation; merged with native 0.25 deg rainfall."
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
    from load_imd_rainfall import load_imd_rainfall
    from load_imd_temp import load_imd_temp

    YEAR = 2023
    OUTPUT = "data/merged_climate_2023.nc"

    # ── Load datasets ──────────────────────────────────────────────────
    print("Loading rainfall …")
    rain_ds = load_imd_rainfall("data/Rainfall_ind2023_rfp25.grd", YEAR)

    print("Loading max temperature …")
    tmax_ds = load_imd_temp("data/Maxtemp_MaxT_2023.GRD", YEAR, "tmax")

    print("Loading min temperature …")
    tmin_ds = load_imd_temp("data/Mintemp_MinT_2023.GRD", YEAR, "tmin")

    # Combine tmax + tmin into one temperature dataset before regridding
    temp_ds = xr.merge([tmax_ds, tmin_ds])
    print(f"Combined temperature dataset: {list(temp_ds.data_vars)}\n")

    # ── Harmonize & merge ──────────────────────────────────────────────
    merged = harmonize_grids(temp_ds, rain_ds)
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
