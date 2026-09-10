"""
drifter_tools.py
=================

Reusable, non-interactive tools for fetching and plotting ASKJA drifter
SST/GPS data.

Mapping uses cartopy. Point/track data is plotted with no network
dependency; the optional tile basemap (Icelandic Met Office basemap by
default, or Esri World Imagery satellite, or any other XYZ source) does
require a live network call, but falls back automatically to a plain
land/coastline fill if that call fails, so a network hiccup degrades
the plot instead of crashing the notebook. All map plotting is done
against a flexible lon/lat `Domain`, matching the style used for the
SWOT tools.

The companion Jupyter notebook (`drifter_tracking.ipynb`) provides the
interactive part (domain, days-ago, SST colour range, smoothing window)
and calls into this module to do the actual work. Every plotting
function also accepts `save=`/`outfile=` so the same functions can later
be called from an unattended script for automatic report generation,
without changing anything here.
"""

from io import StringIO

import numpy as np
import pandas as pd
import requests
from requests.auth import HTTPBasicAuth

import matplotlib.pyplot as plt
import matplotlib.dates as mdates

import cartopy.crs as ccrs
import cartopy.feature as cfeature
import cartopy.io.img_tiles as cimgt

# Web Mercator projection for map axes — deliberately cimgt's native
# `.crs` (cartopy.crs.Mercator under the hood), NOT ccrs.epsg(3857).
# The two are subtly different: epsg(3857) goes through cartopy's
# generic external EPSG/PROJ database lookup (`_EPSGProjection`, a much
# less battle-tested code path with slightly different bounds), while
# tile classes' `.crs` is cartopy's own native, well-integrated
# Mercator class — exactly what the IMO/cartopy tile examples use
# (`projection=IMO_basemap.crs`). The EPSG-based version rendered
# inconsistently between environments (worked locally, produced a
# broken map on GitHub Actions' runner) — this native class is the
# more robust choice. No network call is needed to read `.crs`.
WEB_MERCATOR_CRS = cimgt.GoogleTiles().crs


# ---------------------------------------------------------------------
# Domain
# ---------------------------------------------------------------------
class Domain:
    """
    A simple, flexible lon/lat bounding box (same pattern as the SWOT
    tools' Domain), so the same functions work for any drifter/location.
    """

    def __init__(self, lon1, lon2, lat1, lat2):
        self.lon1 = float(lon1)
        self.lon2 = float(lon2)
        self.lat1 = float(lat1)
        self.lat2 = float(lat2)

    @property
    def extent(self):
        """(lonmin, lonmax, latmin, latmax) — as used by ax.set_extent."""
        return (self.lon1, self.lon2, self.lat1, self.lat2)

    @classmethod
    def from_points(cls, lons, lats, buffer_deg=0.01, lat_buffer_factor=None):
        """
        Build a domain tightly bounding a set of points, plus a margin.

        `buffer_deg` is applied to longitude as-is. The north/south
        margin is `buffer_deg * lat_buffer_factor` — smaller than the
        east/west margin, because a degree of longitude covers less
        ground distance than a degree of latitude away from the equator
        (by a factor of cos(latitude)). Without this, a buffer that
        looks numerically equal in degrees ends up visually much taller
        than wide at high latitude, padding north/south far more than
        east/west in real terms.

        `lat_buffer_factor=None` (default) auto-computes cos(mean
        latitude of the points). Pass a specific value (e.g. 1.0 to
        disable the correction, matching the old behaviour) to override.
        """
        lons = np.asarray(lons)
        lats = np.asarray(lats)
        if lat_buffer_factor is None:
            lat_buffer_factor = np.cos(np.radians(lats.mean()))
        lat_buffer = buffer_deg * lat_buffer_factor
        return cls(lons.min() - buffer_deg, lons.max() + buffer_deg,
                    lats.min() - lat_buffer, lats.max() + lat_buffer)

    def __repr__(self):
        return (f"Domain(lon=[{self.lon1}, {self.lon2}], "
                f"lat=[{self.lat1}, {self.lat2}])")


# ---------------------------------------------------------------------
# 1. Fetch & prepare drifter data
# ---------------------------------------------------------------------
def fetch_drifter_data(platform_id, days_ago, api_url, auth_user, auth_pass,
                        smooth_window=3):
    """
    Fetch drifter SST/GPS data from the LDL API and return a DataFrame
    indexed by UTC timestamp, with an added rolling-mean SST column.
    """
    params = {"platform_id": platform_id, "days_ago": str(days_ago)}
    auth = HTTPBasicAuth(auth_user, auth_pass)
    response = requests.get(api_url, params=params, auth=auth)
    response.raise_for_status()

    df = pd.read_csv(StringIO(response.text))
    df.columns = df.columns.str.strip()
    df["Timestamp(UTC)"] = pd.to_datetime(df["Timestamp(UTC)"])
    df = df.set_index("Timestamp(UTC)").sort_index()
    df["sst_smooth"] = df["SST(degC)"].rolling(window=smooth_window, center=True).mean()
    return df


def summarize_latest_temperature(df, window_hours=3, expected_interval_hours=3,
                                  stale_multiplier=2):
    """
    Summarize the most recent temperature readings: the average SST over
    the last `window_hours`, and how stale that data actually is.

    The drifter normally reports every `expected_interval_hours` (3h by
    default), but can miss reports — most commonly when it's far enough
    north that satellite coverage gets patchy. Rather than silently
    showing a 3h average built from a single very-old point (or no
    points), this flags when the latest report is older than expected,
    so a report/dashboard can say "no update in the last few hours"
    instead of implying the ocean is currently that temperature.

    Parameters
    ----------
    df : DataFrame
        Output of `fetch_drifter_data` (needs a DatetimeIndex and an
        'SST(degC)' column).
    window_hours : float
        Averaging window, measured back from the most recent report
        (not from "now") — e.g. 3 means "the last 3 hours of data the
        drifter actually sent".
    expected_interval_hours : float
        Normal reporting cadence, used only to decide what counts as
        "stale" relative to real time.
    stale_multiplier : float
        Data is flagged stale if the gap since the last report exceeds
        `expected_interval_hours * stale_multiplier`.

    Returns
    -------
    dict with keys:
        'latest_time'      : timestamp of the most recent report
        'age_hours'        : hours between now and that report
        'is_stale'         : bool, True if age exceeds the stale threshold
        'window_mean'      : mean SST over the last `window_hours` of
                              *reported* data (None if no data in window)
        'window_n'         : number of readings averaged
        'text'             : a ready-to-display one/two-line summary
    """
    if df.empty:
        return {
            "latest_time": None, "age_hours": None, "is_stale": True,
            "window_mean": None, "window_n": 0,
            "text": "No drifter data available.",
        }

    latest_time = df.index.max()
    now = pd.Timestamp.now(tz=latest_time.tz) if latest_time.tzinfo else pd.Timestamp.now()
    age_hours = (now - latest_time).total_seconds() / 3600.0
    is_stale = age_hours > expected_interval_hours * stale_multiplier

    window_df = df[df.index >= latest_time - pd.Timedelta(hours=window_hours)]
    window_mean = window_df["SST(degC)"].mean() if len(window_df) else None
    window_n = len(window_df)

    if window_mean is None:
        text = "No temperature readings available in the requested window."
    else:
        text = (f"Average SST over the last {window_hours:g}h of reported "
                f"data: {window_mean:.2f}\u00b0C ({window_n} reading"
                f"{'s' if window_n != 1 else ''}, ending {latest_time:%Y-%m-%d %H:%M} UTC).")

    if is_stale:
        text += (f" Note: last report was {age_hours:.1f}h ago — longer than "
                 f"the usual ~{expected_interval_hours:g}h cadence, likely a "
                 f"gap in satellite coverage (e.g. drifter too far north). "
                 f"This average may not reflect current conditions.")

    return {
        "latest_time": latest_time,
        "age_hours": age_hours,
        "is_stale": is_stale,
        "window_mean": window_mean,
        "window_n": window_n,
        "text": text,
    }


# ---------------------------------------------------------------------
# Shared "very high temperature" detection — same definition used by
# plot_timeseries's dark-red highlight AND plot_map's extreme-location
# markers, so both always agree on what counts as extreme.
# ---------------------------------------------------------------------
def compute_extreme_flags(df, smooth_window=3, extreme_smooth_window=5,
                           extreme_threshold_std=1.0, exclude_first_days=1):
    """
    Identify which rows of `df` have "very high" SST.

    Parameters mean the same as in `plot_timeseries` — see there for the
    full explanation of the smoothing/baseline/threshold logic.

    Returns
    -------
    dict with keys:
        'is_extreme'     : bool Series aligned to df.index
        'sst_smooth'     : Series, main-window smoothed SST
        'extreme_smooth' : Series, extreme-window smoothed SST
        'baseline'       : float, baseline mean SST used
        'extreme_std'    : float, std of the baseline-period anomaly
    """
    sst_smooth = df["SST(degC)"].rolling(window=smooth_window, center=True).mean()

    baseline_start = df.index.min() + pd.Timedelta(days=exclude_first_days)
    baseline_mask = df.index >= baseline_start
    if not baseline_mask.any():
        baseline_mask = np.ones(len(df), dtype=bool)  # exclusion window ate the whole record
    baseline = df.loc[baseline_mask, "SST(degC)"].mean()

    baseline_anomaly = sst_smooth[baseline_mask] - baseline
    extreme_std = baseline_anomaly.std()

    extreme_smooth = df["SST(degC)"].rolling(window=extreme_smooth_window, center=True).mean()
    extreme_anomaly = extreme_smooth - baseline
    is_extreme = extreme_anomaly > extreme_threshold_std * extreme_std

    return {
        "is_extreme": is_extreme,
        "sst_smooth": sst_smooth,
        "extreme_smooth": extreme_smooth,
        "baseline": baseline,
        "extreme_std": extreme_std,
    }


# ---------------------------------------------------------------------
# 2. Time series plot: raw+smoothed SST, plus an anomaly panel
# ---------------------------------------------------------------------
def plot_timeseries(df, smooth_window=3, extreme_smooth_window=5,
                     extreme_threshold_std=1.0, exclude_first_days=1,
                     title="Drifter surface temperature",
                     save=False, outfile="timeseries.png", dpi=300):
    """
    Two-panel time series plot:
      - top: raw + smoothed SST (as before).
      - bottom: anomaly relative to the record's own mean, ENSO/ONI
        style — red where above average, blue where below — with a
        darker red band for sustained very-high-temperature periods.

    The "very high" highlight is computed on a heavier smooth
    (`extreme_smooth_window`, default 5 points) than the main line
    (`smooth_window`, default 3), so a single noisy spike doesn't get
    flagged — only stretches that stay elevated hold up under the
    heavier smoothing. Same definition as `compute_extreme_flags`, which
    `plot_map` also uses so the anomaly panel and the maps agree.

    Date-axis ticks use matplotlib's adaptive locator/formatter, so
    spacing and label format adjust automatically whether the record
    spans days or months, instead of a fixed interval that gets crowded
    or too sparse depending on how much data you actually have.

    Parameters
    ----------
    df : DataFrame
        Needs an 'SST(degC)' column and a DatetimeIndex. `sst_smooth` is
        (re)computed here from `smooth_window`, so the column doesn't
        need to already exist (and isn't mutated in place on `df`).
    smooth_window : int
        Rolling-mean window for the main smoothed line/anomaly fill.
    extreme_smooth_window : int
        Rolling-mean window used only to decide which periods count as
        "very high" — independent of, and normally larger than,
        `smooth_window`.
    extreme_threshold_std : float
        A period is flagged "very high" when its extreme-smoothed
        anomaly exceeds this many standard deviations (of the main
        anomaly series) above zero.
    exclude_first_days : float
        Days at the start of the record excluded from the baseline mean
        and standard deviation (but still shown on the plot) — the first
        day is typically deployment/handling, not representative of the
        water the drifter settles into. Set to 0 to use the whole
        record for the baseline.
    """
    flags = compute_extreme_flags(
        df, smooth_window=smooth_window,
        extreme_smooth_window=extreme_smooth_window,
        extreme_threshold_std=extreme_threshold_std,
        exclude_first_days=exclude_first_days,
    )
    df = df.copy()
    df["sst_smooth"] = flags["sst_smooth"]
    baseline = flags["baseline"]
    is_extreme = flags["is_extreme"]
    anomaly = df["sst_smooth"] - baseline

    fig, (ax1, ax2) = plt.subplots(
        2, 1, figsize=(12, 8), sharex=True,
        gridspec_kw={"height_ratios": [1, 1], "hspace": 0.08},
    )

    # --- panel 1: raw + smoothed SST ---
    ax1.plot(df.index, df["SST(degC)"], marker="o", ms=3, lw=0.8,
            alpha=0.4, color="tab:blue", label="raw SST")
    ax1.plot(df.index, df["sst_smooth"], marker="o", ms=4, lw=1.5,
            color="tab:red", label=f"{smooth_window}-pt smoothed")
    ax1.set_ylabel("SST (\u00b0C)")
    ax1.set_title(title)
    ax1.grid(True)
    ax1.legend()

    # --- panel 2: anomaly, El Ni\u00f1o/La Ni\u00f1a style ---
    ax2.axhline(0, color="k", lw=0.8)
    ax2.fill_between(df.index, anomaly, 0, where=(anomaly >= 0),
                      color="tab:red", alpha=0.5, interpolate=True,
                      label="above average")
    ax2.fill_between(df.index, anomaly, 0, where=(anomaly < 0),
                      color="tab:blue", alpha=0.5, interpolate=True,
                      label="below average")
    ax2.fill_between(df.index, anomaly, 0, where=is_extreme,
                      color="#67000d", alpha=0.95, interpolate=True,
                      label=f"very high (>{extreme_threshold_std:g}\u03c3, "
                            f"{extreme_smooth_window}-pt smooth)")
    ax2.set_ylabel("Anomaly (\u00b0C)")
    ax2.grid(True)
    ax2.legend(loc="upper left", fontsize=8)

    # --- flexible tick spacing, dd.mm date labels ---
    locator = mdates.AutoDateLocator()
    formatter = mdates.DateFormatter("%d.%m")
    ax2.xaxis.set_major_locator(locator)
    ax2.xaxis.set_major_formatter(formatter)
    ax2.set_xlabel("Time (UTC)")
    fig.autofmt_xdate(rotation=0)

    if save:
        fig.savefig(outfile, dpi=dpi, bbox_inches="tight", facecolor="white")
        print(f"Saved {outfile}")

    plt.show()
    return fig


# ---------------------------------------------------------------------
# 3. Bathymetry contours, loaded from a precomputed CSV
# ---------------------------------------------------------------------
def load_contours(path):
    """
    Load bathymetry contour lines from a CSV with columns:
    contour_id, level, lon, lat. Returns a list of dicts:
    {'level': ..., 'lon': array, 'lat': array} — one per contour_id.

    (This replaces the old geopandas/shapely-based `load_contours`, since
    cartopy just needs plain lon/lat arrays to draw lines — no projected
    GeoDataFrame required.)
    """
    df_c = pd.read_csv(path)
    contours = []
    for cid, group in df_c.groupby("contour_id"):
        contours.append({
            "level": group["level"].iloc[0],
            "lon": group["lon"].values,
            "lat": group["lat"].values,
        })
    return contours


def add_contours(ax, contours, color="k", linewidth=0.4):
    """Draw contour lines (from `load_contours`) on a cartopy GeoAxes."""
    if not contours:
        return
    for c in contours:
        ax.plot(c["lon"], c["lat"], color=color, linewidth=linewidth,
                transform=ccrs.PlateCarree(), zorder=4)


# ---------------------------------------------------------------------
# Tile basemaps (any XYZ source, with graceful fallback)
# ---------------------------------------------------------------------
# Named sources — add more here as you find useful ones. Any XYZ URL
# template with {x}/{y}/{z} placeholders works; cartopy's GoogleTiles
# just formats the string, so the placeholder order doesn't matter.
TILE_SOURCES = {
    "satellite": (
        "https://server.arcgisonline.com/ArcGIS/rest/services/"
        "World_Imagery/MapServer/tile/{z}/{y}/{x}"
    ),
    "imo": (
        "https://geo.vedur.is/geoserver/www/"
        "imo_basemap_epsg3857/{z}/{x}/{y}.png"
    ),
}


def _auto_zoom_level(domain, min_zoom=3, max_zoom=17):
    """Pick a tile zoom level that roughly matches the domain's extent."""
    lon_span = max(domain.lon2 - domain.lon1, 1e-9)
    for z in range(max_zoom, min_zoom, -1):
        tile_deg = 360 / (2 ** z)
        if tile_deg >= lon_span:
            return z
    return min_zoom


def _tile_url_reachable(url_template, timeout=3):
    """
    Quick reachability probe for a tile URL template, using a single
    low-zoom sample tile (z=1, x=0, y=0). ax.add_image() from cartopy
    swallows per-tile HTTP errors internally (logs them, doesn't raise),
    so a try/except around it never actually detects failure — it would
    otherwise silently render a blank background. Checking reachability
    up front lets us fall back cleanly instead.
    """
    try:
        sample_url = url_template.format(x=0, y=0, z=1)
        r = requests.get(sample_url, timeout=timeout)
        return r.status_code == 200
    except Exception:
        return False


def add_tile_basemap(ax, domain, source="satellite", zoom=None,
                      fallback_color="0.85"):
    """
    Add a tile-based basemap to a cartopy GeoAxes, from any named source
    in `TILE_SOURCES` (or a raw XYZ URL template string). Falls back to
    a plain land fill (no network needed) if the tile server isn't
    reachable, so a network hiccup degrades the plot instead of
    silently rendering a blank background.

    Parameters
    ----------
    ax : GeoAxes
    domain : Domain
        Used to auto-pick a sensible zoom level if `zoom` isn't given.
    source : str
        A key in `TILE_SOURCES` (e.g. 'satellite', 'imo'), or a raw XYZ
        URL template string with {x}/{y}/{z} placeholders.
    zoom : int or None
        Tile zoom level (higher = more detail, slower). None = auto.
    fallback_color : str
        Land fill colour used only if the tile server is unreachable.

    Returns
    -------
    bool
        True if the tile imagery was added, False if it fell back.
    """
    url_template = TILE_SOURCES.get(source, source)

    if zoom is None:
        zoom = _auto_zoom_level(domain)

    if _tile_url_reachable(url_template):
        try:
            tiler = cimgt.GoogleTiles(url=url_template)
            ax.add_image(tiler, zoom)
            return True
        except Exception as e:
            print(f"Basemap '{source}' failed while rendering ({e}); "
                  f"falling back to a plain land fill.")
    else:
        print(f"Basemap '{source}' tile server unreachable; "
              f"falling back to a plain land fill.")

    ax.add_feature(cfeature.LAND, facecolor=fallback_color)
    ax.add_feature(cfeature.COASTLINE, linewidth=0.8)
    return False


# ---------------------------------------------------------------------
# 4. Maps
# ---------------------------------------------------------------------
def plot_map(df, domain, contours=None, title="Drifter track",
             n_last=None, show_trajectory=True,
             vmin=4.5, vmax=6.5, cmap="plasma",
             basemap="imo", zoom=None, land_color="0.85",
             show_extreme=True, extreme_color="black",
             extreme_marker="o", extreme_size=80,
             smooth_window=5, extreme_smooth_window=10,
             extreme_threshold_std=3.0, exclude_first_days=1,
             figsize=(9, 8),
             save=False, outfile="map.png", dpi=300):
    """
    Plot drifter positions coloured by (smoothed) SST on a cartopy map.

    Parameters
    ----------
    df : DataFrame
        Output of `fetch_drifter_data` — needs GPS-Longitude(deg),
        GPS-Latitude(deg), sst_smooth columns.
    domain : Domain
        Lon/lat extent to plot.
    contours : list of dict or None
        Output of `load_contours`, or None to skip.
    n_last : int or None
        If set, only the last n_last rows are plotted (e.g. "most recent
        positions" map).
    show_trajectory : bool
        Draw a thin line connecting the plotted points.
    basemap : str or None
        A key in `TILE_SOURCES` ('satellite', 'imo'), a raw XYZ URL
        template, 'land', or None.
        'satellite'/'imo'/<url> — tile imagery, with automatic fallback
                      to 'land' if the tile request fails.
        'land'      — plain cartopy land fill + coastline, no network.
        None        — no basemap layer at all (just gridlines/points).
    zoom : int or None
        Tile zoom level for a tile-based basemap. None = auto, based on
        the domain's extent.
    show_extreme : bool
        If True (default), mark locations where SST was "very high" —
        same definition as `plot_timeseries`'s dark-red highlight, via
        `compute_extreme_flags` — with a distinct marker, so spatial
        patterns in extreme readings are visible. Always computed on the
        FULL record (not just the points shown for an n_last map), so
        the baseline/threshold stay consistent across every map.
    extreme_color, extreme_marker, extreme_size :
        Style of the "very high" marker. Default: solid black 'X'.
    smooth_window, extreme_smooth_window, extreme_threshold_std,
    exclude_first_days :
        Same meaning as in `plot_timeseries` — keep these matched to
        whatever you used there so the maps and the anomaly panel agree
        on which points are flagged.
    """
    sub = df.iloc[-n_last:] if n_last else df
    lon = sub["GPS-Longitude(deg)"].values
    lat = sub["GPS-Latitude(deg)"].values
    sst = sub["sst_smooth"].values

    if show_extreme:
        flags = compute_extreme_flags(
            df, smooth_window=smooth_window,
            extreme_smooth_window=extreme_smooth_window,
            extreme_threshold_std=extreme_threshold_std,
            exclude_first_days=exclude_first_days,
        )
        # Computed on the full df; align down to just the rows in `sub`
        # (e.g. for an n_last map) via the shared DatetimeIndex.
        is_extreme_sub = flags["is_extreme"].loc[sub.index].values
    else:
        is_extreme_sub = None

    # Use Web Mercator for the axes themselves (not PlateCarree). All the
    # XYZ tile sources here (imo, satellite, or a custom URL) are Web
    # Mercator, and — more importantly — PlateCarree stretches longitude
    # and latitude by the same amount per degree, which visibly distorts
    # shapes away from the equator (at 65\u00b0N, 1\u00b0 of longitude covers
    # ~42% as much ground distance as 1\u00b0 of latitude). Web Mercator is
    # locally shape-correct, which is why the IMO/cartopy tile examples
    # use it (`projection=IMO_basemap.crs`) instead of PlateCarree.
    # NOTE: no constrained_layout here — matplotlib's layout engines
    # (constrained_layout, and axes_grid1-based colorbars we used to use
    # here) both have known rough edges with cartopy GeoAxes, especially
    # once a real raster tile image is drawn. The colorbar below is
    # positioned manually instead, and bbox_inches="tight" at save time
    # handles final spacing.
    fig = plt.figure(figsize=figsize)
    ax = plt.axes(projection=WEB_MERCATOR_CRS)
    # domain.extent is in lon/lat degrees, not Web Mercator metres, so
    # set_extent needs to be told what CRS those numbers are in.
    ax.set_extent(domain.extent, crs=ccrs.PlateCarree())

    if basemap == "land":
        ax.add_feature(cfeature.LAND, facecolor=land_color)
        ax.add_feature(cfeature.COASTLINE, linewidth=0.8)
    elif basemap:
        add_tile_basemap(ax, domain, source=basemap, zoom=zoom,
                          fallback_color=land_color)
    # basemap=None/falsy: no background layer added

    if show_trajectory:
        ax.plot(lon, lat, color="gray", linewidth=0.5, zorder=2,
                label="drifter track", transform=ccrs.PlateCarree())

    sc = ax.scatter(lon, lat, c=sst, cmap=cmap, s=30,
                     vmin=vmin, vmax=vmax, alpha=0.7, zorder=3,
                     transform=ccrs.PlateCarree())

    add_contours(ax, contours)

    if show_extreme and is_extreme_sub is not None and is_extreme_sub.any():
        ax.scatter(lon[is_extreme_sub], lat[is_extreme_sub],
                   marker=extreme_marker, s=extreme_size,
                   color=extreme_color, alpha=0.7,
                   zorder=6, transform=ccrs.PlateCarree(),
                   label=f"very high SST (>{extreme_threshold_std:g}\u03c3)")

    # mark most recent position
    ax.plot(lon[-1], lat[-1], marker="o", markersize=14,
            markerfacecolor="none", markeredgecolor="black", zorder=5,
            transform=ccrs.PlateCarree())

    gl = ax.gridlines(draw_labels=True, linewidth=0.5)
    gl.top_labels = False
    gl.right_labels = False

    # Colorbar sized to match the axes' ACTUAL rendered position, computed
    # manually rather than via mpl_toolkits.axes_grid1.make_axes_locatable.
    # axes_grid1 has documented compatibility problems with cartopy GeoAxes
    # specifically — more likely to surface once a real raster tile image
    # is drawn (vs. the simple vector scatter points this was tested with
    # locally, since this sandbox can't reach real tile servers). This
    # forces a draw first so ax.get_position() reflects the true final
    # layout (including gridline label spacing), then places cax directly
    # against that real position — no divider/locator machinery involved.
    fig.canvas.draw()
    pos = ax.get_position()
    cax = fig.add_axes([pos.x1 + 0.02, pos.y0, 0.025, pos.height])
    fig.colorbar(sc, cax=cax, label="Temperature (\u00b0C)")

    ax.set_title(title)
    has_extreme_marker = show_extreme and is_extreme_sub is not None and is_extreme_sub.any()
    if show_trajectory or has_extreme_marker:
        ax.legend(loc="lower left", fontsize=8)

    if save:
        fig.savefig(outfile, dpi=dpi, bbox_inches="tight", facecolor="white")
        print(f"Saved {outfile}")

    plt.show()
    return fig
