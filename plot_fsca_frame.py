#!/usr/bin/env python3
"""
plot_fsca_frame.py — Plot a single day of fractional snow covered area.

Two modes of operation:

  MODE 1 — Quick look from a single daily netCDF file (standalone):
      python plot_fsca_frame.py SPIRES_HIST_h09v04_MOD09GA061_20150201_V1.0.nc BRB_outline.shp
      python plot_fsca_frame.py SPIRES_HIST_h09v04_MOD09GA061_20150201_V1.0.nc --bbox -116.8 43.2 -115.0 44.4

      # With a cached mask (much faster after first run):
      python plot_fsca_frame.py SPIRES_HIST_h09v04_MOD09GA061_20150201_V1.0.nc \\
             --mask-file basin_mask_BRB_outline_h09v04.npz

  MODE 2 — Called programmatically with pre-processed data (used by animation script):
      from plot_fsca_frame import plot_fsca_frame
      fig, ax = plot_fsca_frame(frame_data=data_utm[:,:,t], frame_date=dates[t],
                                plot_context=ctx)

Data Citation:
    Rittger, K., et al. (2025). Historical MODIS/Terra L3 Global Daily 500m
    SIN Grid Snow Cover, Snow Albedo, and Snow Surface Properties.
    (SPIRES_HIST, V1). NSIDC. https://doi.org/10.7265/a3vr-c014

Dependencies:
    pip install netCDF4 numpy matplotlib shapefile
"""

import os
import re
import pickle
import argparse
import warnings
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

try:
    from netCDF4 import Dataset as nc4_Dataset, num2date
except ImportError:
    raise ImportError("netCDF4 required:  pip install netCDF4")

try:
    import shapefile  # pyshp
except ImportError:
    shapefile = None

# =========================================================================
#  APPEARANCE — edit these to taste
# =========================================================================
COLORMAP_NAME  = "parula_like"   # built below from matplotlib's viridis+plasma
FSCA_RANGE     = (0.0, 1.0)
BASIN_COLOR    = "k"
BASIN_WIDTH    = 1.2
SNOTEL_MARKER  = "^"
SNOTEL_SIZE    = 55
SNOTEL_FACE    = (1.0, 0.2, 0.2)
SNOTEL_EDGE    = "k"
SNOTEL_FONT    = 7
TITLE_FONT     = 14
LABEL_FONT     = 11
FIG_SIZE       = (10, 8)         # inches
NAN_COLOR      = (1, 1, 1)
# =========================================================================

# =========================================================================
#  CONSTANTS
# =========================================================================
R_EARTH    = 6371007.181
TILE_SIZE  = 1111950.5197
NPIX       = 2400
PIXEL_SIZE = TILE_SIZE / NPIX
H_TILE     = 9
V_TILE     = 4
UTM_ZONE   = 11
UTM_RES    = 500   # output grid resolution (m)

# =========================================================================
#  SNOTEL stations (edit / add / remove as needed)
# =========================================================================
SNOTEL = {
    "names":    ["Bogus Basin", "Mores Creek Summit", "Graham Guard Sta.",
                 "Jackson Peak", "Atlanta Summit", "Trinity Mtn.",
                 "Banner Summit", "Prairie"],
    "lat":      np.array([43+46/60, 43+56/60, 43+57/60, 44+3/60,
                          43+45/60, 43+38/60, 44+18/60, 43+49/60]),
    "lon":      np.array([-(116+6/60), -(115+40/60), -(115+16/60), -(115+27/60),
                          -(115+14/60), -(115+26/60), -(115+14/60), -(115+54/60)]),
    "id":       [978, 637, 496, 550, 306, 830, 312, 700],
    "elev_ft":  [6370, 6090, 5680, 7060, 7570, 7790, 7040, 5580],
}


# =====================================================================
#  Coordinate transforms (pure-numpy, no external projection library)
# =====================================================================
def latlon2sin(lat, lon, R=R_EARTH):
    """Geographic (deg) → MODIS sinusoidal (m)."""
    lr = np.deg2rad(lat); lnr = np.deg2rad(lon)
    return R * lnr * np.cos(lr), R * lr

def sin2latlon(x, y, R=R_EARTH):
    """MODIS sinusoidal (m) → geographic (deg)."""
    lr = y / R
    cl = np.cos(lr); cl[np.abs(cl) < 1e-10] = 1e-10
    return np.rad2deg(lr), np.rad2deg(x / (R * cl))

def latlon2utm(lat, lon, zone=UTM_ZONE):
    """Geographic (deg) → UTM (m), northern hemisphere only."""
    a = 6378137.0; f = 1/298.257223563
    e = np.sqrt(2*f - f**2); ep = e / np.sqrt(1 - e**2); k0 = 0.9996
    lon0 = np.deg2rad((zone - 1) * 6 - 180 + 3)
    lr = np.deg2rad(np.asarray(lat, dtype=np.float64))
    lnr = np.deg2rad(np.asarray(lon, dtype=np.float64))
    N = a / np.sqrt(1 - e**2 * np.sin(lr)**2)
    T = np.tan(lr)**2; C = ep**2 * np.cos(lr)**2
    A = (lnr - lon0) * np.cos(lr)
    M = a * ((1 - e**2/4 - 3*e**4/64 - 5*e**6/256) * lr
             - (3*e**2/8 + 3*e**4/32 + 45*e**6/1024) * np.sin(2*lr)
             + (15*e**4/256 + 45*e**6/1024) * np.sin(4*lr)
             - (35*e**6/3072) * np.sin(6*lr))
    easting = k0 * N * (A + (1-T+C)*A**3/6
              + (5-18*T+T**2+72*C-58*ep**2)*A**5/120) + 500000.0
    northing = k0 * (M + N * np.tan(lr) * (
               A**2/2 + (5-T+9*C+4*C**2)*A**4/24
               + (61-58*T+T**2+600*C-330*ep**2)*A**6/720))
    return easting, northing

def utm2latlon(easting, northing, zone=UTM_ZONE):
    """UTM (m) → geographic (deg), northern hemisphere only."""
    a = 6378137.0; f = 1/298.257223563
    e = np.sqrt(2*f - f**2); ep = e / np.sqrt(1 - e**2); k0 = 0.9996
    x = np.asarray(easting, dtype=np.float64) - 500000.0
    y = np.asarray(northing, dtype=np.float64)
    M = y / k0
    mu = M / (a * (1 - e**2/4 - 3*e**4/64 - 5*e**6/256))
    e1 = (1 - np.sqrt(1-e**2)) / (1 + np.sqrt(1-e**2))
    phi1 = (mu + (3*e1/2 - 27*e1**3/32)*np.sin(2*mu)
               + (21*e1**2/16 - 55*e1**4/32)*np.sin(4*mu)
               + (151*e1**3/96)*np.sin(6*mu))
    N1 = a / np.sqrt(1 - e**2*np.sin(phi1)**2)
    T1 = np.tan(phi1)**2; C1 = ep**2 * np.cos(phi1)**2
    R1 = a * (1 - e**2) / (1 - e**2*np.sin(phi1)**2)**1.5
    D = x / (N1 * k0)
    lat_r = phi1 - (N1*np.tan(phi1)/R1) * (
        D**2/2 - (5+3*T1+10*C1-4*C1**2-9*ep**2)*D**4/24
        + (61+90*T1+298*C1+45*T1**2-252*ep**2-3*C1**2)*D**6/720)
    lon0 = np.deg2rad((zone-1)*6-180+3)
    lon_r = ((D - (1+2*T1+C1)*D**3/6
              + (5-2*C1+28*T1-3*C1**2+8*ep**2+24*T1**2)*D**5/120)
             / np.cos(phi1)) + lon0
    return np.rad2deg(lat_r), np.rad2deg(lon_r)


# =====================================================================
#  Point-in-polygon (vectorised ray-casting)
# =====================================================================
def points_in_polygon(px, py, poly_x, poly_y):
    """Test whether points (px, py) lie inside polygon (poly_x, poly_y).
    Uses the ray-casting algorithm. Returns boolean array."""
    px = np.asarray(px, dtype=np.float64).ravel()
    py = np.asarray(py, dtype=np.float64).ravel()
    vx = np.asarray(poly_x, dtype=np.float64)
    vy = np.asarray(poly_y, dtype=np.float64)
    n = len(vx)
    inside = np.zeros(len(px), dtype=bool)
    for i in range(n):
        j = (i - 1) % n
        xi, yi = vx[i], vy[i]
        xj, yj = vx[j], vy[j]
        # Which points have the test ray crossing this edge?
        crossing = (yi > py) != (yj > py)
        if not np.any(crossing):
            continue
        # X-intercept of the edge at the y-value of each crossing point
        x_int = (xj - xi) * (py[crossing] - yi) / (yj - yi) + xi
        # Toggle inside/outside for points to the left of the intercept
        inside[crossing] ^= (px[crossing] < x_int)
    return inside


def points_in_polygons(px, py, polys_x, polys_y):
    """Test points against multiple polygon parts (union)."""
    result = np.zeros(np.asarray(px).ravel().shape, dtype=bool)
    for poly_x, poly_y in zip(polys_x, polys_y):
        result |= points_in_polygon(px, py, poly_x, poly_y)
    return result


# =====================================================================
#  Build a parula-like colormap (MATLAB's default)
# =====================================================================
def make_parula_cmap():
    """Approximate MATLAB parula with a blue→teal→yellow ramp."""
    # Key parula-like colors
    colors = [
        (0.2422, 0.1504, 0.6603),
        (0.2810, 0.3228, 0.9579),
        (0.1786, 0.5289, 0.9682),
        (0.0689, 0.6948, 0.8394),
        (0.1280, 0.7890, 0.5921),
        (0.5271, 0.8171, 0.2165),
        (0.8763, 0.7956, 0.1168),
        (0.9932, 0.9062, 0.1439),
    ]
    return mcolors.LinearSegmentedColormap.from_list("parula_like", colors, N=256)


# =====================================================================
#  Build / load basin mask
# =====================================================================
def _bbox_to_closed_polygon(min_x, min_y, max_x, max_y):
    if min_x >= max_x or min_y >= max_y:
        raise ValueError("Invalid bounding box: require min_x < max_x and min_y < max_y.")
    return np.array([
        [min_x, min_y],
        [max_x, min_y],
        [max_x, max_y],
        [min_x, max_y],
        [min_x, min_y],
    ], dtype=np.float64)


def _parse_clip_file(path):
    with open(path, "r", encoding="utf-8") as f:
        lines = [ln.strip() for ln in f if ln.strip()]

    if not lines:
        raise ValueError(
            "Invalid Bounding Box File: File must either contain 4 numbers on one line "
            "or a closed polygon with matching start and end points.")

    if len(lines) == 1:
        toks = lines[0].split()
        if len(toks) != 4:
            raise ValueError(
                "Invalid Bounding Box File: single-line bounding box file must contain "
                "4 space-separated numbers.")
        try:
            min_x, min_y, max_x, max_y = [float(v) for v in toks]
        except ValueError as exc:
            raise ValueError(
                "Invalid Bounding Box File: single-line bounding box file must contain "
                "4 space-separated numbers.") from exc
        return [_bbox_to_closed_polygon(min_x, min_y, max_x, max_y)]

    coords = []
    for i, line in enumerate(lines, start=1):
        toks = line.split()
        if len(toks) != 2:
            raise ValueError(
                f"Invalid Polygon File: line {i} must contain exactly 2 numbers.")
        try:
            coords.append([float(toks[0]), float(toks[1])])
        except ValueError as exc:
            raise ValueError(
                f"Invalid Polygon File: line {i} must contain exactly 2 numbers.") from exc
    poly = np.asarray(coords, dtype=np.float64)
    if poly.shape[0] < 4:
        raise ValueError("Invalid Polygon File: polygon file must include at least 4 coordinate lines.")
    if not np.allclose(poly[0], poly[-1]):
        raise ValueError(
            "Invalid Polygon File: polygon must be closed; first and last coordinates must match.")
    return [poly]


def _extract_polygons_xy(clip_input, min_y=None, max_x=None, max_y=None):
    if isinstance(clip_input, (int, float, np.number)):
        if min_y is None or max_x is None or max_y is None:
            raise ValueError(
                "Numeric clipping input requires four values: min_x min_y max_x max_y.")
        return [_bbox_to_closed_polygon(float(clip_input), float(min_y), float(max_x), float(max_y))]

    if isinstance(clip_input, (str, Path)):
        clip_path = Path(clip_input)
        if not clip_path.exists():
            raise FileNotFoundError(f"Clip input file not found: {clip_path}")
        if clip_path.suffix.lower() == ".shp":
            if shapefile is None:
                raise ImportError("pyshp required for shapefile reading:  pip install pyshp")
            sf = shapefile.Reader(str(clip_path))
            shapes = sf.shapes()
            out = []
            for shp in shapes:
                pts = np.array(shp.points, dtype=np.float64)
                parts = list(shp.parts) + [len(pts)]
                for i in range(len(parts) - 1):
                    out.append(pts[parts[i]:parts[i+1], :2])
            return out
        return _parse_clip_file(str(clip_path))

    raise ValueError(
        "Clipping input must be either a file path or 4 numeric values "
        "(min_x, min_y, max_x, max_y).")


def build_mask(clip_input, h_tile=H_TILE, v_tile=V_TILE, min_y=None, max_x=None, max_y=None):
    """Build mask in sinusoidal space from shapefile, clip file, or numeric bbox."""
    polygons_xy = _extract_polygons_xy(clip_input, min_y=min_y, max_x=max_x, max_y=max_y)

    all_xy = np.vstack(polygons_xy)
    is_utm = np.nanmax(np.abs(all_xy[:, 0])) > 1000

    x_origin = -R_EARTH * np.pi + h_tile * TILE_SIZE
    y_origin =  R_EARTH * np.pi / 2 - v_tile * TILE_SIZE

    all_poly_x_sin = []
    all_poly_y_sin = []
    all_x = []
    all_y = []

    for part_pts in polygons_xy:
        px, py = part_pts[:, 0], part_pts[:, 1]
        if is_utm:
            lat, lon = utm2latlon(px, py, UTM_ZONE)
        else:
            lat, lon = py, px
        sx, sy = latlon2sin(lat, lon)
        all_poly_x_sin.append(sx)
        all_poly_y_sin.append(sy)
        all_x.extend(sx)
        all_y.extend(sy)

    all_x = np.array(all_x)
    all_y = np.array(all_y)

    col_min = max(0, int(np.floor((all_x.min() - x_origin) / PIXEL_SIZE)) - 1)
    col_max = min(NPIX - 1, int(np.ceil((all_x.max() - x_origin) / PIXEL_SIZE)) + 2)
    row_min = max(0, int(np.floor((y_origin - all_y.max()) / PIXEL_SIZE)) - 1)
    row_max = min(NPIX - 1, int(np.ceil((y_origin - all_y.min()) / PIXEL_SIZE)) + 2)
    nrows = row_max - row_min + 1
    ncols = col_max - col_min + 1

    print(f"  Mask dimensions: {nrows} x {ncols}")

    # Pixel centers in sinusoidal coords
    cols = np.arange(col_min, col_max + 1)
    rows = np.arange(row_min, row_max + 1)
    cx = x_origin + (cols + 0.5) * PIXEL_SIZE
    cy = y_origin - (rows + 0.5) * PIXEL_SIZE
    cx_grid, cy_grid = np.meshgrid(cx, cy)

    mask = points_in_polygons(cx_grid.ravel(), cy_grid.ravel(),
                              all_poly_x_sin, all_poly_y_sin).reshape(nrows, ncols)

    print(f"  Basin contains {mask.sum()} pixels")

    return {
        "mask": mask,
        "row_min": row_min, "row_max": row_max,
        "col_min": col_min, "col_max": col_max,
        "nrows": nrows, "ncols": ncols,
        "all_poly_x_sin": all_poly_x_sin,
        "all_poly_y_sin": all_poly_y_sin,
        "utm_zone": UTM_ZONE,
        "x_origin": x_origin, "y_origin": y_origin,
    }


# =====================================================================
#  Read one frame from netCDF (handles any SPIRES dimension order)
# =====================================================================
def read_frame_nc(nc_path, varname, row_min, col_min, nrows, ncols, time_idx=0):
    """Read a 2D slice from a SPIRES netCDF file.

    Inspects dimension names to determine ordering. The SPIRES files have
    dimensions named 'time', 'x', and 'y'. Python netCDF4 may report them
    in any order. We slice x with col indices and y with row indices,
    then arrange the result as (row, col) = (y, x).
    """
    with nc4_Dataset(nc_path, "r") as ds:
        var = ds.variables[varname]
        dims = var.dimensions
        shape = var.shape
        ndim = var.ndim

        if ndim == 3:
            # Classify each dimension as time, x, or y
            dim_role = {}
            for d in range(3):
                dname = dims[d].lower()
                if "time" in dname or "day" in dname:
                    dim_role[d] = "time"
                elif dname == "x":
                    dim_role[d] = "x"
                elif dname == "y":
                    dim_role[d] = "y"

            # If names didn't resolve all three, fall back
            if len(dim_role) < 3:
                # Assume the size-1 dim is time
                for d in range(3):
                    if d not in dim_role:
                        if shape[d] == 1:
                            dim_role[d] = "time"
                # Remaining unassigned: first is x, second is y
                unassigned = [d for d in range(3) if d not in dim_role]
                labels = ["x", "y"]
                for d, lbl in zip(unassigned, labels):
                    dim_role[d] = lbl

            # Build the slice
            slices = [None, None, None]
            for d in range(3):
                role = dim_role[d]
                if role == "time":
                    slices[d] = time_idx
                elif role == "x":
                    slices[d] = slice(col_min, col_min + ncols)
                elif role == "y":
                    slices[d] = slice(row_min, row_min + nrows)

            data = np.squeeze(np.asarray(var[tuple(slices)], dtype=np.float64))

            # Determine output axis order: we need (row, col) = (y, x)
            # After squeeze (time removed), the remaining axes are in
            # the same relative order as the non-time dims
            spatial_order = [dim_role[d] for d in range(3) if dim_role[d] != "time"]
            if spatial_order == ["x", "y"]:
                # data is (x, y) = (cols, rows) → transpose
                data = data.T
            # else data is (y, x) = (rows, cols) → already correct

        elif ndim == 2:
            d0 = dims[0].lower()
            d1 = dims[1].lower()
            if d0 == "x" and d1 == "y":
                # (x, y) = (col, row)
                data = np.asarray(
                    var[col_min:col_min+ncols, row_min:row_min+nrows],
                    dtype=np.float64).T
            elif d0 == "y" and d1 == "x":
                # (y, x) = (row, col) — already correct
                data = np.asarray(
                    var[row_min:row_min+nrows, col_min:col_min+ncols],
                    dtype=np.float64)
            else:
                # Unknown names — assume (x, y)
                data = np.asarray(
                    var[col_min:col_min+ncols, row_min:row_min+nrows],
                    dtype=np.float64).T
        else:
            raise ValueError(f"Unexpected ndim={ndim} for {varname}")

    return data


# =====================================================================
#  Find fSCA variable in a netCDF file
# =====================================================================
def find_fsca_var(ds):
    """Return the best fSCA variable name from a netCDF4 Dataset."""
    vnames = list(ds.variables.keys())
    # Priority 1: exact 'snow_fraction'
    if "snow_fraction" in vnames:
        return "snow_fraction"
    # Priority 2: other exact names
    for name in ["fSCA", "fsca", "FSCA", "snow_fraction_on_ground",
                 "Snow_Fraction", "fractional_snow_cover", "snow_cover_fraction"]:
        if name in vnames:
            return name
    # Priority 3: viewable_snow_fraction
    if "viewable_snow_fraction" in vnames:
        print("  NOTE: Using viewable_snow_fraction (on-ground version not found)")
        return "viewable_snow_fraction"
    # Priority 4: fuzzy match
    for name in vnames:
        if "snow" in name.lower() and "frac" in name.lower():
            return name
    return None


# =====================================================================
#  Main plotting function
# =====================================================================
def plot_fsca_frame(nc_file=None, shapefile_path=None, mask_file=None, clip_bbox=None,
                    day_index=0, frame_data=None, frame_date=None,
                    plot_context=None, water_year=0, fig=None, ax=None,
                    visible=True, save_png=None):
    """Plot one day of fSCA over the Boise River Basin.

    Returns (fig, ax).
    """
    cmap = make_parula_cmap()
    cmap.set_bad(color=NAN_COLOR)

    if not visible:
        matplotlib.use("Agg")

    # -----------------------------------------------------------------
    #  MODE 2: pre-processed data from animation script
    # -----------------------------------------------------------------
    if plot_context is not None:
        ctx = plot_context
        frame_utm = frame_data
        dt = frame_date
        wy = water_year
        ek = ctx["ek"]
        nk = ctx["nk"]
        poly_utm_e = ctx["poly_utm_e"]
        poly_utm_n = ctx["poly_utm_n"]
        snotel_e = ctx["snotel_e"]
        snotel_n = ctx["snotel_n"]
        in_view = ctx["in_view"]
        utm_zone = ctx["utm_zone"]

    # -----------------------------------------------------------------
    #  MODE 1: read from netCDF file
    # -----------------------------------------------------------------
    else:
        if nc_file is None:
            raise ValueError("Provide nc_file (Mode 1) or frame_data+plot_context (Mode 2)")
        utm_zone = UTM_ZONE

        # --- Load or build basin mask ---
        if mask_file and os.path.exists(mask_file):
            print(f"Loading cached mask: {mask_file}")
            m = np.load(mask_file, allow_pickle=True)
            m = {k: m[k].item() if m[k].ndim == 0 else m[k] for k in m.files}
            # Handle pickled lists
            if "all_poly_x_sin" in m and isinstance(m["all_poly_x_sin"], np.ndarray):
                m["all_poly_x_sin"] = list(m["all_poly_x_sin"])
                m["all_poly_y_sin"] = list(m["all_poly_y_sin"])
        elif clip_bbox is not None:
            print("Building basin mask from numeric bounding box...")
            m = build_mask(clip_bbox[0], H_TILE, V_TILE, clip_bbox[1], clip_bbox[2], clip_bbox[3])
        elif shapefile_path:
            print("Building basin mask from clipping input...")
            m = build_mask(shapefile_path)
            # Cache it
            stem = Path(shapefile_path).stem
            cache = Path(shapefile_path).parent / f"basin_mask_{stem}_h{H_TILE:02d}v{V_TILE:02d}.npz"
            np.savez(str(cache), **{k: np.array(v, dtype=object) if isinstance(v, list) else v
                                    for k, v in m.items()})
            print(f"Mask cached: {cache}")
        else:
            raise ValueError("Provide clipping input file, --bbox, or mask_file for Mode 1.")

        mask = m["mask"]
        row_min = int(m["row_min"]); row_max = int(m["row_max"])
        col_min = int(m["col_min"]); col_max = int(m["col_max"])
        nrows = int(m["nrows"]); ncols = int(m["ncols"])
        x_origin = float(m["x_origin"]); y_origin = float(m["y_origin"])
        all_poly_x_sin = [np.asarray(p, dtype=np.float64) for p in m["all_poly_x_sin"]]
        all_poly_y_sin = [np.asarray(p, dtype=np.float64) for p in m["all_poly_y_sin"]]

        # --- Read fSCA ---
        print(f"Reading: {nc_file}")
        with nc4_Dataset(nc_file, "r") as ds:
            fsca_var = find_fsca_var(ds)
            if fsca_var is None:
                raise ValueError(f"No fSCA variable found. Available: {list(ds.variables.keys())}")
            print(f"  Using variable: {fsca_var}")
            var = ds.variables[fsca_var]
            print(f"  Variable: {fsca_var}, shape: {var.shape}, dims: {var.dimensions}")

            # Read attributes
            fill_val = getattr(var, "_FillValue", None)
            if fill_val is not None:
                fill_val = float(fill_val)
            mv = getattr(var, "missing_value", None)
            if mv is not None:
                fill_val = float(mv)
            vr = getattr(var, "valid_range", None)
            if vr is not None:
                valid_min, valid_max = float(vr[0]), float(vr[1])
            else:
                valid_min = float(getattr(var, "valid_min", 0))
                valid_max = float(getattr(var, "valid_max", 1))

            is_pct = valid_max > 1.5

            # Read date from time variable
            dt = None
            for tname in ["time", "Time", "date", "day"]:
                if tname in ds.variables:
                    try:
                        tv = ds.variables[tname]
                        t_units = getattr(tv, "units", "")
                        t_cal = getattr(tv, "calendar", "standard")
                        dates_nc = num2date(tv[day_index], t_units, t_cal)
                        dt = datetime(dates_nc.year, dates_nc.month, dates_nc.day)
                    except Exception:
                        pass
                    break

        # Read the data slice
        slice_data = read_frame_nc(nc_file, fsca_var, row_min, col_min,
                                   nrows, ncols, time_idx=day_index)

        # Read grain_size companion for unobserved pixel detection
        unobs = np.zeros_like(slice_data, dtype=bool)
        try:
            with nc4_Dataset(nc_file, "r") as ds:
                if "grain_size" in ds.variables:
                    gs = read_frame_nc(nc_file, "grain_size", row_min, col_min,
                                       nrows, ncols, time_idx=day_index)
                    unobs = (slice_data == 0) & (gs == 0)
        except Exception:
            pass

        # Extract date from filename if not found
        if dt is None:
            fn = Path(nc_file).stem
            m8 = re.search(r"(20\d{6})", fn)
            if m8:
                dt = datetime.strptime(m8.group(1), "%Y%m%d")
            else:
                m7 = re.search(r"(20\d{4,5})", fn)
                if m7 and len(m7.group(1)) == 7:
                    yr = int(m7.group(1)[:4]); doy = int(m7.group(1)[4:])
                    if 1 <= doy <= 366:
                        dt = datetime(yr, 1, 1) + timedelta(days=doy - 1)

        # --- Diagnostics ---
        raw_basin = slice_data[mask]
        raw_finite = raw_basin[np.isfinite(raw_basin)]
        n_unobs = unobs[mask].sum()
        print(f"  --- Data diagnostics (within basin, before filtering) ---")
        print(f"  _FillValue = {fill_val},  valid_range = [{valid_min}, {valid_max}]")
        print(f"  Total pixels in basin: {len(raw_basin)}")
        print(f"  NaN pixels:  {np.isnan(raw_basin).sum()}")
        print(f"  Unobserved pixels (swath gaps): {n_unobs} ({100*n_unobs/len(raw_basin):.1f}%)")
        if len(raw_finite) > 0:
            print(f"  Finite min:  {raw_finite.min():.4g}")
            print(f"  Finite max:  {raw_finite.max():.4g}")
            print(f"  Finite mean (all): {raw_finite.mean():.4g}")
            obs_vals = raw_basin[~unobs[mask] & np.isfinite(raw_basin)]
            if len(obs_vals) > 0:
                print(f"  Finite mean (observed only): {obs_vals.mean():.4g}")
        print(f"  ---------------------------------------------------")

        # --- Filter ---
        slice_data[~mask] = np.nan
        slice_data[unobs] = np.nan
        if fill_val is not None:
            slice_data[slice_data == fill_val] = np.nan
        slice_data[(slice_data < valid_min) | (slice_data > valid_max)] = np.nan
        if is_pct:
            slice_data /= 100.0

        # --- Reproject sinusoidal → UTM ---
        print(f"  Reprojecting to UTM Zone {UTM_ZONE}N...")

        # Get UTM extent from mask pixels
        sin_x_vec = x_origin + (np.arange(col_min, col_max+1) + 0.5) * PIXEL_SIZE
        sin_y_vec = y_origin - (np.arange(row_min, row_max+1) + 0.5) * PIXEL_SIZE
        sx_g, sy_g = np.meshgrid(sin_x_vec, sin_y_vec)
        lat_g, lon_g = sin2latlon(sx_g, sy_g)
        utm_eg, utm_ng = latlon2utm(lat_g, lon_g)

        ue_min = utm_eg[mask].min() - PIXEL_SIZE
        ue_max = utm_eg[mask].max() + PIXEL_SIZE
        un_min = utm_ng[mask].min() - PIXEL_SIZE
        un_max = utm_ng[mask].max() + PIXEL_SIZE

        utm_e_out = np.arange(ue_min, ue_max + UTM_RES, UTM_RES)
        utm_n_out = np.arange(un_max, un_min - UTM_RES, -UTM_RES)
        UE, UN = np.meshgrid(utm_e_out, utm_n_out)
        nr2, nc2 = len(utm_n_out), len(utm_e_out)

        # Inverse-project UTM → sinusoidal (full tile indexing)
        la2, lo2 = utm2latlon(UE.ravel(), UN.ravel())
        sx2, sy2 = latlon2sin(la2, lo2)

        tile_col = np.round((sx2 - x_origin) / PIXEL_SIZE + 0.5).astype(int)
        tile_row = np.round((y_origin - sy2) / PIXEL_SIZE + 0.5).astype(int)

        valid_tile = (tile_col >= 1) & (tile_col <= NPIX) & \
                     (tile_row >= 1) & (tile_row <= NPIX)

        rd_col_min = max(0, tile_col[valid_tile].min() - 1)
        rd_col_max = min(NPIX - 1, tile_col[valid_tile].max() + 1)
        rd_row_min = max(0, tile_row[valid_tile].min() - 1)
        rd_row_max = min(NPIX - 1, tile_row[valid_tile].max() + 1)
        rd_nrows = rd_row_max - rd_row_min + 1
        rd_ncols = rd_col_max - rd_col_min + 1

        print(f"  Original mask subset: cols [{col_min}:{col_max}], rows [{row_min}:{row_max}]")
        print(f"  Expanded read range:  cols [{rd_col_min}:{rd_col_max}], rows [{rd_row_min}:{rd_row_max}]")

        # Local indices into expanded array
        local_col = tile_col - rd_col_min
        local_row = tile_row - rd_row_min
        vl = valid_tile & (local_col >= 0) & (local_col < rd_ncols) & \
                          (local_row >= 0) & (local_row < rd_nrows)

        local_col = local_col.reshape(nr2, nc2)
        local_row = local_row.reshape(nr2, nc2)
        vl = vl.reshape(nr2, nc2)
        sx2_g = sx2.reshape(nr2, nc2)
        sy2_g = sy2.reshape(nr2, nc2)

        # Build lookup with inpolygon test
        print("  Building lookup with basin polygon test...")
        src_idx = np.full((nr2, nc2), -1, dtype=np.int64)
        # Vectorised: test all valid points at once
        vl_flat = vl.ravel()
        valid_indices = np.where(vl_flat)[0]
        if len(valid_indices) > 0:
            sx_test = sx2_g.ravel()[valid_indices]
            sy_test = sy2_g.ravel()[valid_indices]
            in_basin = points_in_polygons(sx_test, sy_test,
                                          all_poly_x_sin, all_poly_y_sin)
            lc = local_col.ravel()[valid_indices]
            lr = local_row.ravel()[valid_indices]
            lin = lr * rd_ncols + lc
            src_flat = src_idx.ravel()
            src_flat[valid_indices[in_basin]] = lin[in_basin]
            src_idx = src_flat.reshape(nr2, nc2)
        mask_out = src_idx >= 0

        # Re-read with expanded range
        print(f"  Re-reading data with expanded range [{rd_nrows}x{rd_ncols}]...")
        slice_exp = read_frame_nc(nc_file, fsca_var, rd_row_min, rd_col_min,
                                  rd_nrows, rd_ncols, time_idx=day_index)

        # Filter
        if fill_val is not None:
            slice_exp[slice_exp == fill_val] = np.nan
        slice_exp[(slice_exp < valid_min) | (slice_exp > valid_max)] = np.nan

        # Unobserved pixel detection on expanded data
        try:
            with nc4_Dataset(nc_file, "r") as ds:
                if "grain_size" in ds.variables:
                    gs_exp = read_frame_nc(nc_file, "grain_size", rd_row_min, rd_col_min,
                                           rd_nrows, rd_ncols, time_idx=day_index)
                    slice_exp[(slice_exp == 0) & (gs_exp == 0)] = np.nan
        except Exception:
            pass

        if is_pct:
            slice_exp /= 100.0

        # Reproject via lookup
        frame_utm = np.full((nr2, nc2), np.nan)
        frame_utm[mask_out] = slice_exp.ravel()[src_idx[mask_out]]

        ek = (ue_min / 1000, ue_max / 1000)
        nk = (un_min / 1000, un_max / 1000)

        # Basin polygons in UTM
        poly_utm_e = []; poly_utm_n = []
        for psx, psy in zip(all_poly_x_sin, all_poly_y_sin):
            plat, plon = sin2latlon(psx, psy)
            pe, pn = latlon2utm(plat, plon)
            poly_utm_e.append(pe)
            poly_utm_n.append(pn)

        # SNOTEL in UTM
        snotel_e, snotel_n = latlon2utm(SNOTEL["lat"], SNOTEL["lon"])
        buf = 2000
        in_view = ((snotel_e >= ue_min - buf) & (snotel_e <= ue_max + buf) &
                   (snotel_n >= un_min - buf) & (snotel_n <= un_max + buf))

        # Water year
        if dt is not None:
            wy = dt.year + 1 if dt.month >= 10 else dt.year
        else:
            wy = 0

    if water_year > 0:
        wy = water_year

    # =================================================================
    #  PLOT THE FRAME
    # =================================================================
    if fig is None:
        fig, ax = plt.subplots(1, 1, figsize=FIG_SIZE)
    else:
        fig.clf()
        ax = fig.add_subplot(111)

    # Image
    extent_km = [ek[0], ek[1], nk[0], nk[1]]
    im = ax.imshow(frame_utm, cmap=cmap, vmin=FSCA_RANGE[0], vmax=FSCA_RANGE[1],
                   extent=extent_km, aspect="equal", origin="upper",
                   interpolation="nearest")

    ax.set_xlim(ek)
    ax.set_ylim(nk)

    # Basin boundary
    for pe, pn in zip(poly_utm_e, poly_utm_n):
        ax.plot(np.array(pe)/1000, np.array(pn)/1000,
                color=BASIN_COLOR, linewidth=BASIN_WIDTH)

    # SNOTEL stations
    for si, name in enumerate(SNOTEL["names"]):
        if in_view[si]:
            ax.scatter(snotel_e[si]/1000, snotel_n[si]/1000,
                       marker=SNOTEL_MARKER, s=SNOTEL_SIZE,
                       facecolors=SNOTEL_FACE, edgecolors=SNOTEL_EDGE,
                       linewidths=1.0, zorder=5)
            ax.annotate(name, (snotel_e[si]/1000 + 0.8, snotel_n[si]/1000),
                        fontsize=SNOTEL_FONT, fontweight="bold",
                        color=(0.1, 0.1, 0.1),
                        bbox=dict(boxstyle="round,pad=0.2", fc="white", alpha=0.7, ec="none"))

    # Labels
    ax.set_xlabel(f"Easting (km, UTM Zone {utm_zone}N)", fontsize=LABEL_FONT)
    ax.set_ylabel("Northing (km)", fontsize=LABEL_FONT)

    # Title
    if dt is not None:
        date_str = dt.strftime("%B %d, %Y")
    else:
        date_str = f"Day index {day_index}"
    if wy > 0:
        ax.set_title(f"Boise River Basin - Fractional Snow Covered Area  (WY {wy})\n{date_str}",
                     fontsize=TITLE_FONT, fontweight="bold")
    else:
        ax.set_title(f"Boise River Basin - Fractional Snow Covered Area\n{date_str}",
                     fontsize=TITLE_FONT, fontweight="bold")

    # Colorbar
    cb = fig.colorbar(im, ax=ax)
    cb.set_label("Fractional Snow Covered Area", fontsize=LABEL_FONT)

    # Basin-mean annotation
    vd = frame_utm[np.isfinite(frame_utm)]
    if len(vd) > 0:
        ax.text(ek[0] + 0.02*(ek[1]-ek[0]), nk[0] + 0.05*(nk[1]-nk[0]),
                f"Basin mean fSCA: {vd.mean():.3f}",
                fontsize=LABEL_FONT,
                bbox=dict(fc="white", alpha=0.8, ec="grey"))

    # Save
    if save_png:
        fig.savefig(save_png, dpi=150, bbox_inches="tight")
        print(f"Saved: {save_png}")

    return fig, ax


# =====================================================================
#  CLI entry point
# =====================================================================
def main():
    parser = argparse.ArgumentParser(
        description="Plot a single day of SPIReS fSCA over the Boise River Basin.")
    parser.add_argument("nc_file", help="Path to SPIRES netCDF file")
    parser.add_argument("shapefile", nargs="?", default=None,
                        help="Path to clipping file (.shp, bbox file, or polygon file)")
    parser.add_argument("--bbox", nargs=4, type=float, default=None,
                        metavar=("MIN_X", "MIN_Y", "MAX_X", "MAX_Y"),
                        help="Numeric clipping bbox values")
    parser.add_argument("--mask-file", default=None,
                        help="Path to cached mask (.npz)")
    parser.add_argument("--day-index", type=int, default=0,
                        help="Time index within the file (default: 0)")
    parser.add_argument("--save-png", default=None,
                        help="Save figure to this PNG path")
    parser.add_argument("--water-year", type=int, default=0,
                        help="Water year label for the title")
    args = parser.parse_args()
    if args.bbox is not None and args.shapefile is not None:
        parser.error("Provide either clipping file path or --bbox, not both.")

    fig, ax = plot_fsca_frame(
        nc_file=args.nc_file,
        shapefile_path=args.shapefile,
        mask_file=args.mask_file,
        clip_bbox=args.bbox,
        day_index=args.day_index,
        save_png=args.save_png,
        water_year=args.water_year,
    )
    # plt.show()


if __name__ == "__main__":
    main()
