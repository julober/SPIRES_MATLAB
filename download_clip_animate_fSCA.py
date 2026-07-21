#!/usr/bin/env python3
"""
download_clip_animate_fSCA.py — Download, clip, and animate SPIReS fSCA.

Downloads SPIRES HIST V01 fractional snow covered area (fSCA) data from:
    ftp://dtn.rc.colorado.edu/shares/snow-today/gridded_data/SPIRES_HIST_V01

Clips the data to the Boise River Basin using a user-supplied clipping geometry,
reprojects from MODIS sinusoidal to UTM, and creates an MP4 animation
and/or KMZ (Google Earth overlay) for a user-specified water year.

Usage:
    python download_clip_animate_fSCA.py 2015 BRB_outline.shp
    python download_clip_animate_fSCA.py 2015 --bbox -116.8 43.2 -115.0 44.4
    python download_clip_animate_fSCA.py 2020 BRB_outline.shp --format both --fps 15

    The year argument is the WATER YEAR:
      WY 2020 = Oct 1 2019 – Sep 30 2020
      WY 2015 = Oct 1 2014 – Sep 30 2015

Data Citation:
    Rittger, K., et al. (2025). Historical MODIS/Terra L3 Global Daily 500m
    SIN Grid Snow Cover, Snow Albedo, and Snow Surface Properties.
    (SPIRES_HIST, V1). NSIDC. https://doi.org/10.7265/a3vr-c014

Dependencies:
    pip install netCDF4 numpy matplotlib shapefile imageio[ffmpeg] Pillow
"""

import os
import re
import sys
import glob
import shutil
import zipfile
import argparse
from ftplib import FTP
from pathlib import Path
from datetime import datetime, timedelta

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from netCDF4 import Dataset as nc4_Dataset, num2date

try:
    from PIL import Image
    HAS_PIL = True
except ImportError:
    HAS_PIL = False

# Import helpers from plot_fsca_frame
from plot_fsca_frame import (
    latlon2sin, sin2latlon, latlon2utm, utm2latlon,
    points_in_polygons, build_mask, read_frame_nc,
    find_fsca_var, make_parula_cmap, plot_fsca_frame,
    R_EARTH, TILE_SIZE, NPIX, PIXEL_SIZE, H_TILE, V_TILE, UTM_ZONE, SNOTEL,
)


# =====================================================================
#  FTP download
# =====================================================================
def discover_and_download(ftp_host, ftp_dir, year, tile, download_dir):
    """Download SPIRES netCDF files for a given year/tile from FTP.
    Falls back to locally cached files if FTP is unreachable."""
    nc_files = []
    year_str = str(year)

    try:
        ftp = FTP(ftp_host, timeout=30)
        ftp.login()
    except Exception as e:
        print(f"  FTP connection failed: {e}")
        print(f"  Falling back to local files in {download_dir}")
        return sorted(glob.glob(os.path.join(download_dir, "*.nc")))

    try:
        ftp.cwd(ftp_dir)
        items = []
        ftp.retrlines("LIST", items.append)

        def parse_listing(lines):
            result = []
            for line in lines:
                parts = line.split()
                if len(parts) >= 9:
                    name = " ".join(parts[8:])
                    is_dir = line.startswith("d")
                    result.append((name, is_dir))
            return result

        def download_file(ftp_obj, remote_name, local_dir):
            local_path = os.path.join(local_dir, remote_name)
            if os.path.exists(local_path):
                print(f"    Already downloaded: {remote_name}")
            else:
                print(f"    Downloading: {remote_name}")
                with open(local_path, "wb") as f:
                    ftp_obj.retrbinary(f"RETR {remote_name}", f.write)
            return local_path

        entries = parse_listing(items)

        # Strategy 1: top-level files matching tile+year
        for name, is_dir in entries:
            if (not is_dir and tile in name and year_str in name
                    and name.lower().endswith(".nc")):
                lp = download_file(ftp, name, download_dir)
                nc_files.append(lp)
        if nc_files:
            ftp.quit(); return nc_files

        # Strategy 2: tile subdirectory
        tile_dirs = [n for n, d in entries if d and n == tile]
        if tile_dirs:
            ftp.cwd(tile)
            sub_items = []; ftp.retrlines("LIST", sub_items.append)
            sub_entries = parse_listing(sub_items)
            for name, is_dir in sub_entries:
                if not is_dir and year_str in name and name.lower().endswith(".nc"):
                    nc_files.append(download_file(ftp, name, download_dir))
            if nc_files:
                ftp.quit(); return nc_files
            yr_dirs = [n for n, d in sub_entries if d and n == year_str]
            if yr_dirs:
                ftp.cwd(year_str)
                yr_items = []; ftp.retrlines("LIST", yr_items.append)
                for name, is_dir in parse_listing(yr_items):
                    if not is_dir and name.lower().endswith(".nc"):
                        nc_files.append(download_file(ftp, name, download_dir))
            ftp.cwd(ftp_dir)
            if nc_files:
                ftp.quit(); return nc_files

        # Strategy 3: year subdirectory at top level
        yr_top = [n for n, d in entries if d and n == year_str]
        if yr_top:
            ftp.cwd(year_str)
            yr_items = []; ftp.retrlines("LIST", yr_items.append)
            for name, is_dir in parse_listing(yr_items):
                if not is_dir and tile in name and name.lower().endswith(".nc"):
                    nc_files.append(download_file(ftp, name, download_dir))
            if not nc_files:
                sub2 = parse_listing(yr_items)
                tile_sub = [n for n, d in sub2 if d and n == tile]
                if tile_sub:
                    ftp.cwd(tile)
                    ti_items = []; ftp.retrlines("LIST", ti_items.append)
                    for name, is_dir in parse_listing(ti_items):
                        if not is_dir and name.lower().endswith(".nc"):
                            nc_files.append(download_file(ftp, name, download_dir))

        ftp.quit()
    except Exception as e:
        print(f"  FTP error: {e}")
        try: ftp.quit()
        except Exception: pass
        nc_files = sorted(glob.glob(os.path.join(download_dir, "*.nc")))

    if not nc_files:
        print(f"  WARNING: No files found for {tile}/{year}")
        nc_files = sorted(glob.glob(os.path.join(download_dir, "*.nc")))

    return nc_files


# =====================================================================
#  Scan netCDF files for dates within water year
# =====================================================================
def extract_fsca_wy(nc_files, wy_start, wy_end):
    """Scan netCDF files and return (dates, file_paths, time_indices, fsca_varname)."""
    dates, nc_paths_out, time_indices_out = [], [], []
    fsca_varname = None

    for nc_path in nc_files:
        fname = Path(nc_path).stem
        print(f"  Scanning: {fname}")
        try:
            ds = nc4_Dataset(nc_path, "r")
        except Exception as e:
            print(f"    Skipping: {e}"); continue

        fvar = find_fsca_var(ds)
        if fvar is None:
            print("    WARNING: No fSCA variable found."); ds.close(); continue
        if fsca_varname is None:
            fsca_varname = fvar

        var = ds.variables[fvar]

        if var.ndim == 3:
            # Find time dimension by name (not by position — Python may reorder)
            dims = var.dimensions
            time_dim_idx = None
            for d in range(3):
                dname = dims[d].lower()
                if "time" in dname or "day" in dname:
                    time_dim_idx = d; break
            if time_dim_idx is None:
                # Fallback: the dimension with size 1
                for d in range(3):
                    if var.shape[d] == 1:
                        time_dim_idx = d; break
                if time_dim_idx is None:
                    time_dim_idx = 0  # last resort

            nt = var.shape[time_dim_idx]
            time_var = None
            for tname in ["time", "Time", "date", "day"]:
                if tname in ds.variables:
                    time_var = ds.variables[tname]; break

            for t in range(nt):
                dt = None
                if time_var is not None:
                    try:
                        t_units = getattr(time_var, "units", "")
                        t_cal = getattr(time_var, "calendar", "standard")
                        dt_nc = num2date(time_var[t], t_units, t_cal)
                        dt = datetime(dt_nc.year, dt_nc.month, dt_nc.day)
                    except Exception: pass
                if dt is None:
                    m8 = re.search(r"(20\d{6})", fname)
                    if m8:
                        base = datetime.strptime(m8.group(1), "%Y%m%d")
                        dt = base + timedelta(days=t) if nt > 1 else base
                    else:
                        dt = wy_start + timedelta(days=t)
                if wy_start <= dt <= wy_end:
                    dates.append(dt); nc_paths_out.append(nc_path)
                    time_indices_out.append(t)

        elif var.ndim == 2 or (var.ndim == 3 and min(var.shape) == 1):
            dt = None
            m8 = re.search(r"(20\d{6})", fname)
            if m8:
                dt = datetime.strptime(m8.group(1), "%Y%m%d")
            else:
                m7 = re.search(r"(\d{7})", fname)
                if m7:
                    yr = int(m7.group(1)[:4]); doy = int(m7.group(1)[4:])
                    if 1 <= doy <= 366:
                        dt = datetime(yr, 1, 1) + timedelta(days=doy - 1)
            if dt is not None and wy_start <= dt <= wy_end:
                dates.append(dt); nc_paths_out.append(nc_path)
                time_indices_out.append(0)
        ds.close()

    if dates:
        order = np.argsort(dates)
        dates = [dates[i] for i in order]
        nc_paths_out = [nc_paths_out[i] for i in order]
        time_indices_out = [time_indices_out[i] for i in order]

    return dates, nc_paths_out, time_indices_out, fsca_varname


# =====================================================================
#  Create MP4 animation
# =====================================================================
def create_movie(dates, data_utm, mask_out, output_path, water_year, fps,
                 utm_e_min, utm_e_max, utm_n_min, utm_n_max,
                 poly_utm_e, poly_utm_n, utm_zone, snotel_e, snotel_n):
    """Create MP4 animation using plot_fsca_frame in Mode 2."""
    from matplotlib.animation import FFMpegWriter

    nt = len(dates)
    print(f"  Creating animation: {nt} frames at {fps} fps")

    buf = 2000
    ctx = {
        "ek": (utm_e_min/1000, utm_e_max/1000),
        "nk": (utm_n_min/1000, utm_n_max/1000),
        "poly_utm_e": poly_utm_e, "poly_utm_n": poly_utm_n,
        "snotel_e": snotel_e, "snotel_n": snotel_n,
        "in_view": ((snotel_e >= utm_e_min-buf) & (snotel_e <= utm_e_max+buf) &
                    (snotel_n >= utm_n_min-buf) & (snotel_n <= utm_n_max+buf)),
        "utm_zone": utm_zone,
    }

    fig, ax = plt.subplots(1, 1, figsize=(10, 8))
    writer = FFMpegWriter(fps=fps, metadata={"title": f"BRB fSCA WY{water_year}"})

    with writer.saving(fig, output_path, dpi=100):
        for t in range(nt):
            if t % 30 == 0 or t == nt - 1:
                print(f"    Frame {t+1}/{nt}: {dates[t].strftime('%Y-%m-%d')}")
            plot_fsca_frame(frame_data=data_utm[:,:,t], frame_date=dates[t],
                            plot_context=ctx, water_year=water_year,
                            fig=fig, ax=ax, visible=False)
            writer.grab_frame()

    plt.close(fig)
    print(f"  Animation saved: {output_path}")


# =====================================================================
#  Create KMZ for Google Earth
# =====================================================================
def create_kmz(dates, data_utm, mask_out, output_path, water_year,
               lat_north, lat_south, lon_west, lon_east):
    """Create a KMZ with time-stamped ground overlays for Google Earth."""
    if not HAS_PIL:
        print("  WARNING: Pillow not installed — skipping KMZ. pip install Pillow")
        return

    stem = Path(output_path).stem
    tmp_dir = Path(output_path).parent / f"{stem}_tmp"
    img_dir = tmp_dir / "images"
    if tmp_dir.exists():
        shutil.rmtree(tmp_dir)
    img_dir.mkdir(parents=True)

    nt = len(dates)
    print(f"  Rendering {nt} overlay images...")
    cmap = make_parula_cmap()

    img_names = []
    for t in range(nt):
        if t % 30 == 0 or t == nt - 1:
            print(f"    Image {t+1}/{nt}: {dates[t].strftime('%Y-%m-%d')}")
        frame = data_utm[:,:,t]
        nr, nc = frame.shape
        cidx = np.clip(np.round(frame * 255).astype(int), 0, 255)

        rgba = np.zeros((nr, nc, 4), dtype=np.uint8)
        valid = mask_out & np.isfinite(frame)
        colors = (cmap(cidx[valid] / 255.0) * 255).astype(np.uint8)
        rgba[valid, :3] = colors[:, :3]
        rgba[valid, 3] = np.where(frame[valid] < 0.01, 0, 200).astype(np.uint8)

        fname = f"fsca_{dates[t].strftime('%Y%m%d')}.png"
        Image.fromarray(rgba, "RGBA").save(str(img_dir / fname))
        img_names.append(fname)

    # KML
    print("  Writing KML...")
    kml = ['<?xml version="1.0" encoding="UTF-8"?>',
           '<kml xmlns="http://www.opengis.net/kml/2.2">', '<Document>',
           f'  <n>Boise River Basin fSCA — WY {water_year}</n>',
           '  <description>Daily fSCA from SPIReS HIST V01. '
           'Rittger et al. (2025), NSIDC, doi:10.7265/a3vr-c014</description>']

    for t in range(nt):
        ds = dates[t].strftime("%Y-%m-%d")
        de = (dates[t+1] if t < nt-1 else dates[t]+timedelta(days=1)).strftime("%Y-%m-%d")
        kml += [f'  <GroundOverlay><n>fSCA {ds}</n>',
                f'    <TimeSpan><begin>{ds}T00:00:00Z</begin><end>{de}T00:00:00Z</end></TimeSpan>',
                f'    <Icon><href>images/{img_names[t]}</href></Icon>',
                f'    <LatLonBox><north>{lat_north:.6f}</north><south>{lat_south:.6f}</south>'
                f'<east>{lon_east:.6f}</east><west>{lon_west:.6f}</west></LatLonBox>',
                '  </GroundOverlay>']

    kml += ['  <Folder><n>SNOTEL Stations</n>',
            '    <Style id="s"><IconStyle><color>ff0000ff</color><scale>0.8</scale>'
            '<Icon><href>http://maps.google.com/mapfiles/kml/shapes/triangle.png</href>'
            '</Icon></IconStyle></Style>']
    for si, name in enumerate(SNOTEL["names"]):
        kml.append(f'    <Placemark><n>{name} (#{SNOTEL["id"][si]})</n>'
                   f'<description>Elev: {SNOTEL["elev_ft"][si]} ft</description>'
                   f'<styleUrl>#s</styleUrl>'
                   f'<Point><coordinates>{SNOTEL["lon"][si]:.6f},{SNOTEL["lat"][si]:.6f},0'
                   f'</coordinates></Point></Placemark>')
    kml += ['  </Folder>', '</Document>', '</kml>']

    (tmp_dir / "doc.kml").write_text("\n".join(kml), encoding="utf-8")

    print("  Packaging KMZ...")
    with zipfile.ZipFile(str(output_path), "w", zipfile.ZIP_DEFLATED) as zf:
        zf.write(str(tmp_dir / "doc.kml"), "doc.kml")
        for img_name in img_names:
            zf.write(str(img_dir / img_name), f"images/{img_name}")
    shutil.rmtree(tmp_dir)
    print(f"  KMZ created: {output_path}")


# =====================================================================
#  Main pipeline
# =====================================================================
def main():
    parser = argparse.ArgumentParser(
        description="Download, clip, and animate SPIReS fSCA for the Boise River Basin.")
    parser.add_argument("water_year", type=int,
                        help="Water year (e.g. 2015 = Oct 2014 – Sep 2015)")
    parser.add_argument("clip_input", nargs="?", default=None,
                        help="Path to clipping file (.shp, bbox file, or polygon file)")
    parser.add_argument("--bbox", nargs=4, type=float, default=None,
                        metavar=("MIN_X", "MIN_Y", "MAX_X", "MAX_Y"),
                        help="Numeric clipping bbox values")
    parser.add_argument("--output-dir", default="./spires_output")
    parser.add_argument("--fps", type=int, default=10)
    parser.add_argument("--tile", default="h09v04")
    parser.add_argument("--format", choices=["mp4", "kmz", "both"], default="both")
    args = parser.parse_args()
    if args.bbox is None and args.clip_input is None:
        parser.error("Provide either clip_input file path or --bbox MIN_X MIN_Y MAX_X MAX_Y.")
    if args.bbox is not None and args.clip_input is not None:
        parser.error("Provide either clip_input or --bbox, not both.")

    water_year = args.water_year
    assert 2001 <= water_year <= 2025, "Water year must be 2001-2025"

    FTP_HOST = "dtn.rc.colorado.edu"
    FTP_DIR  = "/shares/snow-today/gridded_data/SPIRES_HIST_V01"
    wy_start = datetime(water_year - 1, 10, 1)
    wy_end   = datetime(water_year, 9, 30)
    tile = args.tile; h_tile = int(tile[1:3]); v_tile = int(tile[4:6])

    print("=" * 66)
    print("  SPIRES fSCA - Download, Clip, and Animate (Python)")
    print("=" * 66)
    print(f"  Water Year: {water_year}  (Oct 1 {water_year-1} – Sep 30 {water_year})")
    clip_desc = args.clip_input if args.clip_input is not None else f"bbox={args.bbox}"
    print(f"  Tile: {tile}  Clip: {clip_desc}  Output: {args.output_dir}")
    print("=" * 66)

    os.makedirs(args.output_dir, exist_ok=True)
    dl_dir = os.path.join(args.output_dir, "downloads")
    os.makedirs(dl_dir, exist_ok=True)

    # STEP 1: Mask
    print("\n[1/4] Basin mask...")
    if args.clip_input is not None:
        clip_tag = Path(args.clip_input).stem
    else:
        clip_tag = re.sub(r"[^0-9A-Za-z._-]+", "_", "_".join(str(v) for v in args.bbox))
    mask_file = os.path.join(args.output_dir, f"basin_mask_{clip_tag}_{tile}.npz")
    if os.path.exists(mask_file):
        print(f"  Loading cached mask: {mask_file}")
        mf = np.load(mask_file, allow_pickle=True)
        m = {k: mf[k].item() if mf[k].ndim == 0 else mf[k] for k in mf.files}
        if isinstance(m.get("all_poly_x_sin"), np.ndarray):
            m["all_poly_x_sin"] = list(m["all_poly_x_sin"])
            m["all_poly_y_sin"] = list(m["all_poly_y_sin"])
    else:
        print("  Building mask from clipping input...")
        if args.bbox is not None:
            m = build_mask(args.bbox[0], h_tile, v_tile, args.bbox[1], args.bbox[2], args.bbox[3])
        else:
            m = build_mask(args.clip_input, h_tile, v_tile)
        np.savez(mask_file, **{k: np.array(v, dtype=object) if isinstance(v, list) else v
                               for k, v in m.items()})
        print(f"  Mask cached: {mask_file}")

    mask = m["mask"]; row_min = int(m["row_min"]); row_max = int(m["row_max"])
    col_min = int(m["col_min"]); col_max = int(m["col_max"])
    nrows = int(m["nrows"]); ncols = int(m["ncols"])
    x_origin = float(m["x_origin"]); y_origin = float(m["y_origin"])
    all_poly_x_sin = [np.asarray(p, dtype=np.float64) for p in m["all_poly_x_sin"]]
    all_poly_y_sin = [np.asarray(p, dtype=np.float64) for p in m["all_poly_y_sin"]]
    utm_zone = int(m.get("utm_zone", UTM_ZONE))
    print(f"  Basin: {mask.sum()} pixels ({mask.sum()*0.25:.1f} km²)")

    # STEP 2: Download
    print(f"\n[2/4] Downloading SPIRES data for WY{water_year}...")
    nc_y1 = discover_and_download(FTP_HOST, FTP_DIR, water_year-1, tile, dl_dir)
    nc_y2 = discover_and_download(FTP_HOST, FTP_DIR, water_year, tile, dl_dir)
    nc_files = sorted(set(nc_y1 + nc_y2))
    if not nc_files:
        print("ERROR: No data files found."); sys.exit(1)

    # STEP 3: Extract
    print(f"\n[3/4] Extracting fSCA data for WY{water_year}...")
    dates, nc_paths, tidxs, fsca_var = extract_fsca_wy(nc_files, wy_start, wy_end)
    if not dates:
        print("ERROR: No fSCA data extracted."); sys.exit(1)
    print(f"  Found {len(dates)} time steps: {dates[0]:%Y-%m-%d} to {dates[-1]:%Y-%m-%d}")

    # STEP 4: Reproject
    fill_val = None; valid_min = 0; valid_max = 1
    with nc4_Dataset(nc_paths[0], "r") as ds:
        if fsca_var in ds.variables:
            v = ds.variables[fsca_var]
            fv = getattr(v, "_FillValue", getattr(v, "missing_value", None))
            if fv is not None: fill_val = float(fv)
            vr = getattr(v, "valid_range", None)
            if vr is not None: valid_min, valid_max = float(vr[0]), float(vr[1])
            else:
                valid_min = float(getattr(v, "valid_min", 0))
                valid_max = float(getattr(v, "valid_max", 1))
    is_pct = valid_max > 1.5
    print(f"  Attributes: fill={fill_val}, range=[{valid_min},{valid_max}]")

    sin_x = x_origin + (np.arange(col_min, col_max+1) + 0.5) * PIXEL_SIZE
    sin_y = y_origin - (np.arange(row_min, row_max+1) + 0.5) * PIXEL_SIZE
    sxg, syg = np.meshgrid(sin_x, sin_y)
    lg, lng = sin2latlon(sxg, syg)
    ueg, ung = latlon2utm(lg, lng, utm_zone)

    ue_min = ueg[mask].min()-PIXEL_SIZE; ue_max = ueg[mask].max()+PIXEL_SIZE
    un_min = ung[mask].min()-PIXEL_SIZE; un_max = ung[mask].max()+PIXEL_SIZE

    utm_e_out = np.arange(ue_min, ue_max+500, 500)
    utm_n_out = np.arange(un_max, un_min-500, -500)
    UE, UN = np.meshgrid(utm_e_out, utm_n_out)
    nr, nc = len(utm_n_out), len(utm_e_out)

    print(f"\n[4/5] Reprojecting to UTM ({nr}x{nc} grid)...")
    la2, lo2 = utm2latlon(UE.ravel(), UN.ravel(), utm_zone)
    sx2, sy2 = latlon2sin(la2, lo2)

    tc = np.round((sx2 - x_origin)/PIXEL_SIZE + 0.5).astype(int)
    tr = np.round((y_origin - sy2)/PIXEL_SIZE + 0.5).astype(int)
    vt = (tc >= 1) & (tc <= NPIX) & (tr >= 1) & (tr <= NPIX)

    rc_min = max(0, tc[vt].min()-1); rc_max = min(NPIX-1, tc[vt].max()+1)
    rr_min = max(0, tr[vt].min()-1); rr_max = min(NPIX-1, tr[vt].max()+1)
    rn = rr_max-rr_min+1; rcn = rc_max-rc_min+1

    lc = tc - rc_min; lr = tr - rr_min
    vl = vt & (lc >= 0) & (lc < rcn) & (lr >= 0) & (lr < rn)

    print("  Building lookup with basin polygon test...")
    vi = np.where(vl)[0]
    sli = np.full(nr*nc, -1, dtype=np.int64)
    if len(vi) > 0:
        ib = points_in_polygons(sx2[vi], sy2[vi], all_poly_x_sin, all_poly_y_sin)
        sli[vi[ib]] = (lr[vi[ib]] * rcn + lc[vi[ib]])
    sli = sli.reshape(nr, nc)
    mo = sli >= 0
    print(f"  Mapped {mo.sum()} basin pixels")

    print(f"  Reading & reprojecting {len(dates)} frames...")
    data_utm = np.full((nr, nc, len(dates)), np.nan)
    for t in range(len(dates)):
        if t % 30 == 0 or t == len(dates)-1:
            print(f"    {t+1}/{len(dates)}: {dates[t]:%Y-%m-%d}")
        fr = read_frame_nc(nc_paths[t], fsca_var, rr_min, rc_min, rn, rcn, tidxs[t])
        if fill_val is not None: fr[fr == fill_val] = np.nan
        fr[(fr < valid_min)|(fr > valid_max)] = np.nan
        try:
            with nc4_Dataset(nc_paths[t]) as ds:
                if "grain_size" in ds.variables:
                    gs = read_frame_nc(nc_paths[t], "grain_size", rr_min, rc_min, rn, rcn, tidxs[t])
                    fr[(fr == 0) & (gs == 0)] = np.nan
        except Exception: pass
        if is_pct: fr /= 100.0
        out = np.full((nr, nc), np.nan)
        out[mo] = fr.ravel()[sli[mo]]
        data_utm[:,:,t] = out

    la_n = la2.max(); la_s = la2.min(); lo_w = lo2.min(); lo_e = lo2.max()

    pue = []; pun = []
    for px, py in zip(all_poly_x_sin, all_poly_y_sin):
        pl, pln = sin2latlon(px, py); pe, pn = latlon2utm(pl, pln, utm_zone)
        pue.append(pe); pun.append(pn)
    se, sn = latlon2utm(SNOTEL["lat"], SNOTEL["lon"], utm_zone)

    if args.format in ("mp4", "both"):
        print(f"\n[5a] Creating MP4...")
        create_movie(dates, data_utm, mo,
                     os.path.join(args.output_dir, f"BRB_fSCA_WY{water_year}.mp4"),
                     water_year, args.fps, ue_min, ue_max, un_min, un_max,
                     pue, pun, utm_zone, se, sn)

    if args.format in ("kmz", "both"):
        print(f"\n[5b] Creating KMZ...")
        create_kmz(dates, data_utm, mo,
                   os.path.join(args.output_dir, f"BRB_fSCA_WY{water_year}.kmz"),
                   water_year, la_n, la_s, lo_w, lo_e)

    print(f"\n{'='*66}\n  COMPLETE — outputs in: {args.output_dir}\n{'='*66}")


if __name__ == "__main__":
    main()
