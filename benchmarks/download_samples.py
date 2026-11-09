"""
Downloads authentic benchmark subsets (15 pairs each) for local evaluation ONLY.
Do NOT commit downloaded dataset images (`benchmarks/datasets/`) to the Git repository.

Dataset Licenses & Redistribution Terms:
  1. UEyes (Jiang et al., CHI 2023 — Zenodo Record 8010312):
     - License: Creative Commons Attribution 4.0 International (CC BY 4.0).
     - Attribution required; fetched on-demand via HTTP Range over ZIP64 Central Directory.
  2. MIT1003 (Judd et al., ICCV 2009 — MIT CSAIL WherePeopleLook):
     - License: Academic / Non-Commercial Research Use Only.
     - Redistribution in commercial or MIT-licensed source repositories is NOT permitted.
     - Fetched on-demand directly from MIT CSAIL servers for local evaluation.
  3. SALICON (Jiang et al., CVPR 2015 — MS-COCO validation subset):
     - License: SALICON annotations under Academic Research terms; underlying images follow
       MS-COCO / Flickr Creative Commons terms. Fetched on-demand for local benchmarking.
"""

import argparse
import os
import ssl
import struct
import time
import zlib
import json
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Optional

CTX = ssl._create_unverified_context()
BASE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "datasets")


def fetch_bytes(url: str, start: Optional[int] = None, end: Optional[int] = None) -> bytes:
    headers = {"User-Agent": "OptiSight-Benchmark/2.3"}
    if start is not None and end is not None:
        headers["Range"] = f"bytes={start}-{end}"
    for attempt in range(4):
        try:
            req = urllib.request.Request(url, headers=headers)
            with urllib.request.urlopen(req, context=CTX, timeout=45) as resp:
                return resp.read()
        except urllib.error.HTTPError as e:
            if e.code in (429, 502, 503, 504) and attempt < 3:
                time.sleep(2.5 * (attempt + 1))
                continue
            raise
        except Exception:
            if attempt < 3:
                time.sleep(1.5 * (attempt + 1))
                continue
            raise
    return b""


def parse_zip_cd(cd_bytes: bytes):
    """Parses standard & ZIP64 Central Directory entries into a dict {filename: (offset, comp_size, uncomp_size, method)}."""
    entries = {}
    pos = 0
    n = len(cd_bytes)
    while pos + 46 <= n:
        if cd_bytes[pos : pos + 4] != b"PK\x01\x02":
            nxt = cd_bytes.find(b"PK\x01\x02", pos + 1)
            if nxt == -1:
                break
            pos = nxt
            continue
        method = struct.unpack("<H", cd_bytes[pos + 10 : pos + 12])[0]
        comp_size, uncomp_size = struct.unpack("<II", cd_bytes[pos + 20 : pos + 28])
        nlen, elen, clen = struct.unpack("<HHH", cd_bytes[pos + 28 : pos + 34])
        local_off = struct.unpack("<I", cd_bytes[pos + 42 : pos + 46])[0]
        name = cd_bytes[pos + 46 : pos + 46 + nlen].decode("utf-8", "ignore")
        extra = cd_bytes[pos + 46 + nlen : pos + 46 + nlen + elen]

        if local_off == 0xFFFFFFFF or comp_size == 0xFFFFFFFF or uncomp_size == 0xFFFFFFFF:
            ep = 0
            while ep + 4 <= len(extra):
                hid, dsize = struct.unpack("<HH", extra[ep : ep + 4])
                if hid == 0x0001:
                    edata = extra[ep + 4 : ep + 4 + dsize]
                    idx = 0
                    if uncomp_size == 0xFFFFFFFF and idx + 8 <= len(edata):
                        uncomp_size = struct.unpack("<Q", edata[idx : idx + 8])[0]
                        idx += 8
                    if comp_size == 0xFFFFFFFF and idx + 8 <= len(edata):
                        comp_size = struct.unpack("<Q", edata[idx : idx + 8])[0]
                        idx += 8
                    if local_off == 0xFFFFFFFF and idx + 8 <= len(edata):
                        local_off = struct.unpack("<Q", edata[idx : idx + 8])[0]
                        idx += 8
                    break
                ep += 4 + dsize

        entries[name] = (local_off, comp_size, uncomp_size, method)
        pos += 46 + nlen + elen + clen
    return entries


def extract_remote_zip_entry(url: str, entry_tuple) -> bytes:
    local_off, comp_size, _, method = entry_tuple
    raw = fetch_bytes(url, local_off, local_off + 256 + comp_size)
    nlen, elen = struct.unpack("<HH", raw[26:30])
    data_start = 30 + nlen + elen
    comp_data = raw[data_start : data_start + comp_size]
    if method == 0:
        return comp_data
    elif method == 8:
        return zlib.decompress(comp_data, -zlib.MAX_WBITS)
    raise ValueError(f"Unsupported compression method: {method}")


def download_mit1003(count: int = 15):
    print("=== Downloading MIT1003 real samples ===")
    out_img = os.path.join(BASE_DIR, "mit1003", "images")
    out_den = os.path.join(BASE_DIR, "mit1003", "density")
    out_fix = os.path.join(BASE_DIR, "mit1003", "fixations")
    os.makedirs(out_img, exist_ok=True)
    os.makedirs(out_den, exist_ok=True)
    os.makedirs(out_fix, exist_ok=True)

    url_stim = "http://people.csail.mit.edu/tjudd/WherePeopleLook/ALLSTIMULI.zip"
    url_fix = "http://people.csail.mit.edu/tjudd/WherePeopleLook/ALLFIXATIONMAPS.zip"

    # Read ALLFIXATIONMAPS.zip CD (size 9665209)
    fix_tail = fetch_bytes(url_fix, 9665209 - 210000, 9665209 - 1)
    eocd = fix_tail.rfind(b"PK\x05\x06")
    cd_size, cd_off = struct.unpack("<II", fix_tail[eocd + 12 : eocd + 20])
    fix_cd = fetch_bytes(url_fix, cd_off, cd_off + cd_size - 1)
    fix_entries = parse_zip_cd(fix_cd)

    # Read ALLSTIMULI.zip CD (size 235000497)
    stim_tail = fetch_bytes(url_stim, 235000497 - 120000, 235000497 - 1)
    eocd_s = stim_tail.rfind(b"PK\x05\x06")
    cds_size, cds_off = struct.unpack("<II", stim_tail[eocd_s + 12 : eocd_s + 20])
    stim_cd = fetch_bytes(url_stim, cds_off, cds_off + cds_size - 1)
    stim_entries = parse_zip_cd(stim_cd)

    stim_keys = sorted(
        [
            k
            for k in stim_entries.keys()
            if k.startswith("ALLSTIMULI/i") and k.endswith(".jpeg") and "/." not in k
        ]
    )

    tasks = []
    for sk in stim_keys:
        if len(tasks) >= count:
            break
        stem = os.path.splitext(os.path.basename(sk))[0]
        fmap_key = f"ALLFIXATIONMAPS/{stem}_fixMap.jpg"
        fpts_key = f"ALLFIXATIONMAPS/{stem}_fixPts.jpg"
        if fmap_key in fix_entries and fpts_key in fix_entries:
            tasks.append((stem, sk, fmap_key, fpts_key))

    def _fetch_one_mit(item):
        stem, sk, fmap_key, fpts_key = item
        p_img = os.path.join(out_img, f"{stem}.jpg")
        p_den = os.path.join(out_den, f"{stem}.jpg")
        p_fix = os.path.join(out_fix, f"{stem}.jpg")
        if os.path.exists(p_img) and os.path.exists(p_den) and os.path.exists(p_fix):
            return stem
        img_bytes = extract_remote_zip_entry(url_stim, stim_entries[sk])
        den_bytes = extract_remote_zip_entry(url_fix, fix_entries[fmap_key])
        pts_bytes = extract_remote_zip_entry(url_fix, fix_entries[fpts_key])
        with open(p_img, "wb") as f:
            f.write(img_bytes)
        with open(p_den, "wb") as f:
            f.write(den_bytes)
        with open(p_fix, "wb") as f:
            f.write(pts_bytes)
        return stem

    saved = 0
    with ThreadPoolExecutor(max_workers=8) as pool:
        futs = [pool.submit(_fetch_one_mit, t) for t in tasks]
        for fut in as_completed(futs):
            try:
                stem = fut.result()
                saved += 1
                if saved % 20 == 0 or saved == len(tasks):
                    print(f"  [MIT1003 {saved}/{len(tasks)}] {stem}")
            except Exception as e:
                print("  MIT1003 skip:", e)


def download_salicon(count: int = 120):
    print("=== Downloading SALICON real samples ===")
    out_img = os.path.join(BASE_DIR, "salicon", "images")
    out_den = os.path.join(BASE_DIR, "salicon", "density")
    os.makedirs(out_img, exist_ok=True)
    os.makedirs(out_den, exist_ok=True)

    api_url = f"https://huggingface.co/api/datasets/TPagent/salicon/tree/main/val/images?limit={max(200, count + 20)}"
    items = json.loads(fetch_bytes(api_url).decode("utf-8"))
    stems = []
    for item in items:
        if len(stems) >= count:
            break
        path = item.get("path", "")
        if path.endswith(".jpg"):
            stems.append(os.path.splitext(os.path.basename(path))[0])

    def _fetch_one_sal(stem: str) -> str:
        p_img = os.path.join(out_img, f"{stem}.jpg")
        p_den = os.path.join(out_den, f"{stem}.png")
        if os.path.exists(p_img) and os.path.exists(p_den):
            return stem
        img_url = f"https://huggingface.co/datasets/TPagent/salicon/resolve/main/val/images/{stem}.jpg"
        sal_url = f"https://huggingface.co/datasets/TPagent/salicon/resolve/main/val/saliency/{stem}.png"
        img_b = fetch_bytes(img_url)
        sal_b = fetch_bytes(sal_url)
        with open(p_img, "wb") as f:
            f.write(img_b)
        with open(p_den, "wb") as f:
            f.write(sal_b)
        return stem

    saved = 0
    with ThreadPoolExecutor(max_workers=8) as pool:
        futs = [pool.submit(_fetch_one_sal, s) for s in stems]
        for fut in as_completed(futs):
            try:
                stem = fut.result()
                saved += 1
                if saved % 20 == 0 or saved == len(stems):
                    print(f"  [SALICON {saved}/{len(stems)}] {stem}")
            except Exception as e:
                print("  SALICON skip:", e)


def download_ueyes(count: int = 120):
    print("=== Downloading UEyes real UI/Poster/Webpage samples ===")
    out_img = os.path.join(BASE_DIR, "ueyes", "images")
    out_den = os.path.join(BASE_DIR, "ueyes", "density")
    out_fix = os.path.join(BASE_DIR, "ueyes", "fixations")
    os.makedirs(out_img, exist_ok=True)
    os.makedirs(out_den, exist_ok=True)
    os.makedirs(out_fix, exist_ok=True)

    url = "https://zenodo.org/records/8010312/files/UEyes_dataset.zip?download=1"
    cd_off = 12889572595
    # First 7.5 MB of CD covers all of images/ and saliency_maps/
    cd_chunk = fetch_bytes(url, cd_off, cd_off + 7600000)
    entries = parse_zip_cd(cd_chunk)

    img_keys = [
        k
        for k in entries.keys()
        if k.startswith("UEyes_dataset/images/")
        and (k.endswith(".png") or k.endswith(".jpg"))
        and "/._" not in k
    ]
    img_keys.sort()

    tasks = []
    for ik in img_keys:
        if len(tasks) >= count:
            break
        fname = os.path.basename(ik)
        stem = os.path.splitext(fname)[0]
        h_key = None
        f_key = None
        for ext in (".jpg", ".png"):
            cand_h = f"UEyes_dataset/saliency_maps/heatmaps_3s/{stem}{ext}"
            cand_f = f"UEyes_dataset/saliency_maps/fixmaps_3s/{stem}{ext}"
            if cand_h in entries:
                h_key = cand_h
            if cand_f in entries:
                f_key = cand_f
        if h_key and f_key:
            tasks.append((stem, ik, h_key, f_key))

    def _fetch_one_ueyes(item):
        stem, ik, h_key, f_key = item
        img_ext = os.path.splitext(ik)[1]
        den_ext = os.path.splitext(h_key)[1]
        fix_ext = os.path.splitext(f_key)[1]
        p_img = os.path.join(out_img, f"{stem}{img_ext}")
        p_den = os.path.join(out_den, f"{stem}{den_ext}")
        p_fix = os.path.join(out_fix, f"{stem}{fix_ext}")
        if os.path.exists(p_img) and os.path.exists(p_den) and os.path.exists(p_fix):
            return stem
        img_b = extract_remote_zip_entry(url, entries[ik])
        den_b = extract_remote_zip_entry(url, entries[h_key])
        fix_b = extract_remote_zip_entry(url, entries[f_key])
        with open(p_img, "wb") as f:
            f.write(img_b)
        with open(p_den, "wb") as f:
            f.write(den_b)
        with open(p_fix, "wb") as f:
            f.write(fix_b)
        return stem

    saved = 0
    with ThreadPoolExecutor(max_workers=2) as pool:
        futs = [pool.submit(_fetch_one_ueyes, t) for t in tasks]
        for fut in as_completed(futs):
            try:
                stem = fut.result()
                saved += 1
                if saved % 20 == 0 or saved == len(tasks):
                    print(f"  [UEyes {saved}/{len(tasks)}] {stem}")
            except Exception as e:
                print("  UEyes skip:", e)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Download local benchmark samples (UEyes, MIT1003, SALICON) into git-ignored benchmarks/datasets/."
    )
    parser.add_argument(
        "--count",
        type=int,
        default=15,
        help="Number of stimulus/ground-truth pairs per dataset (default: 15).",
    )
    parser.add_argument(
        "--dataset",
        choices=["all", "mit1003", "salicon", "ueyes"],
        default="all",
        help="Which dataset subset to download (default: all).",
    )
    args = parser.parse_args()

    if args.dataset in ("all", "mit1003"):
        download_mit1003(args.count)
    if args.dataset in ("all", "salicon"):
        download_salicon(args.count)
    if args.dataset in ("all", "ueyes"):
        download_ueyes(args.count)
    print("Benchmark dataset sample download completed! (Stored in git-ignored benchmarks/datasets/)")
