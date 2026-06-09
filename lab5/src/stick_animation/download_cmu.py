from __future__ import annotations

import argparse
import os
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Iterable, List, Tuple

BASE = "https://raw.githubusercontent.com/una-dinosauria/cmu-mocap/master/data"

WALK_TRIALS: List[Tuple[str, str]] = [
    ("002", "02_02"),
    ("007", "07_01"), ("007", "07_02"), ("007", "07_03"),
    ("007", "07_06"), ("007", "07_07"), ("007", "07_08"),
    ("007", "07_09"), ("007", "07_10"), ("007", "07_11"),
    ("007", "07_12"),
    ("008", "08_01"), ("008", "08_02"), ("008", "08_03"),
    ("008", "08_06"), ("008", "08_08"), ("008", "08_09"),
    ("008", "08_10"), ("008", "08_11"),
    ("035", "35_01"), ("035", "35_02"), ("035", "35_03"),
    ("035", "35_04"), ("035", "35_05"), ("035", "35_06"),
    ("035", "35_07"), ("035", "35_08"), ("035", "35_09"),
    ("035", "35_10"), ("035", "35_11"), ("035", "35_12"),
    ("035", "35_13"), ("035", "35_14"), ("035", "35_15"),
    ("035", "35_16"), ("035", "35_17"), ("035", "35_18"),
    ("035", "35_19"), ("035", "35_20"), ("035", "35_21"),
    ("035", "35_22"), ("035", "35_23"), ("035", "35_24"),
    ("035", "35_25"), ("035", "35_26"), ("035", "35_27"),
    ("035", "35_28"), ("035", "35_29"), ("035", "35_30"),
    ("035", "35_31"), ("035", "35_32"), ("035", "35_33"),
    ("035", "35_34"),
    ("036", "36_01"), ("036", "36_02"), ("036", "36_03"),
]

JUMP_TRIALS: List[Tuple[str, str]] = [
    ("013", "13_11"), ("013", "13_13"), ("013", "13_19"),
    ("013", "13_27"), ("013", "13_28"), ("013", "13_29"),
    ("013", "13_30"), ("013", "13_32"), ("013", "13_39"),
    ("013", "13_40"), ("013", "13_41"), ("013", "13_42"),
    ("016", "16_01"), ("016", "16_02"), ("016", "16_03"),
    ("016", "16_05"), ("016", "16_06"), ("016", "16_09"),
    ("016", "16_10"),
    ("091", "91_45"),
    ("105", "105_45"),
    ("118", "118_01"), ("118", "118_02"), ("118", "118_03"),
    ("118", "118_04"), ("118", "118_05"), ("118", "118_06"),
    ("118", "118_07"),
]


def _download(url: str, dest: str) -> Tuple[str, bool, str]:
    if os.path.exists(dest) and os.path.getsize(dest) > 0:
        return dest, True, "cached"
    try:
        with urllib.request.urlopen(url, timeout=30) as r, open(dest, "wb") as f:
            f.write(r.read())
        return dest, True, "ok"
    except Exception as e:
        if os.path.exists(dest):
            os.remove(dest)
        return dest, False, str(e)


def _grab(trials: Iterable[Tuple[str, str]], out_dir: str, workers: int) -> Tuple[int, int]:
    os.makedirs(out_dir, exist_ok=True)
    futures = {}
    with ThreadPoolExecutor(max_workers=workers) as pool:
        for subj, trial in trials:
            url = f"{BASE}/{subj}/{trial}.bvh"
            dest = os.path.join(out_dir, f"{trial}.bvh")
            futures[pool.submit(_download, url, dest)] = (subj, trial)
        ok, fail = 0, 0
        for fut in as_completed(futures):
            subj, trial = futures[fut]
            _, success, msg = fut.result()
            tag = "✓" if success else "✗"
            print(f"  {tag} {subj}/{trial}.bvh ({msg})")
            ok += int(success)
            fail += int(not success)
    return ok, fail


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", default="data/raw",
                    help="Destination root. Writes walk/ and jump/ subfolders.")
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--skip-walk", action="store_true")
    ap.add_argument("--skip-jump", action="store_true")
    args = ap.parse_args()

    if not args.skip_walk:
        print(f"\nDownloading {len(WALK_TRIALS)} walk trials → {args.out_dir}/walk")
        ok, fail = _grab(WALK_TRIALS, os.path.join(args.out_dir, "walk"), args.workers)
        print(f"  walk: ok={ok} fail={fail}")

    if not args.skip_jump:
        print(f"\nDownloading {len(JUMP_TRIALS)} jump trials → {args.out_dir}/jump")
        ok, fail = _grab(JUMP_TRIALS, os.path.join(args.out_dir, "jump"), args.workers)
        print(f"  jump: ok={ok} fail={fail}")


if __name__ == "__main__":
    main()
