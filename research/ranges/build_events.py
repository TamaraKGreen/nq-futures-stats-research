"""Build range events for the DEV split: clock-aligned ranges and a 7-minute-offset null control."""
import sys, time
from multiprocessing import Pool
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import pandas as pd
from research.scripts.load import load
from research.ranges.engine import run

OUT = Path(__file__).resolve().parents[1] / "output" / "range_events_dev.parquet"

def job(args):
    d, o = args
    b = load("dev")
    t = time.time(); r = run(b, durs=(d,), offsets=(o,), horizon_mult=4.0)
    print(f"dur {d} offset {o}: {len(r):,} ranges in {time.time()-t:.0f}s", flush=True)
    return r

if __name__ == "__main__":
    with Pool(4) as p:
        parts = p.map(job, [(d, o) for d in (15, 30, 60) for o in (0, 7)])
    ev = pd.concat(parts, ignore_index=True)
    ev.to_parquet(OUT, index=False)
    print("saved", OUT, len(ev))
