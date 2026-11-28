"""python3 blueprint/bias/build_cache.py dev   -> blueprint/cache/{days,races,hours}_dev.parquet"""
import sys, time
from pathlib import Path
HERE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(HERE))
from engine.data import load_split
from engine.levels import build_levels
from bias.dataset import build

split = sys.argv[1] if len(sys.argv) > 1 else "dev"
t = time.time()
d, L = build_levels(load_split(split))
days, races, hours = build(d, L)
for name, df in (("days", days), ("races", races), ("hours", hours)):
    df = df[df.split == split]
    df.to_parquet(HERE / "cache" / f"{name}_{split}.parquet", index=False)
    print(name, len(df))
print(f"{time.time() - t:.0f}s")
