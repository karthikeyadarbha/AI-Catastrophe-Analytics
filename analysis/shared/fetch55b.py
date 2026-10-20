import requests, time, pandas as pd, io, calendar
from concurrent.futures import ThreadPoolExecutor

wins = []
for y in range(1990, 2027):
    for m in range(1, 13, 2):
        if y == 2026 and m > 10:
            continue
        m2 = m + 1
        wins.append((f"{y}-{m:02d}-01", f"{y}-{m2:02d}-{calendar.monthrange(y, m2)[1]}T23:59:59"))


def get(w):
    for t in range(10):
        try:
            r = requests.get(
                "https://earthquake.usgs.gov/fdsnws/event/1/query",
                params=dict(format="csv", starttime=w[0], endtime=w[1], minmagnitude=5.5,
                            maxmagnitude=6.5, eventtype="earthquake", orderby="time-asc"),
                timeout=90)
            if r.status_code == 200:
                return pd.read_csv(io.StringIO(r.text))
        except Exception:
            pass
        time.sleep(3 + t)
    print("FAILED", w, flush=True)
    return None


out = []
n = 0
with ThreadPoolExecutor(4) as ex:
    for df in ex.map(get, wins):
        n += 1
        out.append(df)
        if n % 10 == 0:
            print(n, len(wins), flush=True)
good = [d for d in out if d is not None]
pd.concat(good).to_csv("m55_65.csv", index=False)
print("done", sum(len(d) for d in good), "failed", len(out) - len(good))
