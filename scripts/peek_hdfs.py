from pathlib import Path
from log_anomaly.parsing import parse_hdfs_line
from log_anomaly.templating import to_template

LOG_PATH = Path("data/raw/hdfs/HDFS.log")

def main():
    ok = 0
    bad = 0

    with LOG_PATH.open("r", encoding="utf-8", errors="replace") as f:
        for i, line in enumerate(f, start=1):
            evt = parse_hdfs_line(line)
            if evt is None:
                bad += 1
            else:
                ok += 1
                if ok <= 5:
                    tmpl = to_template(evt.content)
                    print(f"[{ok}] ts={evt.ts} level={evt.level} component={evt.component}")
                    print(f"     content: {evt.content}")
                    print(f"     tmpl   : {tmpl}\n")

            if i >= 2000:
                break

    total = ok + bad
    print(f"Parsed {ok}/{total} lines successfully ({ok/total:.1%}).")

if __name__ == "__main__":
    main()
