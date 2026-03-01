# Dataset: LogPai HDFS

## What this dataset is

- Source: LogPai / LogHub HDFS logs (downloaded via Zenodo tarball, extracted to `data/raw/hdfs/HDFS.log`)
- Task: build a Week 6 pipeline that converts raw text logs into:
  - structured events (parsed fields)
  - normalized templates
  - 5-minute windows
  - window-level feature table

## Where the data lives (local only)

- Raw logs: `data/raw/hdfs/HDFS.log`
- Raw data is not committed to git (ignored in `.gitignore`)

## Assumed log line format (initial)

Expected fields per line:

- date (`YYMMDD`)
- time (`HHMMSS`)
- pid (`int`)
- level (`INFO` / `WARN` / `ERROR` / etc.)
- component (string before colon)
- content/message (rest of line)

## Example raw lines

Paste 2 to 5 example lines from `head -n 3 data/raw/hdfs/HDFS.log` here:

```text
081109 203518 143 INFO dfs.DataNode$DataXceiver: Receiving block blk_-1608999687919862906 src: /10.250.19.102:54106 dest: /10.250.19.102:50010
081109 203518 35 INFO dfs.FSNamesystem: BLOCK* NameSystem.allocateBlock: /mnt/hadoop/mapred/system/job_200811092030_0001/job.jar. blk_-1608999687919862906
081109 203519 143 INFO dfs.DataNode$DataXceiver: Receiving block blk_-1608999687919862906 src: /10.250.10.6:40524 dest: /10.250.10.6:50010
```

## Parsing plan

- Use a regex parser to extract: timestamp, pid, level, component, content
- If a line does not match, skip and count as malformed

## Templating plan

- Normalize variable tokens in content into placeholders (`IDs`, block IDs, numbers, IPs, hex)
- Output a template string + `template_id` (hash) for counting

## Windowing plan

- Window size: 300 seconds (5 minutes)
- Group key: component (or global if component missing)
