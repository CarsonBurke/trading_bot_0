## Setup

The server is local-only by default. Use an authenticated tunnel to access it remotely.

## Usage

Start server

```bash
cargo run -p training_log_server -- --run latest --tail-lines 200
```

Query log

```bash
curl 127.0.0.1:8787/log
```

Public exposure must be explicit, for example `--bind 0.0.0.0:8787`, and should be
protected by host firewall and tunnel access controls.

Select a historical run without changing the active `latest` pointer with
`--run NAME`, or inspect an explicit run directory with `--run-root PATH`.
`--log-path PATH` remains available as an explicit mutually exclusive override.
