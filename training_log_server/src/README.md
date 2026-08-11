## Setup

The server is local-only by default. Use an authenticated tunnel to access it remotely.

## Usage

Start server

```bash
cargo run -p training_log_server -- --log-path training/training.log --tail-lines 200
```

Query log

```bash
curl 127.0.0.1:8787/log
```

Public exposure must be explicit, for example `--bind 0.0.0.0:8787`, and should be
protected by host firewall and tunnel access controls.
