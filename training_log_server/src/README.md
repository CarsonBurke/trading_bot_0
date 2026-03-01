## Setup

1. Port forward 8787 tcp/udp
2. Profit

## Usage

Start server

```bash
cargo run -p training_log_server -- --bind 0.0.0.0:8787 --log-path training/training.log --tail-lines 200
```

Query log

```bash
curl 24.87.159.9:8787/log
```