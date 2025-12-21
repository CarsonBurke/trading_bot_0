# report_cli

CLI for reading serialized training reports and printing line-oriented output for debugging.

Usage:

```
cargo run -p report_cli -- <generation> <report_name> [ticker]
```

Examples:

```
cargo run -p report_cli -- 10 final_assets
cargo run -p report_cli -- 23 assets NVDA
cargo run -p report_cli -- 23 buy_sell NVDA
```

Notes:
- `report_name` is normalized to lowercase with underscores.
- Reports are read from `../training/gens/<generation>/`.
