"""Plain-text metric logging into the run folder, readable without wandb.

Writes <run_dir>/metrics.csv in long format (`step,metric,value`), one row appended per metric per log call,
e.g. `500,eval/heldout/success/all,0.4`.
"""

from pathlib import Path


class LocalMetricsLogger:
    def __init__(self, run_dir: str | Path):
        self.path = Path(run_dir) / "metrics.csv"
        if not self.path.exists():
            self.path.write_text("step,metric,value\n")

    def log(self, metrics: dict, step: int, prefix: str = ""):
        rows = []
        for k, v in metrics.items():
            try:
                rows.append(f"{step},{prefix}{k},{float(v):.8g}\n")
            except (TypeError, ValueError):
                continue  # non-scalar (e.g. arrays, strings)
        with self.path.open("a") as f:
            f.writelines(rows)
