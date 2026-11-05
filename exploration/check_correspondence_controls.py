"""Check whether a learned metric's benefit is explained by normalization alone."""
import argparse
import json
from pathlib import Path

import torch
from torch import nn
from torch.nn import functional as F

from exploration.oracle_transport import digest
from exploration.train_correspondence import evaluate


class NormalizedCosine(nn.Module):
    def forward(self, reference, current):
        # Same placement as the metric's first LayerNorm: after query sampling.
        reference = F.layer_norm(reference, (reference.shape[-1],))
        current = F.layer_norm(current, (current.shape[-1],))
        return F.normalize(reference, dim=-1) @ F.normalize(current, dim=-1).T


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", required=True)
    args = parser.parse_args()
    torch.set_num_threads(2)
    run = Path(args.run).resolve()
    plan = json.loads((run/"plan.json").read_text())
    final = json.loads((run/"final.json").read_text())
    if final["plan_sha256"] != digest(run/"plan.json"):
        raise ValueError("Metric protocol changed")
    for name, value in plan["sources"].items():
        if digest(name) != value:
            raise ValueError("Metric source changed")
    directory, config = Path(plan["cache"]), plan["config"]
    cache_summary = json.loads((directory/"summary.json").read_text())
    if digest(directory/"summary.json") != final["cache_summary_sha256"]:
        raise ValueError("Cache manifest changed")
    cache = {}
    for scene in config["val"]:
        path = directory/(scene+".pt")
        if digest(path) != cache_summary["scenes"][scene]["sha256"]:
            raise ValueError("Validation cache changed")
        cache[scene] = torch.load(path, map_location="cpu", weights_only=True)
    report = evaluate({str(d):NormalizedCosine() for d in config["depths"]}, cache,
                      config["val"], config["seed"], config["evaluation_queries"])
    for category in ("rows", "summary"):
        for row in report[category]:
            if row["method"] == "learned_metric":
                row["method"] = "parameter_free_layernorm_cosine"
            else:
                match = next(r for r in final[category] if all(r.get(k) == row.get(k)
                    for k in ("scene", "depth", "method", "stratum")))
                if row != match:
                    raise ValueError("Frozen baseline no longer reproduces exactly")
    report.update(metric_final_sha256=digest(run/"final.json"), source_sha256=digest(__file__),
        scope="Post-result control for input-normalization confounding. No parameters fitted or thresholds selected; all depths and scenes retained.")
    (run/"normalization_control.json").write_text(json.dumps(report, indent=2)+"\n")
    for row in report["summary"]:
        if row["stratum"] == "all_visible":
            print(json.dumps(row), flush=True)


if __name__ == "__main__":
    main()
