import gzip
import pickle
from itertools import combinations

import pandas as pd
from utils import get_path
from constants import metrics, tumors_group
from models import models
from scipy.stats import wilcoxon
from constants import n_rep

path = get_path()

model_names = list(models.keys()) + ["ensemble"]

models_wilcoxon = []
models_stats = {}
for tumor_group in tumors_group:
    runs = []
    model_stats = {model_name: {} for model_name in model_names}
    for seed in range(n_rep):
        with gzip.open(
            path["files"] / "ARTICLE" / tumor_group / f"{seed}.gz.pkl", "rb"
        ) as file:
            run = pickle.load(file)
            runs.append(run)

    for model_1, model_2 in combinations(model_names, 2):
        for metric in metrics:
            metric_model_1 = [run[model_1]["metric"][metric] for run in runs]
            metric_model_2 = [run[model_2]["metric"][metric] for run in runs]
            wilcoxon_value = wilcoxon(metric_model_1, metric_model_2)
            models_wilcoxon.append(
                [
                    tumor_group,
                    model_1,
                    model_2,
                    metric,
                    wilcoxon_value.statistic,
                    wilcoxon_value.pvalue,
                ]
            )

    for model_name in model_names:
        for metric in metrics:
            metric_model = [run[model_name]["metric"][metric] for run in runs]
            model_stats[model_name][metric] = metric_model
            models_stats[tumor_group] = model_stats
    del runs

models_wilcoxon = pd.DataFrame(
    models_wilcoxon,
    columns=[
        "tumor_group",
        "model_1",
        "model_2",
        "metric",
        "wilcoxon_stats",
        "wilcoxon_pvalue",
    ],
)
