import numpy as np
from pathlib import Path
from dataclasses import dataclass, field
from collections import defaultdict
from typing import List, Dict


@dataclass
class ScoreInstance:
    ref: Path | str
    hyp: Path | str

    def __post_init__(self):
        self.ref = Path(self.ref)
        self.hyp = Path(self.hyp)
        assert self.ref.exists(), "Reference does not exist"
        assert self.ref.suffix.endswith("tsv"), "Reference must be a tsv file"
        assert self.hyp.exists(), "Hypothesis does not exist"
        assert self.hyp.suffix.endswith("tsv"), "Hypothesis must be a tsv file"
        return


@dataclass
class ScoreInstances:
    instances: Dict[str, List[ScoreInstance]] = field(
        default_factory=lambda: defaultdict(list)
    )

    def __getitem__(self, index) -> List[ScoreInstance]:
        return self.instances[index]

    def values(self):
        return self.instances.values()

    def keys(self):
        return self.instances.keys()

    def items(self):
        return self.instances.items()

    def add(self, subject: str, instance: ScoreInstance):
        self.instances[subject].append(instance)
        return


@dataclass
class ResultsAgg:
    """
    Aggregates results from szcore_evaluation and produces an average
    for each metric. This is used to average results across multiple
    cross-validation folds to get an overall performance metric.
    """
    data: Dict[str, Dict[str, List[float]]] = field(
        default_factory=lambda: defaultdict(lambda: defaultdict(list))
    )

    def add(self, result: Dict[str, Dict[str, float]]):
        for result_type, metrics in result.items():
            for metric, value in metrics.items():
                self.data[result_type][metric].append(value)
        return

    def summarise(self, save_path: Path = None) -> Dict[str, Dict[str, float]]:
        avg_data = defaultdict(lambda: defaultdict())
        for result_type, metrics in self.data.items():
            result_type = f"avg_{result_type}"
            avg_data[result_type] = {
                metric: np.nanmean(values)
                for metric, values in metrics.items()
            }

        if save_path:
            import json
            with open(save_path, "w") as f:
                json.dump(avg_data, f, indent=2, sort_keys=False)

        return avg_data


def evaluate_instance(
    instances: ScoreInstances,
    outFile: Path = None,
    avg_per_subject: bool = True
) -> dict:
    """
    Compares two sets of seizure annotation instances.

    Parameters:
    instances (ScoreInstances): A collection of ScoreInstance organised by
        the subject. A ScoreInstance contains the Paths to reference (ref)
        and hypothesis (hyp) annotation files.
    outFile (Path): The path to the output JSON file where the results are
        saved.
    avg_per_subject (bool): Whether to compute average scores per subject or
        average across the full dataset.

    Returns:
    dict. return the evaluation result. The dictionary contains the following
          keys: {'sample_results': {'sensitivity', 'precision', 'f1', 'fpRate',
                    'sensitivity_std', 'precision_std', 'f1_std', 'fpRate_std'},
                 'event_results':{...}
                 }
    """
    import numpy as np
    import json
    from timescoring import scoring
    from szcore_evaluation.evaluate import Annotation, Annotations
    from szcore_evaluation.evaluate import Result

    FS = 1

    sample_results = dict()
    event_results = dict()
    for subject, sub_instances in instances.items():
        sample_results[subject] = Result()
        event_results[subject] = Result()

        for instance in sub_instances:
            # Load reference
            ref = Annotations.loadTsv(instance.ref)
            ref = Annotation(ref.getMask(FS), FS)

            # Load hypothesis
            hyp = Annotations.loadTsv(instance.hyp)
            hyp = Annotation(hyp.getMask(FS), FS)

            # BELOW is identical to evalute_dataset

            # Compute evaluation
            sample_score = scoring.SampleScoring(ref, hyp)
            event_score = scoring.EventScoring(ref, hyp)

            # Store results
            sample_results[subject] += Result(sample_score)
            event_results[subject] += Result(event_score)

        # Compute scores
        sample_results[subject].computeScores()
        event_results[subject].computeScores()

    aggregated_sample_results = dict()
    aggregated_event_results = dict()
    if avg_per_subject:
        for result_builder, aggregated_result in zip(
            (sample_results, event_results),
            (aggregated_sample_results, aggregated_event_results),
        ):
            for metric in ["sensitivity", "precision", "f1", "fpRate"]:
                aggregated_result[metric] = np.nanmean(
                    [getattr(x, metric) for x in result_builder.values()]
                )
                aggregated_result[f"{metric}_std"] = np.nanstd(
                    [getattr(x, metric) for x in result_builder.values()]
                )
    else:
        for result_builder, aggregated_result in zip(
            (sample_results, event_results),
            (aggregated_sample_results, aggregated_event_results),
        ):
            result_builder["cumulated"] = Result()
            for result in result_builder.values():
                result_builder["cumulated"] += result
            result_builder["cumulated"].computeScores()
            for metric in ["sensitivity", "precision", "f1", "fpRate"]:
                aggregated_result[metric] = getattr(result_builder["cumulated"], metric)

    output = {
        "sample_results": aggregated_sample_results,
        "event_results": aggregated_event_results,
    }

    if outFile:
        with open(outFile, "w") as file:
            json.dump(output, file, indent=2, sort_keys=False)

    return output