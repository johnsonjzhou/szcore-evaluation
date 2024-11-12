# szcore-evaluation

Compare szCORE compliant annotations of EEG datasets of people with epilelpsy.

The package compares annotations in TSV. The annotations should be organized in a BIDS compliant manner:

```txt
BIDS_DATASET/
├── ...
├── sub-01/
│   ├── ses-01/
│   │   └── eeg/
│   │       ├── sub-01_ses-01_task-szMonitoring_run-00_events.tsv
│   │       ├── ...
│   ├── ...
├── ...
```

The package compares hypothesis annotations to reference annotations from two folders which follow the same structure. It provides a JSON file with the overall results as an output:

```json
{
  "sample_results": {
    "sensitivity": 0.08,
    "sensitivity_std": 0.04,
    "precision": 0.01,
    "precision_std": 0.01,
    "f1": 0.02,
    "f1_std": 0.01,
    "fpRate": 9792.41,
    "fpRate_std": 4566.68
  },
  "event_results": {
    "sensitivity": 1.0,
    "sensitivity_std": 0.0,
    "precision": 0.08,
    "precision_std": 0.03,
    "f1": 0.16,
    "f1_std": 0.04,
    "fpRate": 280.55,
    "fpRate_std": 0.03
  }
}
```

The library provides a simple interface:

```python
def evaluate_dataset(
    reference: Path, hypothesis: Path, outFile: Path, avg_per_subject=True
) -> dict:
    """
    Compares two sets of seizure annotations accross a full dataset.

    Parameters:
    reference (Path): The path to the folder containing the reference TSV files.
    hypothesis (Path): The path to the folder containing the hypothesis TSV files.
    outFile (Path): The path to the output JSON file where the results are saved.
    avg_per_subject (bool): Whether to compute average scores per subject or
                            average across the full dataset.

    Returns:
    dict. return the evaluation result. The dictionary contains the following
          keys: {'sample_results': {'sensitivity', 'precision', 'f1', 'fpRate',
                    'sensitivity_std', 'precision_std', 'f1_std', 'fpRate_std'},
                 'event_results':{...}
                 }
    """
```
