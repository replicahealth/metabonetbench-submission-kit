# MetaboNet Benchmark Submission Kit

This repository contains the evaluation script for the MetaboNet glucose prediction benchmark. 

[🏆 View Leaderboard 🏆](https://huggingface.co/spaces/MetabonetBench/leaderboard-space)

## Installation

1. Install the required dependencies:

```bash
uv venv .env
source .env/bin/activate  # On Windows: .env\Scripts\activate
uv pip install -r requirements.txt
```


## Quick Start

1. **Get the data**: Download raw training and test data from https://metabo-net.org/

2. **Train your model**: Use the training data to build your glucose prediction model

3. **Generate predictions**: Create predictions for the test set with these columns:
   - `pred_30`: 30-minute ahead glucose prediction
   - `pred_60`: 60-minute ahead glucose prediction  
   - `pred_90`: 90-minute ahead glucose prediction
   - `pred_120`: 120-minute ahead glucose prediction

4. **Format your predictions**: Your predictions file must:
   - Be in parquet format
   - Have the exact same rows and columns as `data/template.parquet` (same `id`, `source_file`, and `date` combinations)
   - Keep rows in the same order as the template
   - Include all prediction columns with no missing values

5. **Validate and evaluate**:
   ```bash
   python run.py your_predictions.parquet        # Default: 60-minute horizon
   python run.py your_predictions.parquet 30     # Evaluate 30-minute horizon only
   python run.py your_predictions.parquet all    # Evaluate all horizons + overall (all horizons combined)
   ```
   
   Available horizon options: `30`, `60`, `90`, `120`, or `all` (defaults to `60`)

6. **Submit**: Once validation passes, submit your predictions at:
https://huggingface.co/spaces/MetabonetBench/leaderboard-space


## Files

- `run.py` - Validation and evaluation script
- `metrics.py` - Metric calculation functions (RMSE, MAE, DTS Error Grid)
- `data/template.parquet` - Template showing required format for submissions
- `data/targets.parquet` - Ground truth values for evaluation


