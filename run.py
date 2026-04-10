import pandas as pd
import numpy as np
import sys
from pathlib import Path
from metrics import calculate_dts_error_grid, calculate_rmse, calculate_mae


def validate_predictions_format(predictions_df, template_df):
    """
    Validate that predictions match the template format exactly.
    Returns (is_valid, error_message)
    """
    errors = []
    
    if len(predictions_df) != len(template_df):
        errors.append(f"Row count mismatch: Expected {len(template_df)} rows, got {len(predictions_df)} rows")
    
    required_cols = ['id', 'source_file', 'date', 'pred_30', 'pred_60', 'pred_90', 'pred_120']
    missing_cols = [col for col in required_cols if col not in predictions_df.columns]
    if missing_cols:
        errors.append(f"Missing required columns: {', '.join(missing_cols)}")
    
    if not errors and len(predictions_df) == len(template_df):
        id_match = (predictions_df['id'].values == template_df['id'].values).all()
        source_match = (predictions_df['source_file'].values == template_df['source_file'].values).all()
        date_match = (predictions_df['date'].values == template_df['date'].values).all()
        
        if not id_match:
            errors.append("Row IDs do not match the template")
        if not source_match:
            errors.append("Source files do not match the template")
        if not date_match:
            errors.append("Dates do not match the template")
        
        if id_match and source_match and date_match:
            for i in range(min(5, len(predictions_df))):
                pred_row = predictions_df.iloc[i]
                temp_row = template_df.iloc[i]
                if (pred_row['id'] != temp_row['id'] or 
                    pred_row['source_file'] != temp_row['source_file'] or 
                    pred_row['date'] != temp_row['date']):
                    errors.append(f"Row order mismatch at row {i+1}")
                    break
    
    pred_columns = ['pred_30', 'pred_60', 'pred_90', 'pred_120']
    for col in pred_columns:
        if col in predictions_df.columns:
            if predictions_df[col].isna().any():
                errors.append(f"Column '{col}' contains missing values (NaN)")
    
    if errors:
        return False, "\n".join(errors)
    return True, None


def calculate_metrics(predictions_df, targets_df, horizon='60'):
    """
    Calculate metrics for specified prediction horizon(s).
    
    Args:
        predictions_df: DataFrame with prediction columns
        targets_df: DataFrame with target columns
        horizon: '30', '60', '90', '120', or 'all'
    """
    results = {}
    
    if horizon == 'all':
        # Calculate individual metrics for each horizon
        horizons = [30, 60, 90, 120]
        all_preds = []
        all_targets = []
        
        for h in horizons:
            pred_col = f'pred_{h}'
            target_col = f'target_{h}'
            
            if pred_col not in predictions_df.columns or target_col not in targets_df.columns:
                continue
            
            pred_values = predictions_df[pred_col].values
            target_values = targets_df[target_col].values
            
            # Store for overall calculation
            all_preds.extend(pred_values)
            all_targets.extend(target_values)
            
            # Calculate individual horizon metrics
            rmse = calculate_rmse(pred_values, target_values)
            mae = calculate_mae(pred_values, target_values)
            dts_zones = calculate_dts_error_grid(pred_values, target_values)
            
            results[f'{h}_min'] = {
                'RMSE': rmse,
                'MAE': mae,
                'DTS_A': dts_zones['DTS_A_ZONE_PERCENT'],
                'DTS_B': dts_zones['DTS_B_ZONE_PERCENT'],
                'DTS_C': dts_zones['DTS_C_ZONE_PERCENT'],
                'DTS_D': dts_zones['DTS_D_ZONE_PERCENT'],
                'DTS_E': dts_zones['DTS_E_ZONE_PERCENT']
            }
        
        # Calculate overall metrics
        if all_preds and all_targets:
            overall_rmse = calculate_rmse(all_preds, all_targets)
            overall_mae = calculate_mae(all_preds, all_targets)
            overall_dts = calculate_dts_error_grid(all_preds, all_targets)
            
            results['overall'] = {
                'RMSE': overall_rmse,
                'MAE': overall_mae,
                'DTS_A': overall_dts['DTS_A_ZONE_PERCENT'],
                'DTS_B': overall_dts['DTS_B_ZONE_PERCENT'],
                'DTS_C': overall_dts['DTS_C_ZONE_PERCENT'],
                'DTS_D': overall_dts['DTS_D_ZONE_PERCENT'],
                'DTS_E': overall_dts['DTS_E_ZONE_PERCENT']
            }
    else:
        # Calculate metrics for single horizon
        pred_col = f'pred_{horizon}'
        target_col = f'target_{horizon}'
        
        if pred_col not in predictions_df.columns:
            raise ValueError(f"Prediction column '{pred_col}' not found")
        if target_col not in targets_df.columns:
            raise ValueError(f"Target column '{target_col}' not found")
        
        pred_values = predictions_df[pred_col].values
        target_values = targets_df[target_col].values
        
        rmse = calculate_rmse(pred_values, target_values)
        mae = calculate_mae(pred_values, target_values)
        dts_zones = calculate_dts_error_grid(pred_values, target_values)
        
        results[f'{horizon}_min'] = {
            'RMSE': rmse,
            'MAE': mae,
            'DTS_A': dts_zones['DTS_A_ZONE_PERCENT'],
            'DTS_B': dts_zones['DTS_B_ZONE_PERCENT'],
            'DTS_C': dts_zones['DTS_C_ZONE_PERCENT'],
            'DTS_D': dts_zones['DTS_D_ZONE_PERCENT'],
            'DTS_E': dts_zones['DTS_E_ZONE_PERCENT']
        }
    
    return results


def main():
    if len(sys.argv) < 2 or len(sys.argv) > 3:
        print("Usage: python run.py <predictions_file.parquet> [horizon]")
        print("\nExamples:")
        print("  python run.py my_predictions.parquet          # Default: 60-minute horizon")
        print("  python run.py my_predictions.parquet 30       # 30-minute horizon only")
        print("  python run.py my_predictions.parquet all      # All horizons + overall")
        print("\nValid horizons: 30, 60, 90, 120, all")
        sys.exit(1)
    
    predictions_file = Path(sys.argv[1])
    
    # Parse horizon parameter (default to 60)
    horizon = '60'
    if len(sys.argv) == 3:
        horizon = sys.argv[2]
        valid_horizons = ['30', '60', '90', '120', 'all']
        if horizon not in valid_horizons:
            print(f"❌ Error: Invalid horizon '{horizon}'")
            print(f"Valid horizons: {', '.join(valid_horizons)}")
            sys.exit(1)
    
    if not predictions_file.exists():
        print(f"❌ Error: File '{predictions_file}' not found")
        sys.exit(1)
    
    if not predictions_file.suffix == '.parquet':
        print(f"❌ Error: File must be in parquet format")
        sys.exit(1)
    
    template_file = Path("data/template.parquet")
    targets_file = Path("data/targets.parquet")
    
    if not template_file.exists():
        print(f"❌ Error: Template file '{template_file}' not found")
        sys.exit(1)
    
    if not targets_file.exists():
        print(f"❌ Error: Targets file '{targets_file}' not found")
        sys.exit(1)
    
    try:
        print("🔍 Loading files...")
        predictions_df = pd.read_parquet(predictions_file)
        template_df = pd.read_parquet(template_file)
        targets_df = pd.read_parquet(targets_file)
        
        print("✅ Files loaded successfully")
        
    except Exception as e:
        print(f"❌ Error loading files: {e}")
        sys.exit(1)
    
    print("\n📋 Validating format...")
    is_valid, error_msg = validate_predictions_format(predictions_df, template_df)
    
    if not is_valid:
        print(f"\n❌ FORMAT VALIDATION FAILED:\n")
        print(error_msg)
        print("\n⚠️ Critical Requirements:")
        print("• Exact Row Matching: Your submission must have the exact same rows (id, source_file, date) as the template")
        print("• Same Order: Rows must be in identical order to the template file")
        print("• Fill All Horizons: Provide predictions for pred_30, pred_60, pred_90, and pred_120 columns")
        print("• No Missing Values: All prediction columns must be filled (no NaN values)")
        print("• No Extra/Missing Rows: Do not add or remove any rows from the template")
        sys.exit(1)
    
    print("✅ Format validation passed!")
    
    print(f"\n📊 Calculating metrics for horizon: {horizon}...")
    metrics = calculate_metrics(predictions_df, targets_df, horizon)
    
    print("\n" + "="*60)
    print("                    EVALUATION RESULTS")
    print("="*60)
    
    # Display metrics based on what was calculated
    for horizon_key, horizon_metrics in metrics.items():
        if horizon_key == 'overall':
            print(f"\n📊 OVERALL METRICS (All Horizons Combined):")
        else:
            print(f"\n📈 {horizon_key.replace('_', ' ').title()} Ahead Predictions:")
        
        print(f"   RMSE: {horizon_metrics['RMSE']:.2f} mg/dL")
        print(f"   MAE:  {horizon_metrics['MAE']:.2f} mg/dL")
        print(f"\n   DTS Error Grid Zones:")
        print(f"   • Zone A (Clinically Accurate):     {horizon_metrics['DTS_A']:.1f}%")
        print(f"   • Zone B (Benign Errors):           {horizon_metrics['DTS_B']:.1f}%")
        print(f"   • Zone C (Overcorrection):          {horizon_metrics['DTS_C']:.1f}%")
        print(f"   • Zone D (Failure to Detect):       {horizon_metrics['DTS_D']:.1f}%")
        print(f"   • Zone E (Erroneous Treatment):     {horizon_metrics['DTS_E']:.1f}%")
    
    print("\n" + "="*60)
    print("\n✅ Format is valid! You are ready to submit!")
    print("🚀 Submit your predictions at:")
    print("   https://huggingface.co/spaces/MetabonetBench/leaderboard-space")
    print("\n" + "="*60)


if __name__ == "__main__":
    main()