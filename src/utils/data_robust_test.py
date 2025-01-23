import torch
import pandas as pd
import numpy as np
from pathlib import Path
from evidently.report import Report
from evidently.metric_preset import DataDriftPreset
from model import QuickDrawModel
from data import load_dataset

def extract_basic_features(dataset: torch.utils.data.Dataset):
    """Extract simple statistical features from sketch images."""
    features_list = []
    
    for img, label in dataset:
        # Convert to numpy and ensure 2D
        img_np = img.squeeze().numpy()
        
        # Basic statistics
        feature_dict = {
            'mean_intensity': float(np.mean(img_np)),
            'num_pixels': float(np.count_nonzero(img_np)),
            'label': float(label)
        }
        
        features_list.append(feature_dict)
    
    return pd.DataFrame(features_list)

def analyze_drift(reference_data: pd.DataFrame, current_data: pd.DataFrame):
    """Generate drift report using Evidently."""
    report = Report(metrics=[DataDriftPreset()])
    report.run(reference_data=reference_data, current_data=current_data)
    
    # Save report
    report_path = "reports/figures/data_drift_report.html"
    Path(report_path).parent.mkdir(parents=True, exist_ok=True)
    report.save_html(report_path)
    
    print(f"Drift report saved to {report_path}")

def main():
    try:
        print("Loading datasets...")
        train_dataset = load_dataset('train')
        test_dataset = load_dataset('test')
        
        print("Extracting features...")
        train_features = extract_basic_features(train_dataset)
        test_features = extract_basic_features(test_dataset)
        
        print("Analyzing drift...")
        analyze_drift(train_features, test_features)
        
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    main()