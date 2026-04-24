from typing import List, Tuple, Dict, Any
import os
import sys
import pandas as pd
import numpy as np

def load_data(
    data_dir: str,
    id_list: List[int],
    date: str,
    pre_loader_func_name: str,
    module_name: str
) -> pd.DataFrame:
    """Load raw data using a dynamically imported pre-loader function."""
    try:
        module = sys.modules[module_name]
        pre_loader = getattr(module, pre_loader_func_name)
        return pre_loader(data_dir, id_list, date)
    except AttributeError:
        raise ValueError(f"Pre-loader '{pre_loader_func_name}' not found in '{module_name}'.")

def get_splits(
    data_dir: str,
    fold_config: Dict[str, List[int]],
    date: str,
    pre_loader_func_name: str,
    module_name: str,
    drop_timestamp: bool = True
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Loads train, validation, and test splits based on cow IDs in fold_config."""
    
    train_ids = [int(x) for x in fold_config.get('train', [])]
    val_ids = [int(x) for x in fold_config.get('val', [])]
    test_ids = [int(x) for x in fold_config.get('test', [])]
    
    train_data = load_data(data_dir, train_ids, date, pre_loader_func_name, module_name)
    
    if val_ids:
        val_data = load_data(data_dir, val_ids, date, pre_loader_func_name, module_name)
    else:
        val_data = pd.DataFrame()
        
    test_data = load_data(data_dir, test_ids, date, pre_loader_func_name, module_name)
    
    if drop_timestamp:
        for df in [train_data, val_data, test_data]:
            if not df.empty:
               df.drop(columns=['timestamp'], errors='ignore', inplace=True)
                
    return train_data, val_data, test_data

def immu_pre_data_loader(sensor_data_dir: str, id_list: List[int], date: str) -> pd.DataFrame:
    """Load raw IMU data for a list of cow IDs."""
    combined_data_df = pd.DataFrame()
    
    for cow_id in id_list:
        cow_name = f'C{cow_id:02d}'
        tag_name = f'T{cow_id:02d}'
        
        accel_file = os.path.join(sensor_data_dir, 'main_data', 'immu', tag_name, f'{tag_name}_{date}.csv')
        head_file = os.path.join(sensor_data_dir, 'sub_data', 'head_direction', tag_name, f'T{cow_id:02d}_{date}.csv')
        label_file = os.path.join(sensor_data_dir, 'behavior_labels', 'individual', f'C{cow_id:02d}_{date}.csv')
        
        if not (os.path.exists(accel_file) and os.path.exists(head_file) and os.path.exists(label_file)):
             print(f"Warning: Missing data files for cow {cow_id}. Skipping.")
             continue

        accel_data = pd.read_csv(accel_file)
        if 'timestamp' not in accel_data.columns or 'accel_x_mps2' not in accel_data.columns:
            continue
        accel_data = accel_data[['timestamp', 'accel_x_mps2', 'accel_y_mps2', 'accel_z_mps2']]
        
        head_data = pd.read_csv(head_file)
        accel_data = pd.merge(accel_data, head_data[['timestamp', 'relative_angle']], 
                             on='timestamp', how='inner')
        
        label_df = pd.read_csv(label_file)
        label_df = label_df[['timestamp', 'behavior']]
        label_df['timestamp'] = label_df['timestamp'].astype(np.float64)
        
        merged_df = pd.merge_asof(accel_data.sort_values('timestamp'), 
                                  label_df.sort_values('timestamp'), 
                                  on='timestamp', direction='nearest')
        merged_df = merged_df.ffill().bfill()
        
        combined_data_df = pd.concat([combined_data_df, merged_df], ignore_index=True)
    
    return combined_data_df

def uwb_pre_data_loader(sensor_data_dir: str, id_list: List[int], date: str) -> pd.DataFrame:
    """Load raw UWB (Ultra-Wideband) positioning data for a list of cow IDs."""
    combined_data_df = pd.DataFrame()
    
    for cow_id in id_list:
        cow_name = f'C{cow_id:02d}'
        tag_name = f'T{cow_id:02d}'
        
        uwb_file = os.path.join(sensor_data_dir, 'main_data', 'uwb', tag_name, f'{tag_name}_{date}.csv')
        label_file = os.path.join(sensor_data_dir, 'behavior_labels', 'individual', f'{cow_name}_{date}.csv')


        
        if not (os.path.exists(uwb_file) and os.path.exists(label_file)):
            print(f"Warning: Missing data files for cow {cow_id}. Skipping.")
            continue
        
        uwb_data = pd.read_csv(uwb_file)
        if 'timestamp' not in uwb_data.columns:
            continue
        
        position_candidates = [
            ['coord_x_cm', 'coord_y_cm', 'coord_z_cm'],
            ['coord_x', 'coord_y', 'coord_z'],
            ['x', 'y', 'z'],
            ['x_m', 'y_m', 'z_m'],
        ]
        
        pos_cols = None
        for cols in position_candidates:
            if all(col in uwb_data.columns for col in cols):
                pos_cols = cols
                break
 
        
        if pos_cols is None:
            print(f"Warning: Could not infer position columns for cow {cow_id}. Skipping.")
            continue
        
        uwb_data = uwb_data[['timestamp'] + pos_cols].copy()
        
        label_df = pd.read_csv(label_file)
        label_df = label_df[['timestamp', 'behavior']]
        label_df['timestamp'] = label_df['timestamp'].astype(np.float64)
        uwb_data['timestamp'] = uwb_data['timestamp'].astype(np.float64)
        
        merged_df = pd.merge_asof(uwb_data.sort_values('timestamp'), 
                                  label_df.sort_values('timestamp'), 
                                  on='timestamp', direction='nearest')
        merged_df = merged_df.ffill().bfill()
        
        merged_df = merged_df.rename(columns={
            pos_cols[0]: 'x',
            pos_cols[1]: 'y',
            pos_cols[2]: 'z'
        })
        
        combined_data_df = pd.concat([combined_data_df, merged_df], ignore_index=True)
    
    return combined_data_df

def immu_uwb_pre_data_loader(sensor_data_dir: str, id_list: List[int], date: str) -> pd.DataFrame:
    """Load and merge UWB plus formatted IMMU data for hybrid feature selection."""
    uwb_df = uwb_pre_data_loader(sensor_data_dir, id_list, date)
    if 'behavior' in uwb_df.columns:
        uwb_df = uwb_df.drop(columns=['behavior'])
    
    immu_df = immu_pre_data_loader(sensor_data_dir, id_list, date)
    if immu_df.empty or uwb_df.empty:
        return pd.DataFrame()
        
    immu_df = immu_df.rename(columns={
        c: f"immu_{c}" for c in immu_df.columns if c not in ["timestamp", "behavior"]
    })
    
    uwb_df['timestamp'] = uwb_df['timestamp'].astype(np.float64)
    immu_df['timestamp'] = immu_df['timestamp'].astype(np.float64)
    
    merged_df = pd.merge_asof(
        uwb_df.sort_values("timestamp"),
        immu_df.sort_values("timestamp"),
        on="timestamp",
        direction="nearest"
    ).dropna()
    return merged_df

def uwb_hd_akl_pre_data_loader(sensor_data_dir: str, id_list: List[int], date: str) -> pd.DataFrame:
    """Load and merge UWB, Head Direction, and Ankle data."""
    uwb_df = uwb_pre_data_loader(sensor_data_dir, id_list, date)
    
    combined_data_df = pd.DataFrame()
    for cow_id in id_list:
        cow_name = f'C{cow_id:02d}'
        tag_name = f'T{cow_id:02d}'
        
        ankle_file = os.path.join(sensor_data_dir, 'main_data', 'ankle', cow_name, f'{cow_name}_{date}.csv')
        head_file = os.path.join(sensor_data_dir, 'sub_data', 'head_direction', tag_name, f'T{cow_id:02d}_{date}.csv')
        
        if not (os.path.exists(ankle_file) and os.path.exists(head_file)):
            continue
            
        ankle_df = pd.read_csv(ankle_file)
        head_df = pd.read_csv(head_file)
        
        head_df['timestamp'] = head_df['timestamp'].astype(np.float64)
        ankle_df['timestamp'] = ankle_df['timestamp'].astype(np.float64)
        
        merged_sub = pd.merge_asof(
            head_df.sort_values("timestamp"),
            ankle_df.sort_values("timestamp"),
            on="timestamp",
            direction="nearest"
        )
        combined_data_df = pd.concat([combined_data_df, merged_sub], ignore_index=True)
    
    if combined_data_df.empty or uwb_df.empty:
        return pd.DataFrame()
        
    uwb_df['timestamp'] = uwb_df['timestamp'].astype(np.float64)
    final_df = pd.merge_asof(
        uwb_df.sort_values("timestamp"),
        combined_data_df.sort_values("timestamp"),
        on="timestamp",
        direction="nearest"
    ).dropna()
    
    return final_df
