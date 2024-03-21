import pandas as pd
import os

OUTPUT_DIR = '../results'
CSV_TRACKLETS = ['c010_tracklet.csv', 'c011_tracklet.csv', 'c012_tracklet.csv', 'c013_tracklet.csv', 'c014_tracklet.csv', 'c015_tracklet.csv']

def concatenate_files(file_list, output_file):
    concatenated_data = pd.DataFrame()  # Initialize an empty DataFrame to store concatenated data
    
    for file in file_list:
        data = pd.read_csv(file)  # Read each file
        if not concatenated_data.empty:
            # Adjust 'frame' column for each subsequent file to avoid overlapping frame values
            data['frame'] += concatenated_data['frame'].max() + 1
        concatenated_data = pd.concat([concatenated_data, data], ignore_index=True)  # Concatenate data
    
    concatenated_data.to_csv(output_file, index=False)  # Save concatenated data to a new CSV file

def concatenate_gt(file_list, output_file):
    concatenated_data = pd.DataFrame()  # Initialize an empty DataFrame to store concatenated data
    
    for file in file_list:
        data = pd.read_csv(file, header=None)  # Read each file without header row
        if not concatenated_data.empty:
            # Adjust 'frame' column for each subsequent file to avoid overlapping frame values
            data.iloc[:, 0] += concatenated_data.iloc[:, 0].max() + 1
        concatenated_data = pd.concat([concatenated_data, data], ignore_index=True)  # Concatenate data
    
    concatenated_data.to_csv(output_file, index=False, header=False)  # Save concatenated data to a new CSV file without header

# List of files to concatenate
csv_paths = []
for i, csv_path in enumerate(CSV_TRACKLETS):
    mtmc_csv_path = os.path.join(
        OUTPUT_DIR, f"{i}_{os.path.split(csv_path)[1]}")
    csv_paths.append(mtmc_csv_path)

gt_paths = ['./Data/train/S03/c010/gt/gt.txt',
            './Data/train/S03/c011/gt/gt.txt',
            './Data/train/S03/c012/gt/gt.txt',
            './Data/train/S03/c013/gt/gt.txt',
            './Data/train/S03/c014/gt/gt.txt',
            './Data/train/S03/c015/gt/gt.txt']

# Output file
output_csv_file = os.path.join(OUTPUT_DIR, 'S03_tracklets.csv')
output_gt_file = os.path.join(OUTPUT_DIR, 'S03_gt.csv')

# Call the function to concatenate files
concatenate_files(csv_paths, output_csv_file)
concatenate_gt(gt_paths, output_gt_file)