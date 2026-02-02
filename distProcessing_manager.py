import json
import glob
import os
import numpy as np
import pandas as pd
import re
from dist_utils import DistUtils

class distProcessing_manager:
    def __init__(self, folder_path,iso_path):
        # We store the path inside the object here
        self.folder_path = folder_path
        self.iso_path = iso_path
        self.master_data = {}
        self.utils = DistUtils()  # Create the tool object

    def removeOutlierGetCenter(self, **kwargs):
        """
        Acts as a bridge to DistUtils. 
        **kwargs catches all the filenames and parameters passed from the main script.
        """
        # Pass everything received directly to the utility object
        return self.utils.removeOutlierGetCenter(**kwargs)

    def merge_results(self):
        """Finds all JSON results and builds the master dictionary."""
        print(f"--- Starting Merge Process in {self.folder_path} ---")
        
        search_pattern = os.path.join(self.folder_path, "Source_*_results.json")
        file_list = glob.glob(search_pattern) # using glob since the order doesn't matter 
    
        if not file_list:
            print("No result files found to merge.")
            return

        print(f"Found {len(file_list)} files to merge.")

        temp_results = {}
        for file_path in file_list:
            file_name = os.path.basename(file_path)
            source_name = file_name.replace("Source_", "").replace("_results.json", "")
        
            try:
                with open(file_path, 'r') as f:
                    data = json.load(f)
                    temp_results[source_name] = data
            except Exception as e:
                print(f"Error reading {file_name}: {e}")

        # IMPORTANT: Store the results in the class object so other methods can see it
        self.master_data = temp_results

        # Save the master file
        master_output_path = os.path.join(self.folder_path, "MASTER_ALL_DISTANCES.json")
        with open(master_output_path, 'w') as f:
            json.dump(self.master_data, f, indent=4)
    
        print(f"✅ Success! Master file created at: {master_output_path}")



    def save_distance_matrix(self, subj_list, cur_region):
        """Generates the classic N x N square distance matrix."""
        num_subj = len(subj_list)
        dist_mat = np.zeros((num_subj, num_subj))

        for i, s1 in enumerate(subj_list):
            for j, s2 in enumerate(subj_list):
                # Pull distance from master_data
                dist = self.master_data.get(s1, {}).get(s2, {}).get('distance', 0)
                dist_mat[i, j] = dist

        # Format with clean names (handling the 'R' flip convention)
        subj_names = [re.sub(r'^R', 'flip-R', n) for n in subj_list]
        # Bug fix: replace function replaces some R in the middle of the name by flip-R
        #subj_names = [n.replace('R', 'flip-R') if n.startswith('R') else n for n in subj_list]
        df_dist = pd.DataFrame(dist_mat, index=subj_names, columns=subj_names)
        
        output_path = os.path.join(self.folder_path, f'distMat_{cur_region}.txt')
        df_dist.to_csv(output_path, index=True, index_label='subjName')
        print(f"✅ Raw Distance Matrix saved to: {output_path}")


    def save_transformation_df(self, cur_region):
        """Generates a clean, long-format CSV for all Trans/Rot data."""
        rows = []
        for src, targets in self.master_data.items():
            if str(src).startswith('R'):  ## Make sure naming for R is handled
                src_name = f"flip-{src}"
            else:
                src_name = src

            for tgt, metrics in targets.items(): # Ensure we have arrays to work with
                if str(tgt).startswith('R'):  ## Make sure naming for R is handled
                    tgt_name = f"flip-{tgt}"
                else:
                    tgt_name = tgt

                trans = np.array(metrics.get('translation', [0, 0, 0]))
                rot = np.array(metrics.get('rotation', [[0]*3]*3))
                
                # Use .item() to get the raw float value without brackets
                rows.append({
                    'Source': src_name,
                    'Target': tgt_name,
                    'dist': metrics.get('distance'),
                    'tx': float(trans[0]), 
                    'ty': float(trans[1]), 
                    'tz': float(trans[2]),
                    'r11': float(rot[0, 0]), 'r12': float(rot[0, 1]), 'r13': float(rot[0, 2]),
                    'r21': float(rot[1, 0]), 'r22': float(rot[1, 1]), 'r23': float(rot[1, 2]),
                    'r31': float(rot[2, 0]), 'r32': float(rot[2, 1]), 'r33': float(rot[2, 2])
                })

        df = pd.DataFrame(rows)
        output_path = os.path.join(self.folder_path, f'transRotMat_{cur_region}.txt')
        df.to_csv(output_path, index=False)
        print(f"✅ Clean Transformation Table saved to: {output_path}")        


    def save_symmetric_matrices(self, cur_region):
        """
        Reads the already saved distance matrix and creates 
        Symmetric Min and Max versions.
        """
        dist_path = os.path.join(self.folder_path, f'distMat_{cur_region}.txt')
        
        if not os.path.exists(dist_path):
            print(f"Error: Could not find {dist_path}. Run save_distance_matrix first.")
            return

        # 1. Load the square matrix
        df = pd.read_csv(dist_path, index_col=0)
        dist_mat = df.values

        # 2. Compute Symmetry
        # np.minimum compares the matrix with its own transpose element-wise
        dist_min = np.minimum(dist_mat, dist_mat.T)
        dist_max = np.maximum(dist_mat, dist_mat.T)

        # 3. Save Min Matrix
        df_min = pd.DataFrame(dist_min, index=df.index, columns=df.columns)
        min_path = os.path.join(self.folder_path, f'minDist{cur_region}.txt')
        df_min.to_csv(min_path, index=True, index_label='subjName')
        iso_min_path = os.path.join(self.iso_path, f'minDist{cur_region}.txt')
        df_min.to_csv(iso_min_path, index=True, index_label='subjName')

        # 4. Save Max Matrix
        df_max = pd.DataFrame(dist_max, index=df.index, columns=df.columns)
        max_path = os.path.join(self.folder_path, f'maxDist{cur_region}.txt')
        df_max.to_csv(max_path, index=True, index_label='subjName')
        iso_max_path = os.path.join(self.iso_path, f'maxDist{cur_region}.txt')
        df_max.to_csv(iso_max_path, index=True, index_label='subjName')

        print(f"✅ Symmetric Min Matrix saved to: {min_path}")
        print(f"✅ Symmetric Max Matrix saved to: {max_path}")
        print(f"✅ Symmetric Min Matrix saved to: {iso_min_path}")
        print(f"✅ Symmetric Max Matrix saved to: {iso_max_path}")


    ###############################################################  OLD format  #########################################################

    def save_matrices(self, subj_list, cur_region):
        """   OLD format 
           Builds and saves the two specific files: one distance, the other rotation and translation"""
        if not self.master_data:
            print("Error: No data in master_data. Run merge_results first.")
            return

        num_subj = len(subj_list)
        
        # 1. Initialize empty matrices
        # Distance matrix is N x N
        dist_mat = np.zeros((num_subj, num_subj))
        # Transformation matrix is N x (N * 12) 
        # (3 translation + 9 rotation per target)
        trans_rot_mat = np.zeros((num_subj, num_subj * 12))

        # 2. Fill the matrices
        for i, s1 in enumerate(subj_list):
            for j, s2 in enumerate(subj_list):
                entry = self.master_data.get(s1, {}).get(s2)
                
                # Column position for the 12-element transformation
                start_col = j * 12
                
                if entry:
                    # Fill Distance
                    dist_mat[i, j] = entry.get('distance', 0)
                    
                    # Fill Transformation (T1, T2, T3, R1...R9)
                    trans = entry.get('translation', [0,0,0])
                    rot = entry.get('rotation', [[0]*3]*3)
                    flat_rot = [val for row in rot for val in row]
                    
                    combined = trans + flat_rot
                    trans_rot_mat[i, start_col : start_col + 12] = combined
                
                elif i == j:
                    # Self-alignment (Identity)
                    dist_mat[i, j] = 0
                    trans_rot_mat[i, start_col : start_col + 12] = [0,0,0, 1,0,0, 0,1,0, 0,0,1]

        # 3. Create DataFrames for easy CSV saving
        # Clean subject names for the index/columns
        subj_names = [re.sub(r'^R', 'flip-R', name) for name in subj_list]
        # Bug fix: replace function replaces some R in the middle of the name by flip-R
        #subj_names = [name.replace('R', 'flip-R') if name.startswith('R') else name for name in subj_list]

        # Save Distance Matrix
        df_dist = pd.DataFrame(dist_mat, index=subj_names, columns=subj_names)
        dist_path = os.path.join(self.folder_path, f'distMat_{cur_region}.txt')
        df_dist.to_csv(dist_path, index=True, index_label='subjName')

        # Save Trans/Rot Matrix
        tr_cols = [str(k) for k in range(1, (num_subj * 12) + 1)]
        df_tr = pd.DataFrame(trans_rot_mat, index=subj_names, columns=tr_cols)
        tr_path = os.path.join(self.folder_path, f'transRotMat_{cur_region}.txt')
        df_tr.to_csv(tr_path, index=True, index_label='subjName')

        print(f"✅ Saved Distance Matrix to: {dist_path}")
        print(f"✅ Saved Trans/Rot Matrix to: {tr_path}")        