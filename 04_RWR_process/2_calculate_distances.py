#!/usr/bin/python3

# Get the passed variables
import sys
import os
import pandas as pd
import re
import separation_measure as sp


# Arguments get passed from command line/SLURM file
sublist = sys.argv[1]
seed_length = int(sys.argv[2])
matrix = sys.argv[3]
module_min = int(sys.argv[4])
module_max = int(sys.argv[5])


disease_ls = sublist.split(",")

# Create the path to the diseases, networks and where the results are to be stores
d_path = f"../00_Data/RWR_results/rwr_ranking_seeds_{seed_length}_{matrix}/"
n_path = f"../00_Data/Sparsified_networks/disease-multiplex-020924/"
r_path = f"../00_Data/Distance_results/distances_seeds_{seed_length}_{matrix}_module_{module_max}/"

# Create the result directory if it does not exist
if not os.path.exists(r_path):
    os.makedirs(r_path)
    print(f"Directory created: {r_path}")

# Create the filenames
filename_suffix = f"_seeds_{seed_length}_{matrix}.csv"
suffix_res_file = f"_seeds_{seed_length}_{matrix}_module_{module_max}"

for pair in disease_ls:
    matches = re.findall("(.*)-(.*)", pair)
    d0 = matches[0][0]
    d1 = matches[0][1]

    # Start the calculations
    df = sp.main(
        disease_pairs=(d0, d1),
        folder_path=d_path,
        filename_prefix="rwr_ranking_",
        filename_suffix=filename_suffix,
        networks_folder=n_path,
        min_rank=module_min,
        max_rank=module_max,
        multixrank=True,  # change accordingly
        multiplex="disease",  # change accordingly
    )

    # Save the df
    save_path = f"{r_path}distances_{d0}-{d1}{suffix_res_file}.csv"
    pd.DataFrame.to_csv(df, save_path)
