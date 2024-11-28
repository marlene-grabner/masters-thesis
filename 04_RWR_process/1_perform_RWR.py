import sys
import os
import multixrank
import pandas as pd
import shutil


sublist = sys.argv[1]
disease_ls = sublist.split(",")  # list of diseases is passed from the SLURM file
seed_len = int(sys.argv[2])  # Length of the seeds is passed from the SLURM file

"""
This function creates a temporary config.yml file which can be used for the RWR with multixrank
"""


def create_yml(disease, seed_len):

    with open("./config.yml", "r") as file:
        # reading in all the lines
        lines = file.readlines()
    # Removing the dummy seed file
    lines = lines[:-1]
    # Adding the new seed file
    lines.append(f"./Temp_seed_files/{disease}_seeds_{seed_len}.txt")

    # Creating the new temporary config file
    with open(
        f"./Temp_config_files/config_{disease}_seeds_{seed_len}.yml", "w"
    ) as file:
        file.writelines(lines)


"""
This function creates a temporary seed file containing those within the top "seed_len" genes from disgenet that are present in the network.
"""


def create_seed_file(disease, seed_len):

    with open(f"../00_Data/DisGeNET_seeds/{disease}_seeds.txt", "r") as file:
        # reading in all the genes
        genes = file.readlines()

    # extracting only the amount specified
    top_genes = genes[:seed_len]

    # Checking if those top genes are actually present in the network - adjust to the correct network node list
    with open(f"../00_Data/Network_nodes/nodes_in_network.txt", "r") as file:
        # reading in all the nodes
        nodes = file.readlines()

    seeds = list(set(top_genes).intersection(set(nodes)))

    # If heterogeneous multplex -> add disease name as seed
    seeds.insert(0, disease + "\n")

    with open(f"./Temp_seed_files/{disease}_seeds_{seed_len}.txt", "w") as file:
        file.writelines(seeds)


"""
Performing the random walk with restart
"""


for disease in disease_ls:

    # Creating temporary config file
    create_yml(disease, seed_len=seed_len)

    # Create temporary seed file
    create_seed_file(disease, seed_len=seed_len)

    # Creating the multixrank object using the temporary config file
    multixrank_obj = multixrank.Multixrank(
        config=f"config_{disease}_seeds_{seed_len}.yml", wdir="./"
    )

    # Performing the random walk
    ranking_df = multixrank_obj.random_walk_rank()

    # Saving the ranking
    multixrank_obj.write_ranking(
        ranking_df,
        path=f"../00_Data/RWR_results/rwr_ranking_seeds_{seed_len}_het_multiplex_dis_0209_gene_1309/rwr_ranking_{disease}_seeds_{seed_len}_disease-multiplex-020924",
        top=None,
    )

    # Removing the temporary config and seed file
    os.remove(f"./Temp_config_files/config_{disease}_seeds_{seed_len}.yml")
    os.remove(f"./Temp_seed_files/{disease}_seeds_{seed_len}.txt")
