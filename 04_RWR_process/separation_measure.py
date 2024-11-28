import pandas as pd
import networkx as nx
import numpy as np


# needs the path to the results, returns a list of the top ranked nodes by RWR
def read_results_from_csv(path, min_rank=0, max_rank=50):

    # Creating df from path
    df = pd.read_csv(path)

    # Calculating the accumulated ranking
    rank_columns = df.filter(regex="rank$")
    total_rank_sum = rank_columns.sum(axis=1)
    df["total_rank_sum"] = total_rank_sum
    df.sort_values(by=["total_rank_sum"], ascending=False, inplace=True)

    # Getting the names for the top ranked nodes
    name_cols = df.filter(regex="gene_name$")
    top_ls = list(df[name_cols.columns[0]][min_rank:max_rank])
    print(top_ls)
    return top_ls


# read in the results and get the top ranked nodes by RWR perforemd through multixrank
def read_results_from_csv_multixrank(path, min_rank=0, max_rank=50):
    df = pd.read_csv(path, sep="\t")
    df.sort_values(by=["score"], ascending=False, inplace=True)
    top_ls = list(df["node"][min_rank:max_rank])
    return top_ls


def common_layers(path_0, path_1):

    # Create the 2 dataframes
    df0 = pd.read_csv(path_0)
    df1 = pd.read_csv(path_1)

    # Getting the unique column names
    s0 = set(df0)
    s1 = set(df1)

    # Removing the suffixes and getting only the unique prefixes which correspond to the layer
    cols = []
    for col in s0.intersection(
        s1
    ):  # only those columns present in both df will be considered
        if col.startswith(
            "co.essential"
        ):  # since the . would break the code it will be changed to a '-'
            prefix = "co-essential"
        else:
            prefix = col.split(".")[0]
        cols.append(prefix)

    return list(set(cols))


#########################################################################
"""
Specify the list of layers for which the distance is to be calulated for
RWR performed by multixrank.
"""


#########################################################################
def multixrank_layers(multiplex):
    if multiplex == "gene":
        ls = [
            "coex_KDN",
            "coex_BST",
            "coex_OVR",
            "coex_LNG",
            "coex_ITI",
            "coex_VGN",
            "coex_HRV",
            "coex_MSG",
            "coex_ADV",
            "coex_EMS",
            "coex_SMU",
            "coex_ARG",
            "coex_TST",
            "coex_PIT",
            "coex_LVR",
            "coex_THY",
            "coex_PNC",
            "coex_ATC",
            "coex_BRO",
            "coex_SKN",
            "coex_ADS",
            "coex_GEJ",
            "coex_BRB",
            "coex_UTR",
            "coex_STM",
            "coex_HRA",
            "coex_PRS",
            "coex_ATA",
            "coex_FIB",
            "coex_BRC",
            "coex_ATT",
            "coex_TNV",
            "coex_SPL",
            "coex_LCL",
            "coex_CLS",
            "coex_EMC",
            "coex_CLT",
            "coex_core",
            "coex_WBL",
            "co-essential",
            "GOBP",
            "ppi",
            "reactome_copathway",
            "MP",
            "HP",
            "GI_net",
            "gene_gene_on_CbG_name_mapped",
            "gene_gene_on_GcG_name_mapped",
            "GOMF",
        ]
    if multiplex == "disease":
        ls = ["disease_on_genes", "disease_on_symptoms", "disease_on_drugs"]
    return ls


# Return list of only those genes, that are present in the specified graph
def clean_gene_list(G, gene_list):
    node_list = G.nodes
    cleaned_gene_list = []
    for i in gene_list:
        if i in node_list:
            cleaned_gene_list.append(i)

    return cleaned_gene_list

    ###########################################
    """
    Read a tab-separated values (TSV) file containing graph data and create a NetworkX Graph object.

    Parameters:
    - filename (str): The name of the TSV file to be read.
    - weighted (bool): A boolean flag indicating whether the edges of the graph are weighted. 
                       If True, the file is expected to contain an additional 'weight' column for edge weights.
                       Defaults to False.

    Returns:
    - Graph (nx.Graph): The NetworkX Graph object created from the TSV file data.
    """


##########################################


def tsv_to_graph(
    filename: str, weighted: bool = False, multixrank: bool = False
) -> nx.Graph:

    # Initialize a new NetworkX Graph object
    G = nx.Graph()

    # Open the TSV file and read its content
    with open(filename, "r") as f:

        if not multixrank:
            next(f)
            next(f)

        for line in f:

            tokens = line.strip().split("\t")
            # extract node names and weight (if applicable)
            u, v = tokens[:2]
            weight = float(tokens[2]) if weighted else None
            # edges to the graph
            if weighted:
                G.add_edge(u, v, weight=weight)
            else:
                G.add_edge(u, v)

    return G

    ########################################################################
    """
    Return the s_AB score (see Menche et al, Science, 2015)
    between nodes of perturbations a and b on the network g
    parameters:
    g = graph
    a_nodes = list of nodes
    b_nodes = list of nodes
    If there is no path between two nodes, this will be ignored for the calculation.
    """


########################################################################


def s_AB(g, a_nodes, b_nodes):

    AA_min_paths = []
    for a in a_nodes:
        d_AA_paths = [
            nx.shortest_path_length(g, a, b)
            for b in a_nodes
            if a != b and nx.has_path(g, a, b)
        ]
        if d_AA_paths:
            AA_min_paths.append(np.min(d_AA_paths))
    d_AA = np.mean(AA_min_paths)

    BB_min_paths = []
    for a in b_nodes:
        d_BB_paths = [
            nx.shortest_path_length(g, a, b)
            for b in b_nodes
            if a != b and nx.has_path(g, a, b)
        ]
        if d_BB_paths:
            BB_min_paths.append(np.min(d_BB_paths))
    d_BB = np.mean(BB_min_paths)

    AB_min_paths = []
    for a in a_nodes:
        d_AB_paths = [
            nx.shortest_path_length(g, a, b) for b in b_nodes if nx.has_path(g, a, b)
        ]
        if d_AB_paths:
            AB_min_paths.append(np.min(d_AB_paths))

    for b in b_nodes:
        d_BA_paths = [
            nx.shortest_path_length(g, b, a) for a in a_nodes if nx.has_path(g, a, b)
        ]
        if d_BA_paths:
            AB_min_paths.append(np.min(d_BA_paths))

    d_AB = np.mean(AB_min_paths)

    return d_AB - (d_AA + d_BB) / 2


#########################################################################
"""
Main function to be called for calculating the distances between modules
Input:
    - disease_pairs ... tuple of the names of the two diseases to be compared (names need to be as specified in the file)
    - folder_pat ... path to the folder where the RWR results for the two diseases are
    - filename_prefix ... prefix before the disease name in the RWR file
    - filename_suffix ... suffix after the disease name in the RWR file
    - networks_folder ... path to the folder with all the network layers that the RWR was performed on
    - min_rank ... lowest ranked node that shall be considered for the module distance
    - max_rank ... highes ranked node that shall be considered for the module distance
"""
#########################################################################


def main(
    disease_pairs=tuple,
    folder_path="./",
    filename_prefix="rwr_ranking_",
    filename_suffix="_seeds_1_disease-multiplex.csv",
    networks_folder="./",
    min_rank=0,
    max_rank=50,
    multixrank=False,
    multiplex="gene",
):

    # Creating the folder paths
    p0 = f"{folder_path}{filename_prefix}{disease_pairs[0]}{filename_suffix}"
    p1 = f"{folder_path}{filename_prefix}{disease_pairs[1]}{filename_suffix}"

    # Getting the top ranked nodes
    if not multixrank:
        l0 = read_results_from_csv(p0, min_rank, max_rank)
        l1 = read_results_from_csv(p1, min_rank, max_rank)
    else:
        l0 = read_results_from_csv_multixrank(p0, min_rank, max_rank)
        l1 = read_results_from_csv_multixrank(p1, min_rank, max_rank)

    # Getting the layers common to both diseases
    if not multixrank:
        layers = common_layers(p0, p1)
    else:
        layers = multixrank_layers(multiplex)

    d0_nodes = []
    d1_nodes = []
    sABs = []

    for layer in layers:
        # Create the networkx graph object
        G_path = f"{networks_folder}/{layer}.tsv"
        G = tsv_to_graph(G_path, multixrank=multixrank)

        # Only get the nodes present in a layer
        c0 = clean_gene_list(G, l0)
        c1 = clean_gene_list(G, l1)

        # If too few nodes are present the calculation can not give a meaningful result and therefor NaN is put in
        if len(c0) < (min_rank - max_rank / 4) or len(c1) < (min_rank - max_rank / 4):
            d0_nodes.append(len(c0))
            d1_nodes.append(len(c1))
            sABs.append(np.nan)

        # Calculate distance with sAB measure
        sAB = s_AB(G, c0, c1)

        # Append the number of nodes present in each layer and the calcualted distance to the lists
        # from which the dataframe will be created
        d0_nodes.append(len(c0))
        d1_nodes.append(len(c1))
        sABs.append(sAB)

    # Create a dataframe
    d = {
        "Layer": layers,
        f"{disease_pairs[0]} nodes": d0_nodes,
        f"{disease_pairs[1]} nodes": d1_nodes,
        "sAB": sABs,
    }

    return pd.DataFrame(d)
