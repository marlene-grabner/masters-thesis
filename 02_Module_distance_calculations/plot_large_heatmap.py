import pandas as pd
import os
import networkx as nx
import re
import matplotlib.pyplot as plt
import random
import seaborn as sns
import numpy as np
import matplotlib.patches as mpatches
from scipy.cluster.hierarchy import linkage, leaves_list


##############################################################
"""
Reading in results
"""
##############################################################


def read_results_from_csv_multixrank(path, min_rank=0, max_rank=50):
    """
    Reads RWR results from a CSV, sorts by score, and returns top nodes.

    Parameters:
        path (str): Path to CSV file.
        min_rank (int): Minimum rank to consider.
        max_rank (int): Maximum rank to consider.

    Returns:
        top_nodes (list): Top nodes based on specified ranks.
    """

    df = pd.read_csv(path, sep="\t")  # Reading in the results
    df.sort_values(by=["score"], ascending=False, inplace=True)
    top_nodes = list(
        df["node"][min_rank:max_rank]
    )  # get only the nodes on the specified ranks
    return top_nodes


##############################################################
"""
Creating Data Frame for plotting
"""
##############################################################


def pair_df(distances_res_paths):
    """
    Creates a DataFrame for a disease pair with module information.

    Parameters:
        distances_res_paths (str): Path to distances results.

    Returns:
       sAB_df (pd.DataFrame): Dataframe with sAB distance data for disease pair.
    """

    sAB_df = pd.read_csv(distances_res_paths, index_col=1)
    return sAB_df


def create_combined_dataframe(df_dict, col_name):
    """
    Creates DataFrame of all sAB distance values from a dictionary of DataFrames.

    Parameters:
        df_dict (dict): Dictionary of DataFrames.
        col_name (str): Column name to extract.

    Returns:
        combined_df (pd.DataFrame): Combined DataFrame containing all sAB values.
    """

    # Get the first key and row indices
    first_key = next(iter(df_dict))
    row_indices = df_dict[first_key].index

    # Create a list to store all the DataFrames to concatenate
    dfs_to_concat = []

    # Iterate through the dictionary
    for key, df in df_dict.items():
        # Reindex the DataFrame to ensure the order of rows matches
        reindexed_df = df.reindex(row_indices)

        # Extract the specific column and add it to the list of DataFrames
        col_df = reindexed_df[
            [col_name]
        ]  # Select the column and keep it as a DataFrame
        col_df.columns = [key]  # Rename the column with the key
        dfs_to_concat.append(col_df)

    # Concatenate all the DataFrames at once along the columns axis
    combined_df = pd.concat(dfs_to_concat, axis=1)

    # Reformat the columns (if format_column_name is defined)
    combined_df.columns = [format_column_name(col) for col in combined_df.columns]

    return combined_df


# Reformat columns
def format_column_name(col):
    """
    Formats column name for display.

    Parameters:
        col (str): Original column name.

    Returns:
        str: Formatted column name.
    """

    # Join the tuple elements with ' - ' and remove '_' and ',' from the result
    return f"{col[0].replace('_', ' ')} - {col[1].replace('_', ' ')}"


##############################################################
"""
Plotting the DataFrame
"""
##############################################################


def plot_clustered(df, drop_cols=[], save_fig=None, cbar=False, as_png=False):
    """
    Generates clustered heatmap for overlap analysis.

    Parameters:
        df (pd.DataFrame): Data for heatmap.
        drop_cols (list): Columns to drop.
        save_fig (str): Path to save the figure.
        cbar (bool): Whether to display colorbar.
        as_png (bool): Save as PNG if True.

    Returns:
        seaborn.ClusterGrid: Clustered heatmap plot.
    """

    df_t = df.T
    df_t_d = drop_columns(df_t, drop_cols)
    df_transposed = rename_layers(df_t_d)
    df_flipped = df_transposed.iloc[::-1]

    # Handle NaN and infinite values for clustering
    df_clean = df_flipped.copy()

    # Replace any infinite values with NaNs first
    df_clean.replace([np.inf, -np.inf], np.nan, inplace=True)

    # Replace all NaNs with 0
    df_clean.fillna(0, inplace=True)

    # Create the clustermap
    if cbar:
        ax = sns.clustermap(
            df_clean,
            cmap="RdBu",
            linewidths=0.01,
            vmin=-0.7,
            vmax=0.7,
            figsize=(20, 10),
            row_cluster=True,  # Enable row clustering
            col_cluster=False,  # Disable column clustering
            dendrogram_ratio=(0.15, 0.15),  # Reduce dendrogram size
            cbar_pos=(0.17, 0.93, 0.52, 0.02),  # [left, bottom, width, height] for cbar
            cbar_kws={
                "orientation": "horizontal",
                "shrink": 0.5,
                "pad": 0.6,
                "ticks": [0.7, -0.7],
            },
        )

        ax.ax_row_dendrogram.set_visible(False)

        cbar = ax.ax_heatmap.collections[0].colorbar
        cbar.set_ticks([0.7, -0.7])  # Set the positions for ticks
        cbar.set_ticklabels(["Low overlap", "High overlap"], fontsize=20)

        # Remove y-axis ticks
        ax.ax_heatmap.set_yticks([])

        # Remove x-axis label (column name)
        ax.ax_heatmap.set_xlabel("")

        # Keep the xtick labels but increase their size
        plt.setp(
            ax.ax_heatmap.xaxis.get_majorticklabels(),
            rotation=45,
            fontsize=12,
            ha="right",
        )

        # Save the figure if needed
        if save_fig:
            if as_png:
                plt.savefig(f"{save_fig}", format="png", bbox_inches="tight")
            else:
                plt.savefig(f"{save_fig}", format="pdf", bbox_inches="tight")

        # Display the plot
        plt.show()

    # if there should not be a colorbar
    else:
        ax = sns.clustermap(
            df_clean,
            cmap="RdBu",
            linewidths=0.01,
            vmin=-0.7,
            vmax=0.7,
            figsize=(20, 10),
            row_cluster=True,  # Enable row clustering
            col_cluster=False,  # Enable column clustering
            dendrogram_ratio=(0.15, 0.15),  # Reduce dendrogram size
            cbar=False,
        )

        # Manually remove any leftover color bar ticks (even though cbar is False)
        if hasattr(ax, "cax"):
            ax.cax.set_visible(False)
            ax.cax.set_xticks([])  # Remove x ticks if they still exist
            ax.cax.set_yticks([])  # Remove y ticks if they still exist

        ax.ax_row_dendrogram.set_visible(False)

        # Remove y-axis ticks
        ax.ax_heatmap.set_yticks([])
        ax.ax_heatmap.set_xticks([])

        # Remove x-axis label (column name)
        ax.ax_heatmap.set_xlabel("")

        # Adjust the layout
        plt.tight_layout()  # Shrink the space around the plot

        # Save the figure if needed
        if save_fig:
            if as_png:
                plt.savefig(f"{save_fig}", format="png", bbox_inches="tight")
            else:
                plt.savefig(f"{save_fig}", format="pdf", bbox_inches="tight")

        # Display the plot
        plt.show()

    return ax


def drop_columns(df, drop_cols):
    """
    Drops specified columns from DataFrame.

    Parameters:
        df (pd.DataFrame): Original DataFrame.
        drop_cols (list): Columns to drop.

    Returns:
        pd.DataFrame: DataFrame without specified columns.
    """

    for col in drop_cols:
        df = df.drop(col, axis=1)
    return df


# Rename the layers for plotting
def rename_layers(df):
    """
    Renames layer columns for plotting.

    Parameters:
        df (pd.DataFrame): Original DataFrame.

    Returns:
        pd.DataFrame: DataFrame with renamed columns.
    """

    # Dictionary giving the original names of the layers and the once for the figure
    rename_dict = {
        "coex_WBL": "coex. Whole blood",
        "coex_VGN": "coex. Vagina",
        "coex_UTR": "coex. Uterus",
        "coex_TST": "coex. Testis",
        "coex_TNV": "coex. Tibial nerve",
        "coex_THY": "coex. Thyroid",
        "coex_STM": "coex. Stomach",
        "coex_SPL": "coex. Spleen",
        "coex_SMU": "coex. Skeletal muscle",
        "coex_SKN": "coex. Skin",
        "coex_PRS": "coex. Prostate",
        "coex_PNC": "coex. Pancreas",
        "coex_PIT": "coex. Pituitary",
        "coex_OVR": "coex. Ovary",
        "coex_MSG": "coex. Minor salivary gland",
        "coex_LVR": "coex. Liver",
        "coex_LNG": "coex. Lung",
        "coex_LCL": "coex. Lymphoblastoid cell line",
        "coex_KDN": "coex. Kidney cortex",
        "coex_ITI": "coex. Intestine terminal ileum",
        "coex_HRV": "coex. Heart left ventricle",
        "coex_HRA": "coex. Heart atrial appendag",
        "coex_GEJ": "coex. Gastroesophageal junction",
        "coex_FIB": "coex. Fibroblast cell line",
        "coex_EMS": "coex. Esophagus muscularis",
        "coex_EMC": "coex. Esophagus mucosa",
        "coex_CLT": "coex. Colon transverse",
        "coex_CLS": "coex. Colon sigmoid",
        "coex_BST": "coex. Breast",
        "coex_BRC": "coex. Brain cerebellum",
        "coex_BRB": "coex. Brain basal ganglia",
        "coex_BRO": "coex. Brain other",
        "coex_ATT": "coex. Artery tibial",
        "coex_ATC": "coex. Artery coronary",
        "coex_ATA": "coex. Artery aorta",
        "coex_ARG": "coex. Adrenal gland",
        "coex_ADV": "coex. Adipose visceral",
        "coex_ADS": "coex. Adipose subcutaneous",
        "coex_core": "co-expression core",
        "gene_gene_on_CbG_name_mapped": "Compund binds gene",
        "gene_gene_on_GcG_name_mapped": "Gene covariates with gene",
        #'GOMF': 'Molecular Function',
        "GI_net": "Genetic interaction",  # genetic interaction, built from biograde data
        "co-essential": "Co-essentiality",
        "reactome_copathway": "Co-pathway",
        "ppi": "Protein-interaction",
        "GOBP": "Biological Process",
        "MP": "Mammalian Phenotype",
        "HP": "Human Phenotype",
    }

    # Renaming according to the rename_dict
    df.rename(columns=rename_dict, inplace=True)
    # Specifying the desired order
    desired_order = list(rename_dict.values())
    # Changing the columns to the desired order
    df = df[desired_order]

    return df


##############################################################
"""
Main function to execute the rest of the code
"""
##############################################################


def main(
    disease_pairs: list,
    rwr_res_path: str,
    rwr_file_prefix: str,
    rwr_file_suffix: str,
    distances_res_path: str,
    distances_file_prefix: str,
    distances_file_suffix: str,
    rwr_max_rank: int = 50,
    rwr_min_rank: int = 0,
    save_fig: str = None,
    drop_cols: list = [],
    cbar=False,
    as_png=False,
    df_only=False,
):

    pair_dfs = {}

    for pair in disease_pairs:
        # Creating path to were the distances are written
        d_path = f"{distances_res_path}{distances_file_prefix}{pair[0]}-{pair[1]}{distances_file_suffix}"
        # Get sAB distance for the pair in DataFrame form
        df = pair_df(distances_res_paths=d_path)
        # Store these dataframes in dictionary
        pair_dfs[pair] = df

    # Concatenate all the dataframes to get a combined one containing all sAB distance values
    comb_df = create_combined_dataframe(pair_dfs, "sAB")

    # If specified only the dataframe of module distances will be returned
    if df_only:
        return comb_df

    # Usually the data will also be plotted
    else:
        plot = plot_clustered(comb_df, drop_cols, save_fig, cbar, as_png)
        return comb_df
