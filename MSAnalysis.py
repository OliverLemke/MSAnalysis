#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: Oliver Lemke
"""

import os
import pathlib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib.patches import Patch
from matplotlib.gridspec import GridSpec
from sklearn.impute import KNNImputer
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.metrics import r2_score
import umap
from scipy.stats import ttest_ind, ranksums, mannwhitneyu
from statsmodels.stats.multitest import multipletests
import Color_Mix as cm

######
##### Plot_Scatter Protein replace R^2 with Pearson R????
##### Include hierarchical clustering
##### Include correlation analysis
##### Diff. Expressed proteins hist with groups
##### Include position selection?
##### DE add label/title (text)
##### Other selection methods for get selection
##### Add flag for pre-scaled data in volcano plot for FC-Berechnung  DONE
######

##### NaN-Euclidean distance for UMAP????

path = pathlib.Path().absolute()

def get_Selection(data, prefix, method="File", labels=None, label_colors=None):
    """
    Obtain a dictionary containing row names, indices, label and fixed color for a selected groups 

    Parameters
    ----------
    data : pandas.DataFrame
        DataFrame containing the data to be analyzed
    prefix : list of str
        List of strings to be filtered for.
    method : {"File"}
        Method for grouping:
            "File": Searching for str in row names. 
        The default is "File".
    labels : list of str, optional
        Labels for every group (Same length as prefix). The default is None.
    label_colors : list of str, optional
        Colors for every group (Same length as prefix). The default is None.

    Returns
    -------
    Selections : dict
        Dictionary denoting groups within the data
    """

    Selections = {}
    colors = cm.colors_noNoise(len(prefix)+1)
    if method in ["File"]: ##### Add further methods
        for ind,pre in enumerate(prefix):
            selection = {}
            if label_colors:
                selection.update({"Color":label_colors[ind]})
            else:
                selection.update({"Color":colors[ind]})
            if labels:
                selection.update({"Label":labels[ind]})
            else:
                selection.update({"Label":pre})
            if method=="File":
                selection.update({"FileNames":[file for file in data.index if pre in file]})
                selection.update({"Indices":[ind for ind,file in enumerate(data.index) if pre in file]})
            Selections.update({pre:selection})
        if len(prefix)==1:
            selection = {}
            selection.update({"Color":colors[ind+1]})
            selection.update({"Label":"Remaining"})
            if method=="File":
                selection.update({"FileNames":[file for file in data.index if pre not in file]})
                selection.update({"Indices":[ind for ind,file in enumerate(data.index) if pre not in file]})
            Selections.update({"Remaining":selection})
    else:
        raise ValueError("Selection method unknown")        
    return Selections


### Maybe add flag for outside legend???

def Plot_DimReduction(data, selection_dict, do_reduction=True, reduction_technique="pca", n_components=4, min_dist=0.0, n_neighbors=30, perplexity=30, learning_rate=10.0, var_cutoff=0.8, do_imputation=True, imputation="min", kNN=5, do_scaling=True, scaling ="log2", random_state=None, variance=None, output="DimReduction.png", grouped=False):
    """
    Plot the dimensionality reduction of the input data. 
    Imputation and Scaling of the data can be included

    Parameters
    ----------
    data : pandas.DataFrame
        Data to be analyzed.
    selection_dict : dict
        Dictionary containing groups. Obtained by get_Selection().
    do_reduction : bool, optional
        Perform a dimensionality reduction. The default is True.
    reduction_technique : {"pca","umap","tsne"}, optional
        Dimensionality reduction technique to be used:
            "pca": Principal Component Analysis (PCA) implemented in scikit-learn
            "umap": Uniform Manifold Approximation and Projection for Dimension Reduction (UMAP) implemented in umap-learn
            "tsne": t-distributed Stochastic Neighbor Embedding (t-SNE) implemented in scikit-learn
        The default is "pca".
    n_components : int, optional
        Reduced number of dimenstions. The default is 4.
    min_dist : float, optional
        Minimal distance (only for UMAP). The default is 0.0.
    n_neighbors : int, optional
        Number of neighbors (only for UMAP). The default is 30.
    perplexity : float, optional
        Perplexity (only for t-SNE). The default is 30.
    learning_rate : float, optional
        Learning Rate (only for t-SNE). The default is 10.0.
    var_cutoff : float, optional
        Variance cutoff for prior PCA to UMAP/t-SNE. The default is 0.8.
    do_imputation : bool, optional
        Perform imputation. The default is True.
    imputation : {"min","max","mean","median","zero","kNN"}, optional
        Imputationmethod to be used:
            "min": Use minimal value
            "max": Use maximal value
            "mean": Use mean value
            "median": Use median value
            "zero": Use 0 (Not to be combined with scaling log2/log10)
            "kNN": Use k-Nearest-Neighbor imputation implemented in scikit-learn
        The default is "min".
    kNN : int, optional
        Number of nearest neighbors for kNN-imputation. The default is 5.
    do_scaling : bool, optional
        Perform scaling of the data. The default is True.
    scaling : {"log2","log10","Standard","MinMax"}, optional
        Scaling of the data: 
            "log2": Logarithmic scaling with basis 2
            "log10": Logarithmic scaling with basis 10
            "Standard": Standard scaler to zero mean and unit variance implemented in scikit-learn
            "MinMax": Scaling in [0,1] implemented in scikit-learn
        The default is "log2".
    random_state : int, optional
        Seed for stochastic processes. The default is None.
    variance : array, optional
        Variances for precomputed principal components. The default is None.
    output : str, optional
        Name of the output file. The default is "DimReduction.png".
    grouped : bool
        One seperate plot for every selection. If set output only requires base name and no suffix. The default is False.

    Returns
    -------
    data_reduced : array
        Reduced data.
    """
    
    if (do_imputation) & (imputation != "kNN"):
        data_imputed = Impute_Data(data=data, method=imputation, kNN=kNN)
    else:
        data_imputed = data
    
    if do_scaling:
        data_scaled = Scale_Data(data=data_imputed, method=scaling)
    else:
        data_scaled = data_imputed
        
    if imputation == "kNN":
        data_scaled = Impute_Data(data=data_scaled, method=imputation, kNN=kNN)
    
    if do_reduction:
        if reduction_technique.lower() in ["pca","umap","tsne",None]:
            if reduction_technique.lower() == "pca":
                data_reduced, variance = DimReduction_PCA(data=data_scaled, n_components=n_components)
            elif reduction_technique.lower() == "umap":
                data_reduced = DimReduction_UMAP(data=data_scaled, n_components=n_components, min_dist=min_dist, n_neighbors=n_neighbors, var_cutoff=var_cutoff, random_state=random_state)
            elif reduction_technique.lower() == "tsne":
                data_reduced = DimReduction_tSNE(data=data_scaled, n_components=n_components, perplexity=perplexity, learning_rate=learning_rate, var_cutoff=var_cutoff, random_state=random_state)
                n_components = np.min((3,n_components))
            else:
                data_reduced = data_scaled.values
        else:
            raise ValueError("dimensionality reduction technique unknown")
    else:
        try:
            data_reduced = data_scaled.values
        except:
            data_reduced = data_scaled
    if grouped:
        list_of_samples = [index for key in selection_dict for index in selection_dict[key]["FileNames"]]
        list_of_indices = [index for key in selection_dict for index in selection_dict[key]["Indices"]]
        for key in selection_dict:
            selection_dict_n = {}
            selection_dict_n.update({"Remaining":{"FileNames":[index for index in list_of_samples if index not in selection_dict[key]["FileNames"]],
                                                  "Indices":[index for index in list_of_indices if index not in selection_dict[key]["Indices"]],
                                                  "Label":"Remaining",
                                                  "Color":"#7F7F7F"}})
            selection_dict_n.update({key:selection_dict[key]})
            fs = 25                       
            legend_elements = [Patch(facecolor=selection_dict_n[key]["Color"],label=selection_dict_n[key]["Label"]) for key in selection_dict_n]
            
            fig = plt.figure()
            if n_components > 3:
                fig.set_size_inches((n_components-1)*6,(n_components-1)*5)
            elif n_components==3:
                fig.set_size_inches((n_components-1)*7,(n_components-1)*5)
            else:
                fig.set_size_inches((n_components-1)*9,(n_components-1)*5)
            gs = GridSpec(n_components-1,n_components-1,figure=fig)
            for i in range(n_components-1):
                for j in range(i,n_components-1):
                    ax = fig.add_subplot(gs[j,i])
                    for key in selection_dict_n:
                        if key=="Remaining":
                            ax.scatter(data_reduced[selection_dict_n[key]["Indices"],i],data_reduced[selection_dict_n[key]["Indices"],j+1],c=selection_dict_n[key]["Color"],alpha=.5)
                        else:
                            ax.scatter(data_reduced[selection_dict_n[key]["Indices"],i],data_reduced[selection_dict_n[key]["Indices"],j+1],c=selection_dict_n[key]["Color"])
                    if j==n_components-2:
                        if reduction_technique.lower() == "pca":
                            try:
                                ax.set_xlabel("PC "+str(i+1)+", "+"{0:.2f}".format(variance[i]),fontsize=fs)
                            except:
                                ax.set_xlabel("PC "+str(i+1),fontsize=fs)
                        elif reduction_technique.lower() == "umap":
                            ax.set_xlabel("UMAP "+str(i+1),fontsize=fs)
                        elif reduction_technique.lower() == "tsne":
                            ax.set_xlabel("t-SNE "+str(i+1),fontsize=fs)
                        else:
                            ax.set_xlabel("Dimension "+str(i+1),fontsize=fs)
                        ax.tick_params(axis='x', labelsize=fs)
                    else:
                        ax.set_xticklabels([])
                    if i==0:
                        if reduction_technique.lower() == "pca":
                            try:
                                ax.set_ylabel("PC "+str(j+2)+", "+"{0:.2f}".format(variance[j+1]),fontsize=fs)
                            except:
                                ax.set_ylabel("PC "+str(j+2),fontsize=fs)
                        elif reduction_technique.lower() == "umap":
                            ax.set_ylabel("UMAP "+str(j+2),fontsize=fs)
                        elif reduction_technique.lower() == "tsne":
                            ax.set_ylabel("t-SNE "+str(j+2),fontsize=fs)
                        else:
                            ax.set_ylabel("Dimension "+str(j+2),fontsize=fs)
                        ax.tick_params(axis='y', labelsize=fs)
                        ax.yaxis.set_label_coords(-.2, 0.5)
                    else:
                        ax.set_yticklabels([])
                    ax.set_xlim(np.min(data_reduced[:,i])-1,np.max(data_reduced[:,i])+1)
                    ax.set_ylim(np.min(data_reduced[:,j+1])-1,np.max(data_reduced[:,j+1])+1)
            if n_components>3:
                plt.legend(handles=legend_elements,fontsize=fs,loc="center",bbox_to_anchor=(0,(n_components-1)),ncol=int(np.ceil(len(selection_dict_n)/10)))
            elif n_components==3:
                plt.legend(handles=legend_elements,fontsize=fs,loc="upper left",bbox_to_anchor=(1.05,2),ncol=int(np.ceil(len(selection_dict_n)/10)))
            else:
                plt.legend(handles=legend_elements,fontsize=fs,loc="upper left",bbox_to_anchor=((n_components-1),1),ncol=int(np.ceil(len(selection_dict_n)/10)))
            
            plt.savefig(os.path.join(path,output+"_"+key+".png"),bbox_inches='tight')
        
    else:
        fs = 25                       
        legend_elements = [Patch(facecolor=selection_dict[key]["Color"],label=selection_dict[key]["Label"]) for key in selection_dict]
        
        fig = plt.figure()
        if n_components > 3:
            fig.set_size_inches((n_components-1)*6,(n_components-1)*5)
        elif n_components==3:
            fig.set_size_inches((n_components-1)*7,(n_components-1)*5)
        else:
            fig.set_size_inches((n_components-1)*9,(n_components-1)*5)
        gs = GridSpec(n_components-1,n_components-1,figure=fig)
        for i in range(n_components-1):
            for j in range(i,n_components-1):
                ax = fig.add_subplot(gs[j,i])
                for key in selection_dict:
                    ax.scatter(data_reduced[selection_dict[key]["Indices"],i],data_reduced[selection_dict[key]["Indices"],j+1],c=selection_dict[key]["Color"])
                if j==n_components-2:
                    if reduction_technique.lower() == "pca":
                        try:
                            ax.set_xlabel("PC "+str(i+1)+", "+"{0:.2f}".format(variance[i]),fontsize=fs)
                        except:
                            ax.set_xlabel("PC "+str(i+1),fontsize=fs)
                    elif reduction_technique.lower() == "umap":
                        ax.set_xlabel("UMAP "+str(i+1),fontsize=fs)
                    elif reduction_technique.lower() == "tsne":
                        ax.set_xlabel("t-SNE "+str(i+1),fontsize=fs)
                    else:
                        ax.set_xlabel("Dimension "+str(i+1),fontsize=fs)
                    ax.tick_params(axis='x', labelsize=fs)
                else:
                    ax.set_xticklabels([])
                if i==0:
                    if reduction_technique.lower() == "pca":
                        try:
                            ax.set_ylabel("PC "+str(j+2)+", "+"{0:.2f}".format(variance[j+1]),fontsize=fs)
                        except:
                            ax.set_ylabel("PC "+str(j+2),fontsize=fs)
                    elif reduction_technique.lower() == "umap":
                        ax.set_ylabel("UMAP "+str(j+2),fontsize=fs)
                    elif reduction_technique.lower() == "tsne":
                        ax.set_ylabel("t-SNE "+str(j+2),fontsize=fs)
                    else:
                        ax.set_ylabel("Dimension "+str(j+2),fontsize=fs)
                    ax.tick_params(axis='y', labelsize=fs)
                    ax.yaxis.set_label_coords(-.2, 0.5)
                else:
                    ax.set_yticklabels([])
                ax.set_xlim(np.min(data_reduced[:,i])-1,np.max(data_reduced[:,i])+1)
                ax.set_ylim(np.min(data_reduced[:,j+1])-1,np.max(data_reduced[:,j+1])+1)
        if n_components>3:
            plt.legend(handles=legend_elements,fontsize=fs,loc="center",bbox_to_anchor=(0,(n_components-1)),ncol=int(np.ceil(len(selection_dict)/10)))
        elif n_components==3:
            plt.legend(handles=legend_elements,fontsize=fs,loc="upper left",bbox_to_anchor=(1.05,2),ncol=int(np.ceil(len(selection_dict)/10)))
        else:
            plt.legend(handles=legend_elements,fontsize=fs,loc="upper left",bbox_to_anchor=((n_components-1),1),ncol=int(np.ceil(len(selection_dict)/10)))
        
        plt.savefig(os.path.join(path,output),bbox_inches='tight')
    return data_reduced

def DimReduction_PCA(data, n_components=4):
    """
    Dimnesionality reduction using principal component analysis.

    Parameters
    ----------
    data : pandas.DataFrame
        Data to be analyzed.
    n_components : int, optional
        Reduced number of dimenstions. The default is 4.

    Returns
    -------
    data_pca : array
        Reduced data.
    variance : array
        Variances of principal components.
    """
    
    pca = PCA(n_components=n_components)
    data_pca = pca.fit_transform(data)
    variance = pca.explained_variance_ratio_
    return data_pca, variance

def DimReduction_UMAP(data, n_components=4, min_dist=0.0, n_neighbors=30, var_cutoff=1, random_state=None):
    """
    Dimensionality reduction using Uniform Manifold Approximation and Projection for Dimension Reduction.
    Parameters
    ----------
    data : pandas.DataFrame
        Data to be analyzed.
    n_components : int, optional
        Reduced number of dimenstions. The default is 4.
    min_dist : float, optional
        Minimal distance. The default is 0.0.
    n_neighbors : int, optional
        Number of neighbors. The default is 30.
    var_cutoff : float, optional
        Variance cutoff for prior PCA. The default is 1.
    random_state : int, optional
        Seed for stochastic processes. The default is None.

    Returns
    -------
    data_umap : array
        Reduced data.
    """   
    try:
        pca_data = PCA().fit(data)
        dim = np.where(np.cumsum(pca_data.explained_variance_ratio_)>var_cutoff)[0][1]
        data_pca = pca_data.transform(data)[:,:dim]
        data_umap = umap.UMAP(n_neighbors=n_neighbors, min_dist=min_dist, n_components=n_components, random_state=random_state).fit_transform(data_pca)
    except:
        print("Taking full dimensionality for UMAP input")
        data_umap = umap.UMAP(n_neighbors=n_neighbors, min_dist=min_dist, n_components=n_components, random_state=random_state).fit_transform(data)  
    return data_umap
    
def DimReduction_tSNE(data, n_components=3, perplexity=30, learning_rate = 10.0, var_cutoff=1, random_state=None):
    """
    Dimensionality reduction using t-distributed Stochastic Neighbor Embedding.

    Parameters
    ----------
    data : pandas.DataFrame
        Data to be analyzed.
    n_components : int, optional
        Reduced number of dimenstions. The default is 3.
    perplexity : float, optional
        Perplexity. The default is 30.
    learning_rate : float, optional
        Learning Rate. The default is 10.0.
    var_cutoff : float, optional
        Variance cutoff for prior PCA. The default is 1.
    random_state : int, optional
        Seed for stochastic processes. The default is None.

    Returns
    -------
    data_tsne : array
        Reduced data.
    """
    
    if n_components > 3:
        n_components = 3
        print("Maximal number of components for t-SNE is 3")
    try:
        pca_data = PCA().fit(data)
        dim = np.where(np.cumsum(pca_data.explained_variance_ratio_)>var_cutoff)[0][1]
        data_pca = pca_data.transform(data)[:,:dim]
        data_tsne = TSNE(perplexity=perplexity, learning_rate=learning_rate, n_components=n_components, random_state=random_state).fit_transform(data_pca)
    except:
        print("Taking full dimensionality for t-SNE input")
        data_tsne = TSNE(perplexity=perplexity, learning_rate=learning_rate, n_components=n_components, random_state=random_state).fit_transform(data)
    return data_tsne    
    
def Impute_Data(data, method="median", kNN=5):
    """
    Data imputation for NaN values

    Parameters
    ----------
    data : pandas.DataFrame
        Data to be analyzed.
    method : {"min","max","mean","median","zero","kNN"}, optional
        Imputation method to be used:
            "min": Use minimal value
            "max": Use maximal value
            "mean": Use mean value
            "median": Use median value
            "zero": Use 0 (Not to be combined with scaling log2/log10)
            "kNN": Use k-Nearest-Neighbor imputation implemented in scikit-learn
        The default is "median".
    kNN : int, optional
        Number of nearest neighbors for kNN-imputation. The default is 5.

    Returns
    -------
    data_imputed : pandas.DataFrame
        Imputed data.
    """
    
    if method in ["min","max","median","mean","zero","kNN"]:
        if method == "min":
            data_imputed = data.fillna(data.min())
        elif method == "max":
            data_imputed = data.fillna(data.max())
        elif method == "median":
            data_imputed = data.fillna(data.median())
        elif method == "mean":
            data_imputed = data.fillna(data.mean())
        elif method == "zero":
            data_imputed = data.fillna(0)
        elif method == "kNN":
            imputer = KNNImputer(n_neighbors=kNN, weights='uniform', metric='nan_euclidean')
            try:
                data_imputed = pd.DataFrame(index = data.index, columns = data.columns, data = imputer.fit_transform(data))
            except:
                data_imputed = imputer.fit_transform(data)
    else:
        raise ValueError("Impuation method unknown")
    return data_imputed
    
def Scale_Data(data, method="log2"):
    """
    Scaling of data

    Parameters
    ----------
    data : pandas.DataFrame
        Data to be analyzed.
    method : {"log2","log10","Standard","MinMax"}, optional
        Scaling of the data to be used: 
            "log2": Logarithmic scaling with basis 2
            "log10": Logarithmic scaling with basis 10
            "Standard": Standard scaler to zero mean and unit variance implemented in scikit-learn
            "MinMax": Scaling in [0,1] implemented in scikit-learn
        The default is "log2".

    Returns
    -------
    data_scaled : pandas.DataFrame
        Scaled data.
    """
    
    if method in ["log2","log10","Standard","MinMax"]:
        if method == "log2":
            if 0 in data.values:
                raise ValueError("Logarithm not defined for imputation method 0")
            else:
                data_scaled = np.log2(data)
        elif method == "log10":
            if 0 in data.values:
                raise ValueError("Logarithm not defined for imputation method 0")
            else:
                data_scaled = np.log10(data)
        elif method == "Standard":
            data_scaled = StandardScaler().fit_transform(data)
        elif method == "MinMax":
            data_scaled = MinMaxScaler().fit_transform(data)
    else:
        raise ValueError("Scaling method unknown")
    return data_scaled

def Plot_Scatter_Protein(data, selection_dict, output="Scatter_Protein.png", scale="log2", groups=None):
    """
    Plot the median protein quantity correlation for different groups

    Parameters
    ----------
    data : pandas.DataFrame
        Data to be analyzed.
    selection_dict : dict
        Dictionary containing groups. Obtained by get_Selection().
    output : str, optional
        Name of the output file. The default is "Scatter_Protein.png".
    scale : {"log2","log10","linear"}, optional
        Scaling of the data. The default is "log2".
    groups : list of str, optional
        Groups to be analyzed. Must be keys of selection_dict. The default is None.
    """
    
    if scale not in ["linear","log2","log10"]:
        raise ValueError("Scale not defined")
        
    fs=25
    if not groups:
        keys = list(selection_dict.keys())  
    else:
        keys = groups
        
    fig = plt.figure()
    fig.set_size_inches((len(keys)-1)*6,(len(keys)-1)*5)
    gs = GridSpec(len(keys)-1,len(keys)-1,figure=fig)
    for i in range(len(keys)-1):
        for j in range(i,len(keys)-1):
            ax = fig.add_subplot(gs[j,i])
            if scale == "log2":
                A = np.nanmedian(np.log2(data).loc[selection_dict[keys[i]]["FileNames"],:],axis=0)
                B = np.nanmedian(np.log2(data).loc[selection_dict[keys[j+1]]["FileNames"],:],axis=0)
                to_be_included = ~np.isnan(A+B)
                A = A[to_be_included]
                B = B[to_be_included]               
                ax.scatter(A,B,c="C0",alpha=.5)                           
                ax.text(np.min(A),np.max(B)-1,r"$R^2 = {0:.2f}$".format(r2_score(A,B)),fontsize=fs)
                ax.set_xlim(np.min(A)-1,np.max(A)+1)
                ax.set_ylim(np.min(B)-1,np.max(B)+1)
            elif scale == "log10":
                A = np.nanmedian(np.log10(data).loc[selection_dict[keys[i]]["FileNames"],:],axis=0)
                B = np.nanmedian(np.log10(data).loc[selection_dict[keys[j+1]]["FileNames"],:],axis=0)
                to_be_included = ~np.isnan(A+B)
                A = A[to_be_included]
                B = B[to_be_included]               
                ax.scatter(A,B,c="C0",alpha=.5)                           
                ax.text(np.min(A),np.max(B)-1,r"$R^2 = {0:.2f}$".format(r2_score(A,B)),fontsize=fs)
                ax.set_xlim(np.min(A)-1,np.max(A)+1)
                ax.set_ylim(np.min(B)-1,np.max(B)+1)
            elif scale == "linear":
                A = np.nanmedian(data.loc[selection_dict[keys[i]]["FileNames"],:],axis=0)
                B = np.nanmedian(data.loc[selection_dict[keys[j+1]]["FileNames"],:],axis=0)
                to_be_included = ~np.isnan(A+B)
                A = A[to_be_included]
                B = B[to_be_included]               
                ax.scatter(A,B,c="C0",alpha=.5)                           
                ax.text(np.min(A),np.max(B)-1,r"$R^2 = {0:.2f}$".format(r2_score(A,B)),fontsize=fs)
                ax.set_xlim(np.min(A)-1,np.max(A)+1)
                ax.set_ylim(np.min(B)-1,np.max(B)+1)

            ax.plot([0,1e+10],[0,1e+10],c="k",ls=":")
            if j==len(keys)-2:
                ax.set_xlabel(str(selection_dict[keys[i]]["Label"]),fontsize=fs)
                ax.tick_params(axis='x', labelsize=fs)
            else:
                ax.set_xticklabels([])
            if i==0:
                ax.set_ylabel(str(selection_dict[keys[j+1]]["Label"]),fontsize=fs)
                ax.tick_params(axis='y', labelsize=fs)
                if scale == "linear":
                    ax.yaxis.set_label_coords(-.4, 0.5)
                else:
                    ax.yaxis.set_label_coords(-.2, 0.5)
            else:
                ax.set_yticklabels([])
    plt.legend(handles=[Patch(label=str(scale)+"(Protein Quantity)")],fontsize=fs,loc="center",bbox_to_anchor=(0,(len(keys)-1)))
    plt.savefig(os.path.join(path,output),bbox_inches='tight')
    return
    
def Plot_BoxPlot_Proteins(data, selection_dict, proteins, output="BoxPlot_Proteins.png", scale="log2"):
    """
    Box-Plot of selected proteins for different groups.

    Parameters
    ----------
    data : pandas.DataFrame
        Data to be analyzed.
    selection_dict : dict
        Dictionary containing groups. Obtained by get_Selection().
    proteins : list of str
        Proteins to be analyzed. Must be columns of data
    output : str, optional
        Name of the ouput file. The default is "BoxPlot_Proteins.png".
    scale : {"log2", "log10", "linear"}, optional
        Scaling of the data. The default is "log2".
    """
    
    if scale not in ["linear","log2","log10"]:
        raise ValueError("Scale not defined")
    
    label_offset = 0.1 * np.max([len(selection_dict[key]["Label"]) for key in selection_dict.keys()])
    xlabel = False
    N = len(proteins)
    fs = 25

    fig = plt.figure()
    fig.set_size_inches(20,(np.ceil(N/4)*5)+label_offset)
    gs = GridSpec(int(np.ceil(N/4)),8,figure=fig, wspace=1)
    
    for i,protein in enumerate(proteins):
        
        if np.floor(i/4)<(np.ceil(N/4)-1):
            ax = fig.add_subplot(gs[int(np.floor(i/4)),int(np.mod(i,4)*2):int((np.mod(i,4)+1)*2)])
        else:
            xlabel = True
            if np.mod(N,4)==0:   
                ax = fig.add_subplot(gs[int(np.floor(i/4)),int(np.mod(i,4)*2):int((np.mod(i,4)+1)*2)])
            elif np.mod(N,4)==1: 
                ax = fig.add_subplot(gs[int(np.floor(i/4)),int((np.mod(i,4)*2)+3):int(((np.mod(i,4)+1)*2)+3)])
            elif np.mod(N,4)==2:
                ax = fig.add_subplot(gs[int(np.floor(i/4)),int((np.mod(i,4)*2)+2):int(((np.mod(i,4)+1)*2)+2)])
            else:
                ax = fig.add_subplot(gs[int(np.floor(i/4)),int((np.mod(i,4)*2)+1):int(((np.mod(i,4)+1)*2)+1)])
        for j,key in enumerate(selection_dict):
            data_masked = data.loc[selection_dict[key]["FileNames"],protein].dropna()
            if scale == "log2":
                ax.boxplot(np.log2(data_masked),positions=[j],boxprops=dict(color=selection_dict[key]["Color"]))
            if scale == "log10":
                ax.boxplot(np.log10(data_masked),positions=[j],boxprops=dict(color=selection_dict[key]["Color"]))
            if scale == "linear":
                ax.boxplot(data_masked,positions=[j],boxprops=dict(color=selection_dict[key]["Color"]))
        ax.set_title(protein,fontsize=fs)
        if xlabel:
            ax.tick_params(axis='x', labelsize=fs)
            ax.set_xticklabels([selection_dict[key]["Label"] for key in selection_dict.keys()],rotation=90,fontsize=fs)
        else:
            ax.set_xticklabels([])
        ax.tick_params(axis='y', labelsize=fs)
        if (np.mod(i,4)==0) & (np.floor(i/4)==np.floor(N/8)):
            if scale == "log2":
                ax.set_ylabel("log2(Protein Quantity)",fontsize=fs)
                ax.yaxis.set_label_coords(-.25, 0.5)
            if scale == "log10":
                ax.set_ylabel("log10(Protein Quantity)",fontsize=fs)
                ax.yaxis.set_label_coords(-.25, 0.5)
            if scale == "linear":
                ax.set_ylabel("linear(Protein Quantity)",fontsize=fs)
                ax.yaxis.set_label_coords(-.4, 0.5)
    plt.savefig(os.path.join(path,output), bbox_inches='tight')
    return

def get_TopFC(data, selection_dict, groups=["qc","Remaining"], Top_N=10, method="median"):
    """
    Obtain the proteins, which show the largest fold-change between two groups

    Parameters
    ----------
    data : pandas.DataFrame
        Data to be analyzed.
    selection_dict : dict
        Dictionary containing groups. Obtained by get_Selection().
    groups : list of str, optional
        Groups to be compared. Must be keys of selection_dict. The default is ["qc","Remaining"].
    Top_N : int, optional
        Number of proteins to be extracted. The default is 10.
    method : {"median","mean"}, optional
        Method to compare protein quantities. The default is "median".

    Returns
    -------
    proteins : pandas.core.series
        N proteins which largest changes between the two groups. Mean/Median as well as protein is returned.
    """
    
    Top_N=np.min((Top_N,len(data.columns.values)))
    
    if method in ["median","mean"]:
        if method == "median":
            fold_change_abs = np.abs(np.log2(data.loc[selection_dict[groups[1]]["FileNames"],:].median()/data.loc[selection_dict[groups[0]]["FileNames"],:].median()))
        elif method == "mean":
            fold_change_abs = np.abs(np.log2(data.loc[selection_dict[groups[1]]["FileNames"],:].mean()/data.loc[selection_dict[groups[0]]["FileNames"],:].mean()))
    else:
        raise ValueError("Method not found")
    return fold_change_abs.nlargest(Top_N)

####
####
#### Get the rejects p[0] as an output?
####
####

def Plot_Volcano(data, selection_dict, groups=["qc","Remaining"], p_cut_1=.05, p_cut_2=.01, fc_cut=2, label=True, p_values=None, method_p="t-test", fold_changes=None, method_fc="median", alpha=.05, output="Volcano.png", title=None, write_file=False, file_out="Volcano.tsv", logscaled=False):
    """
    Get a Volcano plot for differential expression analysis. If not precomputed, p-values are adjusted using Benjamini-Hochberg correction.
    
    Parameters
    ----------
    data : pandas.DataFrame
        Data to be analyzed.
    selection_dict : dict
        Dictionary containing groups. Obtained by get_Selection().
    groups : list of str, optional
        Groups to be compared. Must be keys of selection_dict. The default is ["qc","Remaining"].
    p_cut_1 : float, optional
        First p-Value cut-off. The default is .05.
    p_cut_2 : float, optional
        Second p-Value cut-off (For only one cutoff use p_cut_1 = p_cut_2). The default is .01.
    fc_cut : float, optional
        log10(Fold Change) cut-off. The default is 2.
    label : bool, optional
        Label differential expressed proteins. The default is True.
    p_values : array, optional
        Precomputed p-values. The default is None.
    method_p : {"t-test","Welch","Wilcoxon"}, optional
        Method for p-Value calculation:
            "t-test": Perform Student's t-test (equal varaince) implemented in Scipy
            "Welch": Perform Welch's t-test (different varainces) implemented in Scipy
            "Wilcoxon": Perform Wilcoxon-Ranksum test implemented in Scipy
            "Mann-Whitney-U": Perform Mann-Whitney-U test implemented in Scipy
        The default is "t-test".
    fold_changes : pandas.core.series, optional
        Precalculated fold changes. The default is None.
    method_fc : {"median","mean","min","max"}, optional
        Method for fold change calculation:
            "min": Use minimal value
            "max": Use maximal value
            "mean": Use mean value
            "median": Use median value
        The default is "median".
    alpha : float, optional
        alpha for p-value correction implemented in statsmodels. The default is .05.
    output : str, optional
        Name of the ouput file. The default is "Volcano.png".
    title : str, optional
        Title of the plot. The default is None.
    write_file : bool, optional
        Write differential expressed genes to file. The default is False.
    file_out : str, optional
        Name of the output file if write_file=True. The default is "Volcano.tsv".
    logscaled : bool, optional
        True if data are already log2-transformed. The default is False.

    Returns
    -------
    p_adjusted : array
        Adjusted p-values.
    fold_changes : pandas.core.series
        Foldchanges (labels and values).
    """
    
    if groups[0] not in list(selection_dict.keys()):
        raise ValueError(groups[0]+" not found in selection_dict")
    elif groups[1] not in list(selection_dict.keys()):
        raise ValueError(groups[1]+" not found in selection_dict")
    
    if not fold_changes:
        if method_fc in ["median","mean","min","max"]:
            if logscaled:
                if method_fc == "median":
                    fold_changes = data.loc[selection_dict[groups[1]]["FileNames"],:].median()-data.loc[selection_dict[groups[0]]["FileNames"],:].median()
                elif method_fc == "mean":
                    fold_changes = data.loc[selection_dict[groups[1]]["FileNames"],:].mean()-data.loc[selection_dict[groups[0]]["FileNames"],:].mean()
                elif method_fc == "min":
                    fold_changes = data.loc[selection_dict[groups[1]]["FileNames"],:].min()-data.loc[selection_dict[groups[0]]["FileNames"],:].min()
                elif method_fc == "max":
                    fold_changes = data.loc[selection_dict[groups[1]]["FileNames"],:].max()-data.loc[selection_dict[groups[0]]["FileNames"],:].max()
            else:
                if method_fc == "median":
                    fold_changes = data.loc[selection_dict[groups[1]]["FileNames"],:].median()/data.loc[selection_dict[groups[0]]["FileNames"],:].median()
                elif method_fc == "mean":
                    fold_changes = data.loc[selection_dict[groups[1]]["FileNames"],:].mean()/data.loc[selection_dict[groups[0]]["FileNames"],:].mean()
                elif method_fc == "min":
                    fold_changes = data.loc[selection_dict[groups[1]]["FileNames"],:].min()/data.loc[selection_dict[groups[0]]["FileNames"],:].min()
                elif method_fc == "max":
                    fold_changes = data.loc[selection_dict[groups[1]]["FileNames"],:].max()/data.loc[selection_dict[groups[0]]["FileNames"],:].max()
        else:
            raise ValueError("Method for fold-change calculation not found")    
    
    if not p_values:
        if method_p in ["t-test","Welch","Wilcoxon","Mann-Whitney-U"]:
            if method_p == "t-test":
                p_values = ttest_ind(data.loc[selection_dict[groups[0]]["FileNames"],:],data.loc[selection_dict[groups[1]]["FileNames"],:],equal_var=True,nan_policy="omit")[1]
            elif method_p == "Welch":
                p_values = ttest_ind(data.loc[selection_dict[groups[0]]["FileNames"],:],data.loc[selection_dict[groups[1]]["FileNames"],:],equal_var=False,nan_policy="omit")[1]
            elif method_p == "Wilcoxon":
                p_values = np.asarray([ranksums(data.loc[selection_dict[groups[0]]["FileNames"],column],data.loc[selection_dict[groups[1]]["FileNames"],column])[1] for column in data.columns])
            elif method_p == "Mann-Whitney-U":
                p_values = np.asarray([mannwhitneyu(data.loc[selection_dict[groups[0]]["FileNames"],column],data.loc[selection_dict[groups[1]]["FileNames"],column])[1] for column in data.columns])
            adjust = True
        else:
            raise ValueError("Method for p-value calculation not found")
    
    if np.isnan(fold_changes.values).any():
        p_values = p_values[~np.isnan(fold_changes.values)]
        fold_changes.drop(index=fold_changes.index.values[np.where(np.isnan(fold_changes.values))], inplace=True)
    if np.ma.is_masked(p_values):
        fold_changes.drop(index=fold_changes.index.values[np.where(~(p_values>0).filled(False))], inplace=True)
        p_values = p_values[(p_values>0).filled(False)]
    
    if adjust:
        p_adjusted = multipletests(p_values,alpha=alpha,method="fdr_bh")[1]
        print("adjusting p-values")
    else:
        p_adjusted = p_values
    
    if label:
        genes = fold_changes.index.values
    
    fs = 20
    fig,ax = plt.subplots()
    fig.set_size_inches(10,8)
    if logscaled:
        s0 = (((fold_changes.values)<=np.min((fc_cut,-1*fc_cut))) | ((fold_changes.values)>=np.max((fc_cut,-1*fc_cut)))) & (-1*np.log10(p_adjusted)<=np.min((-1*np.log10(p_cut_1),-1*np.log10(p_cut_2))))
        s1 = ((fold_changes.values)>np.min((fc_cut,-1*fc_cut))) & ((fold_changes.values)<np.max((fc_cut,-1*fc_cut)))
        s2 = (((fold_changes.values)<=np.min((fc_cut,-1*fc_cut))) | ((fold_changes.values)>=np.max((fc_cut,-1*fc_cut)))) & (-1*np.log10(p_adjusted)>=np.min((-1*np.log10(p_cut_1),-1*np.log10(p_cut_2)))) & (-1*np.log10(p_adjusted)<np.max((-1*np.log10(p_cut_1),-1*np.log10(p_cut_2))))
        s3 = (((fold_changes.values)<=np.min((fc_cut,-1*fc_cut))) | ((fold_changes.values)>=np.max((fc_cut,-1*fc_cut)))) & (-1*np.log10(p_adjusted)>=np.max((-1*np.log10(p_cut_1),-1*np.log10(p_cut_2))))
        ax.scatter((fold_changes.values)[s0],-1*np.log10(p_adjusted)[s0],c="k",alpha=.5)
        ax.scatter((fold_changes.values)[s1],-1*np.log10(p_adjusted)[s1],c="k",alpha=.5)
        ax.scatter((fold_changes.values)[s2],-1*np.log10(p_adjusted)[s2],c="C2",alpha=.5)
        ax.scatter((fold_changes.values)[s3],-1*np.log10(p_adjusted)[s3],c="C3",alpha=.5)
    else:
        s0 = ((np.log2(fold_changes.values)<=np.min((fc_cut,-1*fc_cut))) | (np.log2(fold_changes.values)>=np.max((fc_cut,-1*fc_cut)))) & (-1*np.log10(p_adjusted)<=np.min((-1*np.log10(p_cut_1),-1*np.log10(p_cut_2))))
        s1 = (np.log2(fold_changes.values)>np.min((fc_cut,-1*fc_cut))) & (np.log2(fold_changes.values)<np.max((fc_cut,-1*fc_cut)))
        s2 = ((np.log2(fold_changes.values)<=np.min((fc_cut,-1*fc_cut))) | (np.log2(fold_changes.values)>=np.max((fc_cut,-1*fc_cut)))) & (-1*np.log10(p_adjusted)>=np.min((-1*np.log10(p_cut_1),-1*np.log10(p_cut_2)))) & (-1*np.log10(p_adjusted)<np.max((-1*np.log10(p_cut_1),-1*np.log10(p_cut_2))))
        s3 = ((np.log2(fold_changes.values)<=np.min((fc_cut,-1*fc_cut))) | (np.log2(fold_changes.values)>=np.max((fc_cut,-1*fc_cut)))) & (-1*np.log10(p_adjusted)>=np.max((-1*np.log10(p_cut_1),-1*np.log10(p_cut_2))))
        ax.scatter(np.log2(fold_changes.values)[s0],-1*np.log10(p_adjusted)[s0],c="k",alpha=.5)
        ax.scatter(np.log2(fold_changes.values)[s1],-1*np.log10(p_adjusted)[s1],c="k",alpha=.5)
        ax.scatter(np.log2(fold_changes.values)[s2],-1*np.log10(p_adjusted)[s2],c="C2",alpha=.5)
        ax.scatter(np.log2(fold_changes.values)[s3],-1*np.log10(p_adjusted)[s3],c="C3",alpha=.5)
    if label:
        for ind in np.where(s2|s3)[0]:
            if logscaled:
                if fold_changes.values[ind]>0:
                    ax.text((fold_changes.values)[ind]+(np.max(np.abs((fold_changes.values)))*0.02),-1*np.log10(p_adjusted)[ind],genes[ind])
                else:
                    ax.text((fold_changes.values)[ind]-(np.max(np.abs((fold_changes.values)))*0.125),-1*np.log10(p_adjusted)[ind],genes[ind])
            else:
                if np.log2(fold_changes.values)[ind]>0:
                    ax.text(np.log2(fold_changes.values)[ind]+(np.max(np.abs(np.log2(fold_changes.values)))*0.02),-1*np.log10(p_adjusted)[ind],genes[ind])
                else:
                    ax.text(np.log2(fold_changes.values)[ind]-(np.max(np.abs(np.log2(fold_changes.values)))*0.125),-1*np.log10(p_adjusted)[ind],genes[ind])
    if title:
        ax.set_title(title,fontsize=fs)
    ax.plot([fc_cut,fc_cut],[0,1e+4],ls=":",c="k",lw=2)
    ax.plot([-1*fc_cut,-1*fc_cut],[0,1e+4],ls=":",c="k",lw=2)
    ax.plot([-100,100],[-1*np.log10(p_cut_1),-1*np.log10(p_cut_1)],ls=":",c="k",lw=2)
    ax.plot([-100,100],[-1*np.log10(p_cut_2),-1*np.log10(p_cut_2)],ls=":",c="k",lw=2)
    
    if logscaled:
        ax.set_xlim(-1*np.max(np.abs((fold_changes.values))*1.05),np.max(np.abs((fold_changes.values))*1.05))
    else:
        ax.set_xlim(-1*np.max(np.abs(np.log2(fold_changes.values))*1.05),np.max(np.abs(np.log2(fold_changes.values))*1.05))
    ax.set_ylim(0,np.max(-1*np.log10(p_adjusted))*1.05)
    ax.set_xlabel("log2(FC)",fontsize=fs)
    ax.set_ylabel("-log10(adj. p-value)",fontsize=fs)
    ax.tick_params(axis='both', labelsize=fs)
    
    plt.savefig(os.path.join(path,output), bbox_inches='tight')
    
    if write_file:
        Write_DifferentialExpressed(p_adjusted, fold_changes, p_cut=np.max((p_cut_1, p_cut_2)), fc_cut=fc_cut, output=file_out)
    
    return p_adjusted, fold_changes

def Write_DifferentialExpressed(p_values, fold_changes, p_cut=.05, fc_cut=2, output="Differential_Expressed.tsv"):
    """
    Write the output file for differential expressed proteins.

    Parameters
    ----------
    p_values : array 
        Precomputed p-Values.
    fold_changes : pandas.core.series
        Precomputed fold changes.
    p_cut : float, optional
        p-Value cut-off. The default is .05.
    fc_cut : float, optional
        Fold change cut-off. The default is 2.
    output : str, optional
        Name of the output file. The default is "Differential_Expressed.tsv".
    """
    
    z=np.where((p_values<p_cut) & (np.abs(np.log2(fold_changes))>=fc_cut))
    proteins = fold_changes.index.values[z]
    with open(os.path.join(path,output), "w") as file:
        file.write("Protein\t")
        file.write("log2(FC)\t")
        file.write("log10(adj. p-value)\n")
        for protein,index in zip(proteins,z[0]):
            file.write(protein)
            file.write("\t")
            file.write("{0:.3f}".format(np.log2(fold_changes.loc[protein])))
            file.write("\t")
            file.write("{0:.3f}".format(np.log10(p_values[index])))
            if index != z[0][-1]:
                file.write("\n")
    return

def Plot_DataCompleteness(data, selection_dict, group="qc", fdr=0.05, max_nan=0.2, output="Data_Completeness_reduced"):
    """
    Plot the Data completeness of the data set given a fixed FDR. Also preview the data completeness using a NaN filter.

    Parameters
    ----------
    data : pandas.DataFrame
        Data to be analyzed.
    selection_dict : dict
        Dictionary containing groups. Obtained by get_Selection().
    group : str, optional
        Group to be analyzed. Must be key of selection_dict. The default is "qc".
    fdr : float, optional
        False discovery rate for protein filtering (Only used as a label). The default is 0.05.
    max_nan : float, optional
        Maximal fraction of NaN values per protein. The default is 0.2.
    output : str, optional
        Name of the output file. The default is "Data_Completeness_reduced".
    """

    data_binary = np.ones_like(data.loc[selection_dict[group]["FileNames"],:].values)
    data_binary[np.where(np.isnan(data.loc[selection_dict[group]["FileNames"],:].values))]=0
    data_binary = data_binary[:,np.argsort(np.sum(data_binary,axis=0))[::-1]]
    cmap = mpl.colors.ListedColormap(["C0","C1"])
    boundaries = [0,0.5,1]
    norm = mpl.colors.BoundaryNorm(boundaries, cmap.N, clip=True)
    fig,ax=plt.subplots()
    fig.set_size_inches(5,10)
    ax.matshow(data_binary.T,cmap=cmap,norm=norm,aspect="auto")
    ax.set_xticks([])
    ax.set_title("FDR: {:.2f}\n".format(fdr)+"Completeness: {:.3f}".format(np.sum(data_binary)/(np.shape(data_binary)[0]*np.shape(data_binary)[1])))
    ax.set_xlabel("Samples")
    ax.set_ylabel("sorted Genes")    
    legend_elements = [Patch(facecolor='C0',
                             label='not present'),
                       Patch(facecolor='C1',
                             label='present')]
    plt.legend(handles=legend_elements,bbox_to_anchor=(1.2, .5))
    plt.savefig(os.path.join(path,output[:-4]+"_full.png"), bbox_inches='tight')
    
    data_binary_filtered = data_binary[:,np.where(np.sum(data_binary,axis=0)>=(1-max_nan)*np.shape(data_binary)[0])[0]]
    fig,ax=plt.subplots()
    fig.set_size_inches(6,8)
    ax.matshow(data_binary_filtered.T,cmap=cmap,norm=norm,aspect="auto")
    ax.set_xticks([])
    ax.set_title("FDR: {:.2f}\n".format(fdr)+"max NaN: {:.2f}\n".format(max_nan)+"Completeness: {:.3f}".format(np.sum(data_binary_filtered)/(np.shape(data_binary_filtered)[0]*np.shape(data_binary_filtered)[1])))
    ax.set_xlabel("Samples")
    ax.set_ylabel("sorted Genes")
    
    legend_elements = [Patch(facecolor='C0',
                             label='not present'),
                       Patch(facecolor='C1',
                             label='present')]
    plt.legend(handles=legend_elements,bbox_to_anchor=(1.2, .5))    
    plt.savefig(os.path.join(path,output), bbox_inches='tight')
    return

def Plot_BoxPlot_MS(data, selection_dict, group="qc", output="BoxPlot_MS.png"):
    """
    Boxplot protein quantities and number for selected measurements.

    Parameters
    ----------
    data : pandas.DataFrame
        Data to be analyzed.
    selection_dict : dict
        Dictionary containing groups. Obtained by get_Selection().
    group : str, optional
        Group to be analyzed. Must be key of selection_dict. The default is "qc".
    output : TYPE, optional
        Name of the output file. The default is "BoxPlot_MS.png".
    """

    fs=20
    Values = [[np.log2(value) for value in data.loc[index,:].values if not np.isnan(value)] for index in selection_dict[group]["FileNames"]]
    Values_len = [len(values) for values in Values]
    x = np.arange(1,len(selection_dict[group]["Indices"])+1)
    
    fig = plt.figure()
    gs = fig.add_gridspec(ncols=1,nrows=2, height_ratios = [2,8], hspace=.1)
    fig.set_size_inches(7,10)
    
    ax1 = fig.add_subplot(gs[0,0])
    ax1.plot(x,Values_len, c="r",lw=0, marker = "o")
    ax1.plot([0,len(selection_dict[group]["Indices"])+1],[np.max(Values_len),np.max(Values_len)],c="k",ls=":")
    ax1.plot([0,len(selection_dict[group]["Indices"])+1],[np.min(Values_len),np.min(Values_len)],c="k",ls=":")
    ax1.set_xlim(np.min(x)-.5,np.max(x)+.5)
    ax1.set_ylim(np.min(Values_len)*.99, np.max(Values_len)*1.01)
    ax1.set_xticks([])
    ax1.set_ylabel("#Genes", fontsize=fs)
    ax1.set_title("Box and Whisker Plot", fontsize=fs)
    ax1.tick_params(axis='y', labelsize=fs)
    
    ax2 = fig.add_subplot(gs[1,0])
    ax2.boxplot(Values)
    ax2.set_ylabel("log2(Gene.Normalised.Unique)", fontsize=fs)
    ax2.set_xticks(np.arange(1,len(selection_dict[group]["Indices"])+1))
    ax2.set_xticklabels(selection_dict[group]["FileNames"],rotation=90)
    ax2.tick_params(axis='y', labelsize=fs)
    ax2.set_xlim(np.min(x)-.5,np.max(x)+.5)
    plt.savefig(os.path.join(path,output),bbox_inches="tight")
    return

def Filter_NaN(data, selection_dict, group="", max_nan=0.2):
    """
    Filter data for NaN values.

    Parameters
    ----------
    data : pandas.DataFrame
        Data to be analyzed.
    selection_dict : dict
        Dictionary containing groups. Obtained by get_Selection().
    group : str, optional
        Group to be used for filtering. Must be key of selection_dict. The default is "".
    max_nan : float, optional
        Maximal fraction of NaN values per protein. The default is 0.2.

    Returns
    -------
    data_output : pandas.DataFrame
        Filtered data.
    """

    Genes_to_drop_nan = [column for column in data.columns if np.sum(np.isnan(data.loc[selection_dict[group]["FileNames"],column].values))/len(selection_dict[group]["Indices"]) > max_nan]
    return data.drop(columns=Genes_to_drop_nan)

def Plot_CV(data, selection_dict, output="CV.png"):
    """
    Plot the Coefficient of variance for groups in selection_dict.

    Parameters
    ----------
    data : pandas.DataFrame
        Data to be analyzed.
    selection_dict : dict
        Dictionary containing groups. Obtained by get_Selection().
    output : str, optional
        Name of the output file. The default is "CV.png".
    """

    cv_min = 0
    cv_max = 2
    nbins = 50
    y_max = 0
    fs = 20
    
    fig,ax = plt.subplots()
    fig.set_size_inches(8,6)
    for key in selection_dict:
        Values = [np.asarray([value for value in data.loc[selection_dict[key]["FileNames"],column].values if not np.isnan(value)]) for column in data.columns]
        Cv = np.asarray([np.std(values)/np.mean(values) for values in Values])
        hist = np.histogram(Cv, bins=nbins, range = (cv_min,cv_max),density=True)
        
        ax.plot((hist[1][:-1]+hist[1][1:])/2,hist[0],c=selection_dict[key]["Color"],label=selection_dict[key]["Label"])
        ax.fill_between((hist[1][:-1]+hist[1][1:])/2,hist[0],np.zeros(len(hist[0])),alpha=.5, color=selection_dict[key]["Color"])
        y_max = np.nanmax((y_max,np.nanmax(hist[0])))
        
    ax.set_xlim((cv_max-cv_min)/nbins,cv_max)
    ax.set_ylim(0,y_max*1.05)
    ax.set_ylabel("Density",fontsize=fs)
    ax.set_xlabel("CV",fontsize=fs)
    ax.tick_params(axis='x', labelsize=fs)
    ax.tick_params(axis='y', labelsize=fs)
    ax.legend(loc="upper right", fontsize=fs)
    
    plt.savefig(os.path.join(path,output), bbox_inches="tight")
    return

def get_samples_outliers(data, selection_dict, group="", cut_std_samples = 2, cut_std_nan = 2):
    """
    Filter samples for outliers if cut_std_samples*std of total signal and cut_std_nan*std of percentage of NaN are exceeded.

    Parameters
    ----------
    data : pandas.DataFrame
        Data to be analyzed.
    selection_dict : dict
        Dictionary containing groups. Obtained by get_Selection().
    group : str, optional
        Group to be used for filtering. Must be key of selection_dict. The default is "".
    cut_std_samples : float
        n-fold of the standard deviation for the total signal as threshold. The default is 2.
    cut_std_nan : float
        n-fold of the standard deviation for the percentage of NaN as threshold. The default is 2.
    
    Returns
    -------
    outliers : list of str
        Samples classified as outliers.

    """
    
    sum_samples = np.nansum(data.loc[selection_dict[group]["FileNames"],:],axis=1)
    std_samples = np.nanstd(sum_samples)
    mean_samples = np.nanmean(sum_samples)
    num_nan = [np.sum(np.isnan(data.loc[index,:].values))/len(data.loc[index,:].values) for index in selection_dict[group]["FileNames"]]
    std_num_nan = np.nanstd(num_nan)
    mean_num_nan = np.nanmean(num_nan)
    outliers = [selection_dict[group]["FileNames"][out] for out in np.where((np.abs(sum_samples-mean_samples)>(cut_std_samples*std_samples)) | (np.abs(num_nan-mean_num_nan)>(cut_std_nan*std_num_nan)))[0]]
    return outliers

### Add automated selection for position effects (metadata needed!!!)