import os
import itertools
import numpy as np
import networkx as nx
from scipy.spatial import distance
from sklearn.cluster import AgglomerativeClustering

from metricutils import *



def create_connectivity_matrix(number_of_pages: int) -> scipy.sparse._csr.csr_matrix:
    """
    This method creates an irreflexive connectivity matrix from the number of pages in a document.
    :param: number_of_pages (int) The number of pages is used to determine the eventual shape of the
    output matrix.
    
    The function works by creating a 'path' between nodes (1 -> 2 -> 3) and creating the adjacency matrix
    from this. here, the node itself is not considered a neighbour of itself. 
    
    Example: stream with 4 pages:
    >>> create_connectivity_matrix(4).todense()
    matrix([[0, 1, 0, 0],
            [1, 0, 1, 0],
            [0, 1, 0, 1],
            [0, 0, 1, 0]])
    """
    G = nx.Graph() 
    nx.add_path(G, list(range(number_of_pages)))
    return nx.adjacency_matrix(G)    

def groups_to_lengths(cluster_output_list: list):
    consec_blocks = [len(list(group)) for bit, group in itertools.groupby(cluster_output_list)]
    # return binary representation of prediction
    return consec_blocks

def cluster_with_switch(labels, vectors, switch_first = True, mult = 2):
    n_pages = len(labels)
    n_docs = int(sum(labels))
    c_mat = create_connectivity_matrix(n_pages) 

    dist_list = np.array([distance.cosine(vectors[i], vectors[i+1]) for i in range(len(vectors)-1)])
    
    if len(dist_list) > 1:
        dist_list_norm = (dist_list - np.min(dist_list)) / (np.max(dist_list) - np.min(dist_list))
        nth_highest = np.sort(dist_list_norm)[-n_docs]
    else:
        dist_list_norm = dist_list
        nth_highest = dist_list[0]
  
    avg = np.mean(dist_list_norm)
    dist_mat = np.zeros((n_pages,n_pages))
    N = len(dist_list_norm)
    if switch_first:
        dist_list_norm[0] = max(mult*avg-dist_list_norm[0],0)
    for i,dist in enumerate(dist_list_norm):
        if i < N-1:
            if dist > nth_highest:
                next_dist = dist_list_norm[i+1]
                dist_list_norm[i+1] = max(mult*avg-next_dist,0)

    for i,dist in enumerate(dist_list_norm):
        if i < n_pages:
            dist_mat[i,i+1] = dist
            dist_mat[i+1,i] = dist

    cluster = AgglomerativeClustering(n_clusters=n_docs, affinity='precomputed', linkage='average',compute_distances = True, connectivity=c_mat)  
    return dist_list_norm, length_list_to_bin(groups_to_lengths(cluster.fit_predict(dist_mat)))


def select_topk(pred_list: list, k: int):
    pred_labels = np.zeros(pred_list.shape[0])
    inds = np.argpartition(pred_list, -k)[-k:]
    pred_labels[inds] = 1
    return pred_labels


def get_base_metrics(gold_standard: list, prediction: list):
    fn = np.logical_and(gold_standard == 1, prediction == 0).sum()
    fp = np.logical_and(gold_standard == 0, prediction == 1).sum()
    tp = np.logical_and(gold_standard == 1, prediction == 1).sum()
    tn = np.logical_and(gold_standard == 0, prediction == 0).sum()
    
    return {'TP': tp, 'TN': tn, 'FP': fp, 'FN': fn}

def vec_comparison(vector, vector_list):
    if len(vector_list) > 0:
        tot_dist = 0
        for vec in vector_list:
            if (vector == vec).all():
                continue
            tot_dist += distance.cosine(vector, vec)
        return tot_dist / len(vector_list)
    else:
        return 0

# Fix the spaces on this function as well.
def distance_against_eachother(vector_list, gold_standard):
  gs_len = bin_to_length_list(gold_standard)
  longer_than_two = len([x for x in gs_len if x > 1])
  index = 0
  results= {'same_document':0,'different_document':0}
  for doc in gs_len:
    this_doc = index+doc
    if doc < 2:
      index = this_doc
      continue
    same_doc = 0
    diff_doc = 0
    vectors_before = vector_list[:index]
    current_vectors = vector_list[index:this_doc]
    vectors_after = vector_list[this_doc:]
    for i, vec in enumerate(current_vectors):
      same = vec_comparison(vec,current_vectors[i+1:])
      bef = vec_comparison(vec,vectors_before)
      aft = vec_comparison(vec,vectors_after)
      same_doc += same
      diff_doc += (bef+aft) /2
    
    index = this_doc
    results['same_document'] += same_doc / (doc - 1)
    results['different_document'] += diff_doc / doc

    index = this_doc
  results['same_document'] = (results['same_document'] / longer_than_two)
  results['different_document'] = (results['different_document'] / longer_than_two)

  return results
  