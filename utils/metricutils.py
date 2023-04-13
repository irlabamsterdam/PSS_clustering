import scipy
import json
import numpy as np
import pandas as pd
from typing import Dict, Union, List
from collections import defaultdict
from sklearn.metrics import f1_score, precision_score, recall_score, \
    accuracy_score

"""
Example: t = [1, 3], h = [3, 1]

"""

def make_index(split):
    l= sum(split)
    pages= list(np.arange(l))
    out = defaultdict(set)
    for block_length in split:
        block= pages[:block_length]
        pages= pages[block_length:]
        for page in block:
            out[page]= set(block)
    return out



def align(t: List[int], h: List[int], kind: str="and"):
    """

    params:
        t: list of integers containing the cluster sizes of the gold standard.
        Each element represents the size of a cluster. for example, [3, 5] represent
        a gold standard with two clusters, the first of size 3, the second of size 5
        h: list of integers containing the cluster sizes of the predictions, see
        above for an example.
        kind: either "plus" or "and"" specifying whether to use the classic definition of 
        PQ, or the altered defined presented in (maarten_paper_link)
    """

    # valid number in the input
    assert (min(t) > 0) and (min(h) > 0), "a document with length 0 was encountered, maybe you are using the binary format?"
    # both sets must have the same number of elements
    assert sum(t) == sum(h)
    # valid kind argument
    assert kind in ["plus", "and"]

    '''A True Positive is a pair h_block, t_block with an IoU>.5.
    This function returns the sum of all IoUs(h_block,t_block) for these bvlocks in t and h,
    and the sets of TPs, FPs and FNs, in essence aligning the sets.'''
    def IoU(S,T):
        '''Jaccard similarity between sets S and T'''
        return len(S&T)/len(S|T) 
    def get_docs(t):
        '''Get the set of documents (where each document is a set of pagenumbers)'''
        return {frozenset(S) for S in make_index(t).values()}
    def find_match(S,Candidates, kind):
        '''Finds, if it exists,  the unique T in Candidates such that IoU(S,T) >.5'''
        if kind == 'plus': 
            match =  [T for T in Candidates if IoU(S,T) >.5]
        elif kind == 'and': 
            match =  [T for T in Candidates if len(S&T) > len(S-T) and len(S&T) > len(T-S)]
        
        return match[0] if match else None
    
    t,h= get_docs(t), get_docs(h) # switch to set of docs representation

    TP = set()
    for S in h:
        # loop through all elements in the prediction
        # add t,h pairs if a match is found
        match = find_match(S, t, kind)
        if match:
            TP.add((S, match))

    TP = set(TP)
    
    FP = h-{S for (S,_) in TP} 
    FN= t - {T for (_,T) in TP}
    
    return sorted([IoU(S,T) for (S,T) in TP]), TP, FP, FN


"""
For recognition quality, the formulas are as follows
Precision: |TP| / (|TP|+|FP|)
Recall: |TP| / (|TP|+|FN|)
F1: |TP| / (|TP|+0.5*|FN| + 0.5*|FP|)

For segmentation quality, the formula is the sum over the IoU values of each TP pair,
divided by the total number of TPs, in essence a weight to know how good it was matched.
Segmentation Quality = (sum of IoU(gold_standard_clust, prediction_clust) for all pairs in TP) / (|TP|)

Using the previous definitions, PQ is defined as the product of SQ and RQ.
"""

def PQ_metrics(t, h, kind='and'):
    """
    This function return the PQ metric scores, both the complete PQ metric
    as well as both its parts, recognition quality and segmentation quality.

    params:
        t: list of integers containing the cluster sizes of the gold standard.
        Each element represents the size of a cluster. for example, [3, 5] represent
        a gold standard with two clusters, the first of size 3, the second of size 5
        h: list of integers containing the cluster sizes of the predictions, see
        above for an example.
        kind: either "plus" or "and"" specifying whether to use the classic definition of 
        PQ, or the altered defined presented in (maarten_paper_link)
        
        NOTE: when there are no TPs, then SQ is undefined, but all metrics are equal to 0.
        Note that in this edge case, wP is not equal to SQ * P, and the same for other metrics.

    """
    IOU_SCORES, TP, FP, FN = align(t, h, kind)

    # Segmentation Score
    SQ = np.nan if len(TP) == 0 else sum(IOU_SCORES) / len(TP)

    # Recogntion quality
    PRECISION = len(TP) / (len(TP) + len(FP))
    RECALL = len(TP) / (len(TP) + len(FN))
    F1 = (len(TP) / (len(TP) + 0.5*len(FP) + 0.5*len(FN)))
    # wF1 is kirilov's PQ 
    # F1 is kirilov's RQ
    def nan_multiply(a,b):
        '''if  np.isnan(a), then a*b = 0'''
        return 0 if np.isnan(a) else a*b
    return {'wP': nan_multiply(SQ,PRECISION),
            'wR': nan_multiply(SQ,RECALL),
            'wF1': nan_multiply(SQ,F1),
           'SQ': SQ, 'P': PRECISION, 'R': RECALL, 'F1': F1, 'iou_scores': IOU_SCORES}

    

import PIL.Image
PIL.Image.MAX_IMAGE_PIXELS = None


def read_json(filename: str) -> dict:
    """
    This function provides a convenience function for reading in a file
    that is in the json standard.
    :param filename: string specifying the path to the json file
    :return: JSON-style dict with the contents from the file
    """
    with open(filename, 'r') as json_file:
        contents = json.load(json_file)
    
    for key, value in contents.items():
        if isinstance(value, list):
            contents[key][0] = 1
        else:
            contents[key] = [1]
    return contents


def get_ground_truth_from_dataframe(dataframe: pd.DataFrame, col: str) -> Dict[str, list]:
    """
    This function takes as input the test dataframe, and return a dictionary
    with stream names as keys and the gold standard streams in
    binary vector format.

    """
    out = {}
    for doc_id, content in dataframe.groupby('name'):
        out[doc_id] = [int(item) for item in content[col].tolist()]
    return out


def length_list_to_bin(list_of_lengths: Union[list, np.array]) -> Union[list, np.array]:
    """
    :param list_of_lengths:  containing the lengths of the individual documents
    in a stream as integers.
    :return: list representing the stream in binary format.
    """

    if not all([item > 0 for item in list_of_lengths]):
        raise ValueError

    # Set up the output array
    out = np.zeros(shape=(sum(list_of_lengths)))

    # First document is always a boundary
    out[0] = 1

    # if only one document return the current representation
    if len(list_of_lengths) == 1:
        if type(list_of_lengths) == list:
            return out.tolist()
        else:
            return out

    # Boundaries are at the cumulative sums of the number of pages
    # >>> doc_list = [2, 4, 3, 1]
    # >>> np.cumsum(doc_list) -> [2 6 9]

    # [:-1] because last document has boundary at end of array
    out[np.cumsum(list_of_lengths[:-1])] = 1
    if type(list_of_lengths) == list:
        return out.tolist()
    else:
        return out


def bin_to_length_list(binary_vector: Union[list, np.array]) -> Union[list, np.array]:
    """
    :param binary_vector: np array containing the stream of pages
    in the binary format.
    :return: A numpy array representing the stream as a list of
    document lengths.
    """

    # make sure the vector only contains 1s and zeros
    if not all([item in [0, 1] for item in binary_vector]):
        raise ValueError

    return_type = type(binary_vector)

    if type(binary_vector) == list:
        binary_vector = np.array(binary_vector)

    # We retrieve the indices of the ones with np.nonzero
    bounds = binary_vector.nonzero()[0]

    # We add the length of the array so that it works
    # with ediff1d, as this get the differences between
    # consecutive elements, and otherwise we would miss
    # the list document.
    bounds = np.append(bounds, len(binary_vector))

    # get consecutive indices
    out = np.ediff1d(bounds)

    if return_type == list:
        return out.tolist()
    else:
        return out


def window_diff(gold: np.array, prediction: np.array) -> float:

    assert len(gold) == len(prediction)

    gold[0] = 1
    prediction[0] = 1

    k = int(bin_to_length_list(gold).mean()*0.5)
    # small check, in case of a singleton cluster, k will be too large
    if k > len(gold):
        k = len(gold)

    string_gold = ''.join(str(item) for item in gold.astype(int).tolist())
    string_prediction = ''.join(str(item) for item in prediction.astype(int).tolist())

    return windowdiff(string_gold, string_prediction, k=k)


def f1(gold: np.array, prediction: np.array) -> float:
    return f1_score(gold, prediction)


def precision(gold: np.array, prediction: np.array) -> float:
    return precision_score(gold, prediction)


def recall(gold: np.array, prediction: np.array) -> float:
    return recall_score(gold, prediction)


def calculate_metrics_one_stream(gold_vec, prediction_vec):
    out = {}

    gold_vec = np.array(gold_vec)
    prediction_vec = np.array(prediction_vec)

    prediction_vec[0] = 1
    scores = {
              'Page': f1(gold_vec, prediction_vec),
              'Doc': PQ_metrics(bin_to_length_list(gold_vec), bin_to_length_list(prediction_vec), kind='plus')['wP']}

    scores_precision = {
                        'Page': precision(gold_vec, prediction_vec),
                        'Doc': PQ_metrics(bin_to_length_list(gold_vec), bin_to_length_list(prediction_vec), kind='plus')['wR']}

    scores_recall = {
                     'Page': recall(gold_vec, prediction_vec),
                     'Doc': PQ_metrics(bin_to_length_list(gold_vec), bin_to_length_list(prediction_vec), kind='plus')['wF1']}

    out['precision'] = scores_precision
    out['recall'] = scores_recall
    out['F1'] = scores

    return out


def calculate_scores_df(gold_standard_dict, prediction_dict):
    all_scores = defaultdict(dict)
    for key in gold_standard_dict.keys():
        metric_scores = calculate_metrics_one_stream(gold_standard_dict[key],
                                                     prediction_dict[key])
        for key_m in metric_scores.keys():
            all_scores[key_m][key] = metric_scores[key_m]
    return {key: pd.DataFrame(val) for key, val in all_scores.items()}


def calculate_mean_scores(gold_standard_dict, prediction_dict,
                          show_confidence_bounds=True):
    scores_df = {key: val.T.mean().round(2) for key, val in
                 calculate_scores_df(gold_standard_dict,
                                     prediction_dict).items()}
    scores_combined = pd.DataFrame(scores_df)
    test_scores = scores_combined

    confidence = 0.95

    # total number of documents is the number of ones in the binary array
    n = sum([np.sum(item) for item in prediction_dict.values()])

    z_value = scipy.stats.norm.ppf((1 + confidence) / 2.0)
    ci_length = z_value * np.sqrt((test_scores * (1 - test_scores)) / n)

    ci_lower = (test_scores - ci_length).round(2)
    ci_upper = (test_scores + ci_length).round(2)

    precision_ci = ci_lower['precision'].astype(str) + '-' + ci_upper[
        'precision'].astype(str)
    recall_ci = ci_lower['recall'].astype(str) + '-' + ci_upper[
        'recall'].astype(str)
    f1_ci = ci_lower['F1'].astype(str) + '-' + ci_upper['F1'].astype(str)

    out = pd.DataFrame(scores_df)
    out = out.rename({0: 'value'}, axis=1)
    out['support'] = sum(
        [np.sum(item).astype(int) for item in gold_standard_dict.values()])
    if show_confidence_bounds:
        out['CI Precision'] = precision_ci
        out['CI Recall'] = recall_ci
        out['CI F1'] = f1_ci

    return out


def evaluation_report(gold_standard_json, prediction_json, round_num=2,
                      show_confidence_bounds=True):

    return calculate_mean_scores(gold_standard_json, prediction_json,
                                show_confidence_bounds=show_confidence_bounds).round(round_num)