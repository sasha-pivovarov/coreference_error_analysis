import itertools
import random
from collections import Counter
from statistics import mean

from scipy.stats import ttest_1samp

def pair_dicts(dict1, dict2):
    pair1 = []
    pair2 = []
    for key in dict1.keys():
        pair1.append(dict1[key])
        pair2.append(dict2.get(key, 1))
    return pair1, pair2


def normalize_dict(d):
    factor = 1.0 / sum(d.values())
    normalised_d = {k: v * factor for k, v in d.items()}
    return normalised_d


def pair_norm(dict1, dict2):
    normalized_d1 = normalize_dict(dict1)
    normalized_d2 = normalize_dict(dict2)
    res = {}
    for key in dict1.keys():
        res[key] = (normalized_d1[key] / normalized_d2.get(key, 1), dict1[key], dict2.get(key, 1))

    return res


normalize = lambda x: [float(i)/sum(x) for i in x]


def bootstrap_report(all_obs, test_counter, test_size, n_resamples, count_bound:int=10, pval_bound:float=0.05):
    all_counter = dict(Counter(all_obs))
    normed_results = pair_norm(all_counter, test_counter)
    subsamples = [dict(Counter(random.sample(all_obs, test_size))) for _ in range(n_resamples)]
    unified_subsamples = {}
    results = {}
    report_significant = "SIGNIFICANT DIFFERENCE: \n\n"
    report_insignificant = "NOT SIGNIFICANT DUE TO LOW COUNT OR HIGH TTEST PVALUE: \n\n"
    for tag in set(itertools.chain.from_iterable([s.keys() for s in subsamples])):
        unified_subsamples[tag] = [d.get(tag, 1) for d in subsamples]
    for tag, value in unified_subsamples.items():
        print(value)
        test_val = test_counter.get(tag, 1)
        theta = ttest_1samp(value, test_val)
        normed = normed_results.get(tag, (1.0, 1, 1))
        results[tag] = theta.pvalue, normed
        if theta.pvalue < pval_bound and all_counter.get(tag, 1) > count_bound:
            report_significant += f"Tag {tag}, p-value {theta.pvalue}, normed count ratio: {normed[0]}, absolute count in general sample: {all_counter.get(tag, 1)}, absolute count in test sample: {test_counter.get(tag, 1)}\n"
        else:
            report_insignificant += f"Tag {tag}, p-value {theta.pvalue}, normed count ratio: {normed[0]}, absolute count in general sample: {all_counter.get(tag, 1)}, absolute count in test sample: {test_counter.get(tag, 1)}\n"

    print(f"{report_significant} \n\n ================================================================= \n\n {report_insignificant}")
    return results