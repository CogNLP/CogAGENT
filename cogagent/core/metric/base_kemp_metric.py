import math
from cogagent.core.metric.base_metric import BaseMetric
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
from sklearn.metrics import accuracy_score
import collections

class BaseKempMetric(BaseMetric):
    def __init__(self, mode, default_metric_name=None):
        super().__init__()
        if mode not in ["binary", "multi"]:
            raise ValueError("Please choose mode in binary or multi")
        self.mode = mode
        self.label_list = [[],[]]
        self.pre_list = [[],[]]
        # self.label_list = list()
        # self.pre_list = list()
        self.default_metric_name = default_metric_name
        if default_metric_name is None:
            self.default_metric_name = "Acc" if mode == "binary" else "Acc"
        else:
            self.default_metric_name = default_metric_name

    def evaluate(self, pred, label):
        
        self.pre_list[0] = self.pre_list[0] + pred[0] 
        self.label_list[0] = self.label_list[0] + label[0]
        self.pre_list[1] = self.pre_list[1] + pred[1] 
        self.label_list[1] = self.label_list[1] + label[1]
        # .cpu().numpy().tolist()

    def get_metric(self, reset=True):
        evaluate_result = {}
        res = self.pre_list[0]
        gdn = self.label_list[0]
        if self.mode == "binary":
            Acc = accuracy_score(self.label_list[1], self.pre_list[1])
            ma_dist1, ma_dist2, mi_dist1, mi_dist2, avg_len = get_dist(res)
            ma_bleu, ma_bleu1, ma_bleu2, ma_bleu3, ma_bleu4, \
            mi_bleu, mi_bleu1, mi_bleu2, mi_bleu3, mi_bleu4 = get_bleu(res,gdn)
            evaluate_result = {"ma_dist1": ma_dist1, 
                               "ma_dist2": ma_dist2, 
                            "mi_dist1": mi_dist1,
                            "mi_dist2": mi_dist2, 
                            "ma_bleu": ma_bleu,
                            "ma_bleu1": ma_bleu1,
                            "ma_bleu2": ma_bleu2,
                            "ma_bleu3": ma_bleu3,
                            "ma_bleu4": ma_bleu4,
                            "mi_bleu": mi_bleu, 
                            "mi_bleu1": mi_bleu1,
                            "mi_bleu2": mi_bleu2,
                            "mi_bleu3": mi_bleu3,
                            "mi_bleu4": mi_bleu4,
                            #    "ppl": ppl,
                            #    "bce": bce,
                                "Acc": Acc,
                            }
        if reset:
            self.label_list = [[],[]]
            self.pre_list = [[],[]]
        return evaluate_result

def get_bleu(res, gdn):
    assert len(res) == len(gdn)

    ma_bleu = 0.
    ma_bleu1 = 0.
    ma_bleu2 = 0.
    ma_bleu3 = 0.
    ma_bleu4 = 0.
    ref_lst = []
    hyp_lst = []
    for q, r in enumerate(res):
    # for q, r in res.items():
        references = gdn[q]
        hypothesis = r
        ref_lst.append(references)
        hyp_lst.append(hypothesis)
        bleu, precisions, _, _, _, _ = compute_bleu([references], [hypothesis], smooth=False)
        ma_bleu += bleu
        ma_bleu1 += precisions[0]
        ma_bleu2 += precisions[1]
        ma_bleu3 += precisions[2]
        ma_bleu4 += precisions[3]
    n = len(res)
    ma_bleu /= n
    ma_bleu1 /= n
    ma_bleu2 /= n
    ma_bleu3 /= n
    ma_bleu4 /= n

    mi_bleu, precisions, _, _, _, _ = compute_bleu(ref_lst, hyp_lst, smooth=False)
    mi_bleu1, mi_bleu2, mi_bleu3, mi_bleu4 = precisions[0], precisions[1], precisions[2], precisions[3]
    return ma_bleu, ma_bleu1, ma_bleu2, ma_bleu3, ma_bleu4, \
        mi_bleu, mi_bleu1, mi_bleu2, mi_bleu3, mi_bleu4
        
def compute_bleu(reference_corpus, translation_corpus, max_order=4, smooth=False):
    """Computes BLEU score of translated segments against one or more references.

    Args:
    reference_corpus: list of lists of references for each translation. Each
        reference should be tokenized into a list of tokens.
    translation_corpus: list of translations to score. Each translation
        should be tokenized into a list of tokens.
    max_order: Maximum n-gram order to use when computing BLEU score.
    smooth: Whether or not to apply Lin et al. 2004 smoothing.

    Returns:
    3-Tuple with the BLEU score, n-gram precisions, geometric mean of n-gram
    precisions and brevity penalty.
    """
    matches_by_order = [0] * max_order
    possible_matches_by_order = [0] * max_order
    reference_length = 0
    translation_length = 0
    for (references, translation) in zip(reference_corpus,
                                        translation_corpus):
        reference_length += min(len(r) for r in references)
        translation_length += len(translation)

        merged_ref_ngram_counts = collections.Counter()
        for reference in references:
            merged_ref_ngram_counts |= _get_ngrams(reference, max_order)
        translation_ngram_counts = _get_ngrams(translation, max_order)
        overlap = translation_ngram_counts & merged_ref_ngram_counts
        for ngram in overlap:
            matches_by_order[len(ngram) - 1] += overlap[ngram]
        for order in range(1, max_order + 1):
            possible_matches = len(translation) - order + 1
            if possible_matches > 0:
                possible_matches_by_order[order - 1] += possible_matches

    precisions = [0] * max_order
    for i in range(0, max_order):
        if smooth:
            precisions[i] = ((matches_by_order[i] + 1.) /
                            (possible_matches_by_order[i] + 1.))
        else:
            if possible_matches_by_order[i] > 0:
                precisions[i] = (float(matches_by_order[i]) /
                                possible_matches_by_order[i])
            else:
                precisions[i] = 0.0

    if min(precisions) > 0:
        p_log_sum = sum((1. / max_order) * math.log(p) for p in precisions)
        geo_mean = math.exp(p_log_sum)
    else:
        geo_mean = 0

    ratio = float(translation_length) / reference_length

    if ratio > 1.0:
        bp = 1.
    else:
        bp = math.exp(1 - 1. / (ratio + 1e-16))

    bleu = geo_mean * bp

    return (bleu, precisions, bp, ratio, translation_length, reference_length)

def get_dist(res):
    unigrams = []
    bigrams = []
    avg_len = 0.
    ma_dist1, ma_dist2 = 0., 0.
    
    for q, r in enumerate(res):
    # for q, r in res.items():
        ugs = r
        bgs = []
        i = 0
        while i < len(ugs) - 1:
            bgs.append(ugs[i] + ugs[i + 1])
            i += 1
        unigrams += ugs
        bigrams += bgs
        ma_dist1 += len(set(ugs)) / (float)(len(ugs) + 1e-16)
        ma_dist2 += len(set(bgs)) / (float)(len(bgs) + 1e-16)
        avg_len += len(ugs)
    n = len(res)
    ma_dist1 /= n
    ma_dist2 /= n
    mi_dist1 = len(set(unigrams)) / (float)(len(unigrams))
    mi_dist2 = len(set(bigrams)) / (float)(len(bigrams))
    avg_len /= n
    return ma_dist1, ma_dist2, mi_dist1, mi_dist2, avg_len

def _get_ngrams(segment, max_order):
        """Extracts all n-grams upto a given maximum order from an input segment.

        Args:
        segment: text segment from which n-grams will be extracted.
        max_order: maximum length in tokens of the n-grams returned by this
            methods.

        Returns:
        The Counter containing all n-grams upto max_order in segment
        with a count of how many times each n-gram occurred.
        """
        ngram_counts = collections.Counter()
        for order in range(1, max_order + 1):
            for i in range(0, len(segment) - order + 1):
                ngram = tuple(segment[i:i + order])
                ngram_counts[ngram] += 1
        return ngram_counts
