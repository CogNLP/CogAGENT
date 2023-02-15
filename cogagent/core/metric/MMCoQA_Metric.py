import os
import pytrec_eval
from cogagent.core.metric.base_metric import BaseMetric
from cogagent.utils.MMCoQA_utils import (LazyQuacDatasetGlobal, RawResult,
                                         write_predictions, write_final_predictions,
                                         get_retrieval_metrics, gen_reader_features, quac_eval)



class MMCoQAMetric(BaseMetric):
    def __init__(self, default_metric_name=None, evaluator=None):
        super().__init__()
        self.pytrec_eval_evaluator = evaluator
        self.default_metric_name = default_metric_name
        self.examples = {}
        self.features = {}
        self.retriever_run_dict = {}
        self.all_results = []
        self.prefix = '4000'
        self.predict_dir = '/data/yangcheng/CogAgent/datapath/MMCoQA_data/experimental_result/predictions/'
        self.version_2_with_negative = False
        self.n_best_size = 20
        self.max_answer_length = 40
        self.do_lower_case = True
        self.verbose_logging = False
        self.null_score_diff_threshold = 0.0
        self.use_rerank_prob = True
        self.use_retriever_prob = True
        self.orig_eval_file = '/data/yangcheng/CogAgent/datapath/MMCoQA_data/data/MMCoQA_dev.txt'

        if default_metric_name is None:
            self.default_metric_name = "f1"
        else:
            self.default_metric_name = default_metric_name

    def evaluate(self, all_results, examples, features, retriever_run_dict):
        # for i in retriever_run_dict:
        #     self.retriever_run_dict.update(retriever_run_dict[i])
        self.retriever_run_dict.update(retriever_run_dict)
        # for i in range(len(retriever_run_dict)):
        #     self.retriever_run_dict.update(retriever_run_dict[i])
        for i in range(len(all_results)):
            self.all_results.append(all_results[i])
        # self.all_results.append(all_results)
        self.examples.update(examples)
        self.features.update(features)

    def get_metric(self, reset=True):
        output_prediction_file = os.path.join(
            self.predict_dir, "instance_predictions_{}.json".format(self.prefix))
        output_nbest_file = os.path.join(
            self.predict_dir, "instance_nbest_predictions_{}.json".format(self.prefix))
        output_final_prediction_file = os.path.join(
            self.predict_dir, "final_predictions_{}.json".format(self.prefix))
        if self.version_2_with_negative:
            output_null_log_odds_file = os.path.join(
                self.predict_dir, "instance_null_odds_{}.json".format(self.prefix))
        else:
            output_null_log_odds_file = None

        all_predictions = write_predictions(self.examples, self.features, self.all_results, self.n_best_size,
                                            self.max_answer_length, self.do_lower_case, output_prediction_file,
                                            output_nbest_file, output_null_log_odds_file, self.verbose_logging,
                                            self.version_2_with_negative, self.null_score_diff_threshold)
        final_texts = write_final_predictions(all_predictions, output_final_prediction_file,
                                use_rerank_prob=self.use_rerank_prob,
                                use_retriever_prob=self.use_retriever_prob)
        eval_metrics = quac_eval(
            self.orig_eval_file, output_final_prediction_file)

        rerank_metrics = get_retrieval_metrics(
            self.pytrec_eval_evaluator, all_predictions, eval_retriever_probs=True)

        rerank_metrics = get_retrieval_metrics(
            self.pytrec_eval_evaluator, all_predictions, eval_retriever_probs=True, retriever_run_dict=self.retriever_run_dict)

        eval_metrics.update(rerank_metrics)

        evaluate_result = {
            "f1": eval_metrics['f1'],
            # "human_f1": eval_metrics['human_f1'],
            "EM": eval_metrics['EM'],
            "retriever_ndcg": eval_metrics['retriever_ndcg'],
            "retriever_recall": eval_metrics['retriever_recall'],

        }
        if reset:
            self.examples = {}
            self.features = {}
            self.all_results = []
            self.retriever_run_dict = {}

        return evaluate_result

