import torch
from lifelines.utils import concordance_index
import numpy as np
from lifelines.statistics import logrank_test

def R_set(x):
    n_sample = x.size(0)
    matrix_ones = torch.ones(n_sample, n_sample)
    indicator_matrix = torch.tril(matrix_ones)

    return indicator_matrix

def PartialLogLikelihood(hazard_pred, censor, survtime, ties=None):

    n_observed = censor.sum(0)+1
    ytime_indicator = R_set(survtime)
    ytime_indicator = torch.FloatTensor(ytime_indicator).to(survtime.device)
    risk_set_sum = ytime_indicator.mm(torch.exp(hazard_pred).float())
    diff = hazard_pred - torch.log(risk_set_sum)
    sum_diff_in_observed = torch.transpose(diff, 0, 1).mm(censor.unsqueeze(1).float())
    cost = (- (sum_diff_in_observed / n_observed)).reshape((-1,))
    return cost

def calc_concordance_index(logits, fail_indicator, fail_time):
    """
    Compute the concordance-index value.
    Parameters:
        label_true: dict, like {'e': event, 't': time}, Observation and Time in survival analyze.
        y_pred: np.array, predictive proportional risk of network.
    Returns:
        concordance index.
    """

    logits = logits.cpu().detach().numpy()
    fail_indicator = fail_indicator.cpu().detach().numpy()
    fail_time = fail_time.cpu().detach().numpy()

    hr_pred = -logits
    ci = concordance_index(fail_time,
                            hr_pred,
                            fail_indicator)

    # ci = concordance_index_censored(fail_indicator.astype(np.bool_), fail_time, logits)

    return ci

def cox_log_rank(hazardsdata, labels, survtime_all):

    hazardsdata = hazardsdata.cpu().detach().numpy()
    labels = labels.cpu().detach().numpy()
    survtime_all = survtime_all.cpu().detach().numpy()

    median = np.median(hazardsdata)
    hazards_dichotomize = np.zeros([len(hazardsdata)], dtype=int)
    hazards_dichotomize[hazardsdata > median] = 1
    idx = hazards_dichotomize == 0
    T1 = survtime_all[idx]
    T2 = survtime_all[~idx]
    E1 = labels[idx]
    E2 = labels[~idx]
    results = logrank_test(T1, T2, event_observed_A=E1, event_observed_B=E2)
    pvalue_pred = results.p_value
    return(pvalue_pred)

