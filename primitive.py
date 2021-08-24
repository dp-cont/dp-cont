import numpy as np

from scipy import special
from numpy import linalg as LA


def laplace(beta, data):
    output = data + np.random.laplace(0, beta, len(data))
    return output


def smooth_ell(p_rank, b, max_ell, data):
    m = len(data)
    data = np.append([0] * m, data)
    data = np.append(data, [max_ell] * m)
    p_rank += m
    local_ss = np.zeros(m + 2)
    for k in range(m + 2):
        max_local_s = (data[p_rank: p_rank + k + 1] - data[p_rank - k - 1: p_rank]).max()
        # for t in range(k + 2):
        #     v1 = data[P + t]
        #     v2 = data[P + t - k - 1]
        #     if v1 - v2 > max_local_s:
        #         max_local_s = v1 - v2
        local_ss[k] = max_local_s * np.exp(-b * k)

    return max(local_ss)


# noisy max
def nm(q_ans, eps, sensitivity=1, monotonic=True):
    coeff = eps / sensitivity if monotonic else eps / 2 / sensitivity
    noisy_ans = laplace(1 / coeff, q_ans)
    return np.argmax(noisy_ans)


'''some LDP primitives'''


# stochastic rounding (Duchi et al.)
def sr(ori_samples, l, h, eps):
    output = (np.exp(eps) + 1) / (np.exp(eps) - 1)
    sample_size = len(ori_samples)
    samples = (ori_samples - l) * 2 / (h - l) - 1
    probs = (np.exp(eps) - 1) / (2 * np.exp(eps) + 2) * samples + 1 / 2
    ns = np.zeros(sample_size)
    tmps = np.random.binomial(1, probs)
    ns[tmps == 1] = output
    ns[tmps == 0] = - output
    # mean = (np.mean(ns) + 1) / 2 * (h - l) + l
    return (ns + 1) / 2 * (h - l) + l


# piecewise mechanism (Wang et al.)
def pm(ori_samples, l, h, eps):
    c = (np.exp(eps / 2) + 1) / (np.exp(eps / 2) - 1)
    p = (np.exp(eps) - np.exp(eps / 2)) / (2 * np.exp(eps / 2) + 2)
    q = p / np.exp(eps)
    sample_size = len(ori_samples)
    samples = (ori_samples - l) * 2 / (h - l) - 1
    ns = np.zeros(sample_size)
    y = np.random.uniform(0, 1, sample_size)
    for i, sample in enumerate(samples):
        l = (c + 1) / 2 * sample - (c - 1) / 2
        r = l + c - 1
        if y[i] < (l + c) * q:
            ns[i] = (y[i] / q - c)
        elif y[i] < (c - 1) * p + (l + c) * q:
            ns[i] = ((y[i] - (l + c) * q) / p + l)
        else:
            ns[i] = ((y[i] - (l + c) * q - (c - 1) * p) / q + r)

    return (ns + 1) / 2 * (h - l) + l


def hm(ori_samples, l, h, eps):
    if eps < 0.61:
        alpha = 0
    else:
        alpha = 1 - np.exp(-eps / 2)
    tmps = np.random.binomial(1, alpha, len(ori_samples))
    ns = np.zeros_like(ori_samples)
    ns[tmps == 0] = sr(ori_samples[tmps == 0], l, h, eps)
    ns[tmps == 1] = pm(ori_samples[tmps == 1], l, h, eps)
    return ns


# square wave (Li et al.)
def sw(ori_samples, l, h, eps, randomized_bins=1024, domain_bins=1024, smoothing=False):
    ee = np.exp(eps)
    w = ((eps * ee) - ee + 1) / (2 * ee * (ee - 1 - eps)) * 2
    p = ee / (w * ee + 1)
    q = 1 / (w * ee + 1)

    samples = (ori_samples - l) / (h - l)
    randoms = np.random.uniform(0, 1, len(samples))

    noisy_samples = np.zeros_like(samples)

    # report
    index = randoms <= (q * samples)
    noisy_samples[index] = randoms[index] / q - w / 2
    index = randoms > (q * samples)
    noisy_samples[index] = (randoms[index] - q * samples[index]) / p + samples[index] - w / 2
    index = randoms > q * samples + p * w
    noisy_samples[index] = (randoms[index] - q * samples[index] - p * w) / q + samples[index] + w / 2

    # report matrix
    m = randomized_bins
    n = domain_bins
    m_cell = (1 + w) / m
    n_cell = 1 / n

    transform = np.ones((m, n)) * q * m_cell
    for i in range(n):
        left_most_v = (i * n_cell)
        right_most_v = ((i + 1) * n_cell)

        ll_bound = int(left_most_v / m_cell)
        lr_bound = int((left_most_v + w) / m_cell)
        rl_bound = int(right_most_v / m_cell)
        rr_bound = int((right_most_v + w) / m_cell)

        ll_v = left_most_v - w / 2
        rl_v = right_most_v - w / 2
        l_p = ((ll_bound + 1) * m_cell - w / 2 - ll_v) * (p - q) + q * m_cell
        r_p = ((rl_bound + 1) * m_cell - w / 2 - rl_v) * (p - q) + q * m_cell
        if rl_bound > ll_bound:
            transform[ll_bound, i] = (l_p - q * m_cell) * ((ll_bound + 1) * m_cell - w / 2 - ll_v) / n_cell * 0.5 + q * m_cell
            transform[ll_bound + 1, i] = p * m_cell - (p * m_cell - r_p) * (rl_v - ((ll_bound + 1) * m_cell - w / 2)) / n_cell * 0.5
        else:
            transform[ll_bound, i] = (l_p + r_p) / 2
            transform[ll_bound + 1, i] = p * m_cell

        lr_v = left_most_v + w / 2
        rr_v = right_most_v + w / 2
        r_p = (rr_v - (rr_bound * m_cell - w / 2)) * (p - q) + q * m_cell
        l_p = (lr_v - (lr_bound * m_cell - w / 2)) * (p - q) + q * m_cell
        if rr_bound > lr_bound:
            if rr_bound < m:
                transform[rr_bound, i] = (r_p - q * m_cell) * (rr_v - (rr_bound * m_cell - w / 2)) / n_cell * 0.5 + q * m_cell

            transform[rr_bound - 1, i] = p * m_cell - (p * m_cell - l_p) * ((rr_bound * m_cell - w / 2) - lr_v) / n_cell * 0.5

        else:
            transform[rr_bound, i] = (l_p + r_p) / 2
            transform[rr_bound - 1, i] = p * m_cell

        if rr_bound - 1 > ll_bound + 2:
            transform[ll_bound + 2: rr_bound - 1, i] = p * m_cell

    max_iteration = 10000
    loglikelihood_threshold = 1e-3
    ns_hist, _ = np.histogram(noisy_samples, bins=randomized_bins, range=(-w / 2, 1 + w / 2))

    if smoothing:
        return EMS(n, ns_hist, transform, max_iteration, loglikelihood_threshold) * len(ori_samples)
    else:
        return EM(n, ns_hist, transform, max_iteration, loglikelihood_threshold) * len(ori_samples)


def EMS(n, ns_hist, transform, max_iteration, loglikelihood_threshold):
    # smoothing matrix
    smoothing_factor = 2
    binomial_tmp = [special.binom(smoothing_factor, k) for k in range(smoothing_factor + 1)]
    smoothing_matrix = np.zeros((n, n))
    central_idx = int(len(binomial_tmp) / 2)
    for i in range(int(smoothing_factor / 2)):
        smoothing_matrix[i, : central_idx + i + 1] = binomial_tmp[central_idx - i:]
    for i in range(int(smoothing_factor / 2), n - int(smoothing_factor / 2)):
        smoothing_matrix[i, i - central_idx: i + central_idx + 1] = binomial_tmp
    for i in range(n - int(smoothing_factor / 2), n):
        remain = n - i - 1
        smoothing_matrix[i, i - central_idx + 1:] = binomial_tmp[: central_idx + remain]
    row_sum = np.sum(smoothing_matrix, axis=1)
    smoothing_matrix = (smoothing_matrix.T / row_sum).T

    # EMS
    theta = np.ones(n) / float(n)
    theta_old = np.zeros(n)
    r = 0
    sample_size = sum(ns_hist)
    old_logliklihood = 0

    while LA.norm(theta_old - theta, ord=1) > 1 / sample_size and r < max_iteration:
        theta_old = np.copy(theta)
        X_condition = np.matmul(transform, theta_old)

        TMP = transform.T / X_condition

        P = np.copy(np.matmul(TMP, ns_hist))
        P = P * theta_old

        theta = np.copy(P / sum(P))

        # Smoothing step
        theta = np.matmul(smoothing_matrix, theta)
        theta = theta / sum(theta)

        logliklihood = np.inner(ns_hist, np.log(np.matmul(transform, theta)))
        imporve = logliklihood - old_logliklihood

        if r > 1 and abs(imporve) < loglikelihood_threshold:
            # print("stop when", imporve / old_logliklihood, loglikelihood_threshold)
            break

        old_logliklihood = logliklihood

        r += 1
    return theta


def EM(n, ns_hist, transform, max_iteration, loglikelihood_threshold):
    theta = np.ones(n) / float(n)
    theta_old = np.zeros(n)
    r = 0
    sample_size = sum(ns_hist)
    old_logliklihood = 0

    while LA.norm(theta_old - theta, ord=1) > 1 / sample_size and r < max_iteration:
        theta_old = np.copy(theta)
        X_condition = np.matmul(transform, theta_old)

        TMP = transform.T / X_condition

        P = np.copy(np.matmul(TMP, ns_hist))
        P = P * theta_old

        theta = np.copy(P / sum(P))

        logliklihood = np.inner(ns_hist, np.log(np.matmul(transform, theta)))
        imporve = logliklihood - old_logliklihood

        if r > 1 and abs(imporve) < loglikelihood_threshold:
            # print("stop when", imporve, loglikelihood_threshold)
            break

        old_logliklihood = logliklihood

        r += 1
    return theta
