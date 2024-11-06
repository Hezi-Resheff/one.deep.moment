from numpy.linalg import matrix_power
from scipy.stats import rv_discrete
from scipy.special import factorial
import numpy as np
import sys
from scipy.linalg import expm, sinm, cosm


def compute_pdf_within_range(x_vals, s, A):
    '''
    compute_pdf_within_range
    :param x_vals:
    :param s:
    :param A:
    :return:
    '''
    pdf_list = []
    for x in x_vals:
        pdf_list.append(compute_pdf(x, s, A).flatten())

    return pdf_list

def compute_cdf_within_range(x_vals, s, A):
    '''
    compute_cdf_within_range
    :param x_vals:
    :param s:
    :param A:
    :return:
    '''
    pdf_list = []
    for x in x_vals:
        pdf_list.append(compute_cdf(x, s, A).flatten())

    return pdf_list

def compute_pdf(x, s, A):
    '''
    x: the value of pdf
    s: inital probs
    A: Generative matrix
    '''
    A0 = -np.dot(A, np.ones((A.shape[0], 1)))
    return np.dot(np.dot(s, expm(A * x)), A0)


def compute_cdf(x, s, A):
    '''
    x: the value of pdf
    s: inital probs
    A: Generative matrix
    '''
    A0 = -np.dot(A, np.ones((A.shape[0], 1)))
    return 1 - np.sum(np.dot(s, expm(A * x)))



def ser_moment_n(s, A, mom):
    '''
    ser_moment_n
    :param s:
    :param A:
    :param mom:
    :return:
    '''
    e = np.ones((A.shape[0], 1))
    try:
        mom_val = ((-1) ** mom) *factorial(mom)*np.dot(np.dot(s, matrix_power(A, -mom)), e)
        if mom_val > 0:
            return mom_val
        else:
            return False
    except:
        return False

def compute_first_n_moments(s, A, n=3):
    '''
    compute_first_n_moments
    :param s:
    :param A:
    :param n:
    :return:
    '''
    moment_list = []
    for moment in range(1, n + 1):
        moment_list.append(ser_moment_n(s, A, moment).item())
    return np.array(moment_list)


def create_erlang_row(rate, ind, size):
    aa = np.zeros(size)
    aa[ind] = -rate
    if ind < size - 1:
        aa[ind + 1] = rate
    return aa
