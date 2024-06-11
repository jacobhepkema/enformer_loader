#!/usr/bin/env python3

# Many functions adapted from Adapted from
# https://github.com/wconnell/enformer-finetune/blob/c2145a628efcb91b932cc063a658e4a994bc4baa/eft/preprocess.py

import numpy as np


def get_chrom_sizes(chrom_sizes_file) -> dict:
    """
    Get chromosome sizes from a file.
    """
    chrom_sizes = {}
    with open(chrom_sizes_file, "r") as f:
        for line in f:
            fields = line.split()
            chrom, size = fields[0], int(fields[1])
            if 'MT' not in chrom and 'KI' not in chrom and 'GL' not in chrom:
                chrom_sizes[chrom] = size
    return chrom_sizes


def avg_bin(array, n_bins):
    """
    Averages array values in n_bins.
    """
    splitted = np.array_split(array, n_bins)
    return [np.mean(a) for a in splitted]


def sum_bin(array, n_bins):
    """
    Sums array values in n_bins.
    """
    splitted = np.array_split(array, n_bins)
    return [np.sum(a) for a in splitted]


def get_bw_signal(bw_file, chrom, start, end, SEQ_LEN=114688):
    """
    Get signal from a bigwig file.
    If the chromosome is not found, return a list of np.nan.

    Arguments:
    - bw_file: pyBigWig file
    - chrom: chromosome
    - start: start position
    - end: end position
    - SEQ_LEN: length of the sequence (default: 114688)
    """
    center = (start + end) // 2
    start = center - (SEQ_LEN // 2)
    end = center + (SEQ_LEN // 2)
    try:
        values = bw_file.values(chrom, start, end)
        values = np.nan_to_num(values).tolist()
    except:
        values = [np.nan] * SEQ_LEN
    return values


def random_region(chrom_sizes, bw_file, p=None, SEQ_LEN=114688):
    """
    Get a random region from the genome.
    """
    chrom = np.random.choice(list(chrom_sizes.keys()), p=p)
    start = np.random.randint(0, chrom_sizes[chrom] - SEQ_LEN)
    end = start + SEQ_LEN
    values = get_bw_signal(bw_file, chrom, start, end, SEQ_LEN)
    return chrom, start, end, values
