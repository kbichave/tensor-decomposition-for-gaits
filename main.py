# -*- coding: utf-8 -*-
"""
Created on Mon Feb 18 13:04:45 2019

@author: kbich
"""

import os, argparse
from experiments.acc_vs_rd import AccVsReducedDimension
from experiments.acc_vs_samples import AccVsSamples

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--samples', dest='exp', action='store_true')
    parser.add_argument('--dimension', dest='exp', action='store_false')
    parser.set_defaults(exp=False)
    args = parser.parse_args()

    if args.exp: 
        experiment = AccVsSamples()
    else:
        experiment = AccVsReducedDimension()
    
    experiment.run_exp()

    