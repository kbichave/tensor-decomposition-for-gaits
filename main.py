
"""
Created on Mon Feb 18 13:04:45 2019

@author: kbich
"""

import argparse
from experiments.acc_vs_rd import AccVsReducedDimension
from experiments.acc_vs_samples import AccVsSamples
from experiments.acc_vs_rd_vs_samples import AccVsReducedDimensionVsSamples

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--samples', dest='exp', action='store_true')
    parser.add_argument('--dimension', dest='exp', action='store_false')
    parser.add_argument('--all', dest='exp_all', action='store_true')
    parser.set_defaults(exp_all = False)
    parser.set_defaults(exp=False)
    args = parser.parse_args()

    if args.exp and not args.exp_all: 
        experiment = AccVsSamples()
    elif not args.exp and not args.exp_all:
        experiment = AccVsReducedDimension()
    else:
        experiment = AccVsReducedDimensionVsSamples()
    
    experiment.run_exp()

    