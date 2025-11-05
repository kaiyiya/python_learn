import sys
#sys.path.append('../lightNDF')
import numpy as np
import os
import configargparse

def config_parser():
    parser = configargparse.ArgumentParser()
    """
    Parse input arguments
    """
    parser = configargparse.ArgumentParser(description='myproject')

    parser.add_argument('--batch_size', type=int, default=4,
                        help='batch')



    return parser


def get_config():
    parser = config_parser()
    cfg = parser.parse_args()


    return cfg

