#
# Implements the HORUS probabilistic WiFi Localization system
# ( Cyber Physical System Group )
# Jan. 2019
# Chris Xiaoxuan Lu
#

import numpy as np
import os
import glob
import sys
import matplotlib.pyplot as plt
from statsmodels.distributions.empirical_distribution import ECDF
from scipy.stats import norm
import argparse
import csv

def load_real_csv(csv_file):
  # EXAMPLE: "2017-06-22_15-02-47_wifi.csv"

  # 1.498139656195E9,eduroam,00:81:c4:85:07:a0,2462,-60
  # timestamp, ssid, mac, channel (ignore), rss
  return list((ssid, mac, rss) for timestamp, ssid, mac, channel, rss in csv.reader(open(csv_file, 'r')))


# TODO: Implement a fingerprinting algorithm that predicts the location given the testset
