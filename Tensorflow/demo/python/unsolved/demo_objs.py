#!/usr/bin/env python3 
# -*- coding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt

for property, value in np.getmembers([]):
    print( property, ": ", value);
