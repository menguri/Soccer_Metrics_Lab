import os
import math
from tqdm import tqdm

import numpy as np
import pandas as pd
import ast
import matplotlib.pyplot as plt
import plotly
import plotly.graph_objects as go
import plotly.express as px

if os.getcwd().split(os.sep)[-1] == "notebooks":
    os.chdir("../")