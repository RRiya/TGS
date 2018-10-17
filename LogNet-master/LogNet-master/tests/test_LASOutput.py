import pytest

import os
import numpy as np
import pandas as pd 

from lognet.utilities.LASOutput import LASOutput

def test_LASOutput():

    output_path = os.getcwd()
    las_output = LASOutput(output_path)

    # Assert the output directory
    assert(las_output.output_path == output_path)

    # Asserts for writing LAS file