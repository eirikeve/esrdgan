"""
test_config.py
Written by Eirik Vesterkjær, 2019
Apache License

tests for models/architectures/generator.py

run this using unittest: 
python -m unittest module_tests/generator_test.py

"""

import unittest
import tempfile
from models.architectures.generator import *



class TestGenerator(unittest.TestCase):
    def test_init(self):
        in_nc = 3
        out_nc = 3
        nf = 64
        n_rrdb = 23
        n_rrdb_convs = 5
        gc = 32
        upscale = 4
        rdb_res_scaling = 0.2
        rrdb_res_scaling = 0.2
        act_type = 'leakyrelu'



        net = ESRDnet( in_nc, out_nc, nf, n_rrdb, n_rrdb_convs, gc, 
                        rdb_res_scaling, rrdb_res_scaling, upscale=upscale, act_type=act_type)
        print(net)
        