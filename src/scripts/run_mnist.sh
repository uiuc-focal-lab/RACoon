#!/bin/bash
python -m unittest -v src.tests.test_rabbit_network_cutoff.TestRaven.test_mnist_standard
python -m unittest -v src.tests.test_rabbit_network_cutoff.TestRaven.test_mnist_diffai
python -m unittest -v src.tests.test_rabbit_network_cutoff.TestRaven.test_mnist_sabr
python -m unittest -v src.tests.test_rabbit_network_cutoff.TestRaven.test_mnist_citrus
python -m unittest -v src.tests.test_rabbit_network_cutoff.TestRaven.test_mnist_convbig