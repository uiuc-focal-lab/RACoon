from unittest import TestCase
import raven.src.config as config
import src.adaptiveRaven as ver
from src.common import Dataset, RavenMode

class TestRaven(TestCase):
    def test_mnist_uap(self):
        net_names = [config.MNIST_FFN_PGD]
        args = ver.RavenArgs(raven_mode=RavenMode.UAP, dataset=Dataset.MNIST, net_names=net_names,
                count_per_prop=20, prop_count=5, eps=0.055,
                threshold_execution=10, cross_executional_threshold=4, maximum_cross_execution_count=4, 
                baseline_iteration=10, refinement_iterations=40, unroll_layers = False, unroll_layer_count=3, 
                refine_intermediate_bounds = False, dataloading_seed = 0, 
                result_dir=None, write_file=True)
        
        ver.adptiveRaven(raven_args=args)