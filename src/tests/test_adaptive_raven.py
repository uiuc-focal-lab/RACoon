from unittest import TestCase
import raven.src.config as config
import src.adaptiveRaven as ver
from src.common import Dataset, RavenMode

class TestRaven(TestCase):
    def test_mnist_uap(self):
        net_names = [config.MNIST_FFN_01]
        args = ver.RavenArgs(raven_mode=RavenMode.UAP, dataset=Dataset.MNIST, net_names=net_names,
                count_per_prop=15, prop_count=1, eps=0.11,
                threshold_execution=5, baseline_iteration=10,
                refinement_iterations=30, unroll_layers = False, unroll_layer_count=3, 
                refine_intermediate_bounds = False, dataloading_seed = 0, 
                result_dir=None, write_file=True)
        
        ver.adptiveRaven(raven_args=args)