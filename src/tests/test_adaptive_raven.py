from unittest import TestCase
import raven.src.config as config
import src.adaptiveRaven as ver
from src.common import Dataset, RavenMode

class TestRaven(TestCase):
    def test_mnist_uap(self):
        # net_names = [config.MNIST_CONV_SMALL_DIFFAI]
        # eps = 0.1
        # for _ in range(10):
        #     args = ver.RavenArgs(raven_mode=RavenMode.UAP, dataset=Dataset.MNIST, net_names=net_names,
        #         count_per_prop=10, prop_count=10, eps=eps,
        #         threshold_execution=10, cross_executional_threshold=4, maximum_cross_execution_count=4, 
        #         baseline_iteration=20, refinement_iterations=30, unroll_layers = False, unroll_layer_count=3, 
        #         refine_intermediate_bounds = True, optimize_layers_count=2, 
        #         bounds_for_individual_refinement=False, dataloading_seed = 0, 
        #         result_dir='results', write_file=True)
        #     ver.adptiveRaven(raven_args=args)
        #     eps += 0.01

        # net_names = [config.MNIST_CROWN_IBP]
        # eps = 0.1
        # for _ in range(10):
        #     args = ver.RavenArgs(raven_mode=RavenMode.UAP, dataset=Dataset.MNIST, net_names=net_names,
        #         count_per_prop=10, prop_count=10, eps=eps,
        #         threshold_execution=10, cross_executional_threshold=4, maximum_cross_execution_count=4, 
        #         baseline_iteration=20, refinement_iterations=30, 
        #         unroll_layers = False, unroll_layer_count=3, 
        #         refine_intermediate_bounds=True, optimize_layers_count=2, 
        #         bounds_for_individual_refinement=False, dataloading_seed = 0, 
        #         result_dir='results', write_file=True)
        #     ver.adptiveRaven(raven_args=args)
        #     eps += 0.01

        net_names = [config.MNIST_CONV_BIG]
        eps = 0.12
        for _ in range(1):
            args = ver.RavenArgs(raven_mode=RavenMode.UAP, dataset=Dataset.MNIST, net_names=net_names,
                count_per_prop=10, prop_count=10, eps=eps,
                threshold_execution=5, cross_executional_threshold=4, maximum_cross_execution_count=4, 
                baseline_iteration=20, refinement_iterations=30, unroll_layers = False, unroll_layer_count=3, 
                refine_intermediate_bounds =True, optimize_layers_count=2, 
                bounds_for_individual_refinement=False, dataloading_seed = 0, 
                result_dir='results', write_file=True)
            ver.adptiveRaven(raven_args=args)
            eps += 0.01


    def test_cifar_uap(self):
        net_names = [config.CIFAR_CROWN_IBP]
        eps = 5.0/255
        for _ in range(1):
            args = ver.RavenArgs(raven_mode=RavenMode.UAP, dataset=Dataset.CIFAR10, net_names=net_names,
                count_per_prop=10, prop_count=10, eps=eps,
                threshold_execution=5, cross_executional_threshold=4, maximum_cross_execution_count=4, 
                baseline_iteration=30, refinement_iterations=30, unroll_layers = False, unroll_layer_count=3, 
                refine_intermediate_bounds = True, optimize_layers_count=2, full_alpha=False,
                dataloading_seed=0, result_dir='results', write_file=True)
            ver.adptiveRaven(raven_args=args)
            eps += 0.005