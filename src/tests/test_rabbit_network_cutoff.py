from unittest import TestCase
import raven.src.config as config
import src.adaptiveRaven as ver
from src.common import Dataset, RavenMode

class TestRaven(TestCase):
    def test_mnist_standard(self):
        net_names = [config.MNIST_CONV_SMALL]
        eps = 0.15
        for _ in range(1):
            args = ver.RavenArgs(raven_mode=RavenMode.UAP, dataset=Dataset.MNIST, net_names=net_names,
                count_per_prop=50, prop_count=10, eps=eps,
                threshold_execution=6, cross_executional_threshold=5, maximum_cross_execution_count=3, 
                baseline_iteration=20, refinement_iterations=20, unroll_layers = False, unroll_layer_count=3, 
                refine_intermediate_bounds = False, optimize_layers_count=2, 
                bounds_for_individual_refinement=False, dataloading_seed = 0,
                parallelize_executions=False,
                lp_threshold=-1.5,
                max_linear_apprx=3,
                device='cuda:0',
                always_correct_cross_execution = True,
                result_dir='recent_res', write_file=True, only_get_cutoff=True)
            ver.adptiveRaven(raven_args=args)
            eps += 0.01

    def test_mnist_diffai(self):
        net_names = [config.MNIST_CONV_SMALL_DIFFAI]
        eps = 0.15
        for _ in range(1):
            args = ver.RavenArgs(raven_mode=RavenMode.UAP, dataset=Dataset.MNIST, net_names=net_names,
                count_per_prop=50, prop_count=10, eps=eps,
                threshold_execution=6, cross_executional_threshold=5, maximum_cross_execution_count=3, 
                baseline_iteration=20, refinement_iterations=20, unroll_layers = False, unroll_layer_count=3, 
                refine_intermediate_bounds = False, optimize_layers_count=2, 
                bounds_for_individual_refinement=False, dataloading_seed = 0,
                parallelize_executions=False,
                lp_threshold=-1.5,
                max_linear_apprx=3,
                device='cuda:0',
                always_correct_cross_execution = True,
                result_dir='recent_res', write_file=True, only_get_cutoff=True)
            ver.adptiveRaven(raven_args=args)
            eps += 0.01

    def test_mnist_sabr(self):
        net_names = [config.MNIST_CONV_SMALL_SABR_1]
        eps = 0.15
        for _ in range(1):
            args = ver.RavenArgs(raven_mode=RavenMode.UAP, dataset=Dataset.MNIST, net_names=net_names,
                count_per_prop=50, prop_count=10, eps=eps,
                threshold_execution=6, cross_executional_threshold=5, maximum_cross_execution_count=3, 
                baseline_iteration=20, refinement_iterations=20, unroll_layers = False, unroll_layer_count=3, 
                refine_intermediate_bounds = False, optimize_layers_count=2, 
                bounds_for_individual_refinement=False, dataloading_seed = 0,
                parallelize_executions=False,
                lp_threshold=-1.5,
                max_linear_apprx=3,
                device='cuda:0',
                always_correct_cross_execution = True,
                result_dir='recent_res', write_file=True, only_get_cutoff=True)
            ver.adptiveRaven(raven_args=args)
            eps += 0.01

    def test_mnist_citrus(self):
        net_names = [config.MNIST_CONV_SMALL_CITRUS_1]
        eps = 0.15
        for _ in range(1):
            args = ver.RavenArgs(raven_mode=RavenMode.UAP, dataset=Dataset.MNIST, net_names=net_names,
                count_per_prop=50, prop_count=10, eps=eps,
                threshold_execution=6, cross_executional_threshold=5, maximum_cross_execution_count=3, 
                baseline_iteration=20, refinement_iterations=20, unroll_layers = False, unroll_layer_count=3, 
                refine_intermediate_bounds = False, optimize_layers_count=2, 
                bounds_for_individual_refinement=False, dataloading_seed = 0,
                parallelize_executions=False,
                lp_threshold=-1.5,
                max_linear_apprx=3,
                device='cuda:0',
                always_correct_cross_execution = True,
                result_dir='recent_res', write_file=True, only_get_cutoff=True)
            ver.adptiveRaven(raven_args=args)
            eps += 0.01

    def test_mnist_convbig(self):
        net_names = [config.MNIST_CONV_BIG]
        eps = 0.15
        for _ in range(1):
            args = ver.RavenArgs(raven_mode=RavenMode.UAP, dataset=Dataset.MNIST, net_names=net_names,
                count_per_prop=50, prop_count=10, eps=eps,
                threshold_execution=6, cross_executional_threshold=5, maximum_cross_execution_count=3, 
                baseline_iteration=20, refinement_iterations=20, unroll_layers = False, unroll_layer_count=3, 
                refine_intermediate_bounds = False, optimize_layers_count=2, 
                bounds_for_individual_refinement=False, dataloading_seed = 0,
                parallelize_executions=False,
                lp_threshold=-1.5,
                max_linear_apprx=3,
                device='cuda:0',
                always_correct_cross_execution = True,
                result_dir='recent_res', write_file=True, only_get_cutoff=True)
            ver.adptiveRaven(raven_args=args)
            eps += 0.01

    def test_cifar_sabr(self):
        net_names = [config.CIFAR_SABR_2]
        eps = 2.0
        for _ in range(1):
            args = ver.RavenArgs(raven_mode=RavenMode.UAP, dataset=Dataset.CIFAR10, net_names=net_names,
                count_per_prop=50, prop_count=10, eps=eps/255,
                threshold_execution=10, cross_executional_threshold=4, maximum_cross_execution_count=4, 
                baseline_iteration=20, refinement_iterations=20, unroll_layers = False, unroll_layer_count=3, 
                refine_intermediate_bounds =True, optimize_layers_count=2, 
                bounds_for_individual_refinement=True, dataloading_seed = 0,
                parallelize_executions=False,
                lp_threshold=-0.5,
                max_linear_apprx=3,
                device='cuda:2',
                always_correct_cross_execution = False,
                result_dir='recent_res', write_file=True, bound_improvement_ration=False,
                bound_ration_dir='bound_ration_final', only_get_cutoff=True)
            ver.adptiveRaven(raven_args=args)
            eps += 1.0

