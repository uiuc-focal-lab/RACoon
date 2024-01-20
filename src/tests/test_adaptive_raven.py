from unittest import TestCase
import raven.src.config as config
import src.adaptiveRaven as ver
from src.common import Dataset, RavenMode

class TestRaven(TestCase):
    def test_mnist_uap(self):
        net_names = [config.MNIST_CONV_SMALL_DIFFAI]
        eps = 0.1
        for _ in range(11):
            args = ver.RavenArgs(raven_mode=RavenMode.UAP, dataset=Dataset.MNIST, net_names=net_names,
                count_per_prop=20, prop_count=10, eps=eps,
                threshold_execution=10, cross_executional_threshold=4, maximum_cross_execution_count=4, 
                baseline_iteration=20, refinement_iterations=20, unroll_layers = False, unroll_layer_count=3, 
                refine_intermediate_bounds =True, optimize_layers_count=2, 
                bounds_for_individual_refinement=True, dataloading_seed = 0,
                parallelize_executions=False,
                lp_threshold=-1.5,
                max_linear_apprx=3,
                always_correct_cross_execution = False,
                result_dir='icml_results', write_file=True)
            ver.adptiveRaven(raven_args=args)
            eps += 0.005 
        
        net_names = [config.MNIST_CROWN_IBP]
        eps = 0.1
        for _ in range(11):
            args = ver.RavenArgs(raven_mode=RavenMode.UAP, dataset=Dataset.MNIST, net_names=net_names,
                count_per_prop=20, prop_count=10, eps=eps,
                threshold_execution=10, cross_executional_threshold=4, maximum_cross_execution_count=4, 
                baseline_iteration=20, refinement_iterations=20, unroll_layers = False, unroll_layer_count=3, 
                refine_intermediate_bounds =True, optimize_layers_count=2, 
                bounds_for_individual_refinement=True, dataloading_seed = 0,
                parallelize_executions=False,
                lp_threshold=-1.5,
                max_linear_apprx=3,
                always_correct_cross_execution = False,
                result_dir='icml_results', write_file=True)
            ver.adptiveRaven(raven_args=args)
            eps += 0.005 

        net_names = [config.MNIST_FFN_01]
        eps = 0.1
        for _ in range(11):
            args = ver.RavenArgs(raven_mode=RavenMode.UAP, dataset=Dataset.MNIST, net_names=net_names,
                count_per_prop=20, prop_count=10, eps=eps,
                threshold_execution=10, cross_executional_threshold=4, maximum_cross_execution_count=4, 
                baseline_iteration=20, refinement_iterations=20, unroll_layers = False, unroll_layer_count=3, 
                refine_intermediate_bounds =True, optimize_layers_count=2, 
                bounds_for_individual_refinement=True, dataloading_seed = 0,
                parallelize_executions=False,
                lp_threshold=-1.5,
                max_linear_apprx=3,
                always_correct_cross_execution = False,
                result_dir='icml_results', write_file=True)
            ver.adptiveRaven(raven_args=args)
            eps += 0.005 

        net_names = [config.MNIST_CROWN_IBP_MED]
        eps = 0.2
        for _ in range(16):
            args = ver.RavenArgs(raven_mode=RavenMode.UAP, dataset=Dataset.MNIST, net_names=net_names,
                count_per_prop=20, prop_count=10, eps=eps,
                threshold_execution=10, cross_executional_threshold=4, maximum_cross_execution_count=4, 
                baseline_iteration=20, refinement_iterations=20, unroll_layers = False, unroll_layer_count=3, 
                refine_intermediate_bounds =True, optimize_layers_count=2, 
                bounds_for_individual_refinement=True, dataloading_seed = 0,
                parallelize_executions=False,
                lp_threshold=-1.5,
                max_linear_apprx=3,
                always_correct_cross_execution = False,
                result_dir='icml_results', write_file=True)
            ver.adptiveRaven(raven_args=args)
            eps += 0.01

        net_names = [config.MNIST_CONV_SMALL]
        eps = 0.05
        for _ in range(15):
            args = ver.RavenArgs(raven_mode=RavenMode.UAP, dataset=Dataset.MNIST, net_names=net_names,
                count_per_prop=20, prop_count=10, eps=eps,
                threshold_execution=10, cross_executional_threshold=4, maximum_cross_execution_count=4, 
                baseline_iteration=20, refinement_iterations=20, unroll_layers = False, unroll_layer_count=3, 
                refine_intermediate_bounds =True, optimize_layers_count=2, 
                bounds_for_individual_refinement=True, dataloading_seed = 0,
                parallelize_executions=False,
                lp_threshold=-1.5,
                max_linear_apprx=3,
                always_correct_cross_execution = False,
                result_dir='icml_results', write_file=True)
            ver.adptiveRaven(raven_args=args)
            eps += 0.005 

        net_names = [config.MNIST_CONV_PGD]
        eps = 0.05
        for _ in range(15):
            args = ver.RavenArgs(raven_mode=RavenMode.UAP, dataset=Dataset.MNIST, net_names=net_names,
                count_per_prop=20, prop_count=10, eps=eps,
                threshold_execution=10, cross_executional_threshold=4, maximum_cross_execution_count=4, 
                baseline_iteration=20, refinement_iterations=20, unroll_layers = False, unroll_layer_count=3, 
                refine_intermediate_bounds =True, optimize_layers_count=2, 
                bounds_for_individual_refinement=True, dataloading_seed = 0,
                parallelize_executions=False,
                lp_threshold=-1.5,
                max_linear_apprx=3,
                always_correct_cross_execution = False,
                result_dir='icml_results', write_file=True)
            ver.adptiveRaven(raven_args=args)
            eps += 0.005 

    def test_mnist_uap_big(self):
        net_names = [config.MNIST_CONV_BIG]
        eps = 0.1
        for _ in range(31):
            args = ver.RavenArgs(raven_mode=RavenMode.UAP, dataset=Dataset.MNIST, net_names=net_names,
                count_per_prop=10, prop_count=10, eps=eps,
                threshold_execution=4, cross_executional_threshold=4, maximum_cross_execution_count=4, 
                baseline_iteration=20, refinement_iterations=20, unroll_layers = False, unroll_layer_count=3, 
                refine_intermediate_bounds =True, optimize_layers_count=2, 
                bounds_for_individual_refinement=True, dataloading_seed = 0,
                parallelize_executions=False,
                device='cuda:3',
                lp_threshold=-1.5,
                max_linear_apprx=3,
                always_correct_cross_execution = False,
                result_dir='icml_results', write_file=True)
            ver.adptiveRaven(raven_args=args)
            eps += 0.005 



    def test_cifar_uap(self):
        net_names = [config.CIFAR_CROWN_IBP]
        eps = 2.0
        for _ in range(15):
            args = ver.RavenArgs(raven_mode=RavenMode.UAP, dataset=Dataset.CIFAR10, net_names=net_names,
                count_per_prop=10, prop_count=10, eps=eps/255,
                threshold_execution=10, cross_executional_threshold=4, maximum_cross_execution_count=4, 
                baseline_iteration=20, refinement_iterations=20, unroll_layers = False, unroll_layer_count=3, 
                refine_intermediate_bounds =True, optimize_layers_count=2, 
                bounds_for_individual_refinement=True, dataloading_seed = 0,
                parallelize_executions=False,
                lp_threshold=-0.5,
                max_linear_apprx=3,
                device='cuda:2',
                always_correct_cross_execution = False,
                result_dir='icml_results', write_file=True)
            ver.adptiveRaven(raven_args=args)
            eps += 0.5
        

        # net_names = [config.CIFAR_CROWN_IBP_MEDIUM]
        # eps = 2.0
        # for _ in range(15):
        #     args = ver.RavenArgs(raven_mode=RavenMode.UAP, dataset=Dataset.CIFAR10, net_names=net_names,
        #         count_per_prop=20, prop_count=10, eps=eps/255,
        #         threshold_execution=10, cross_executional_threshold=4, maximum_cross_execution_count=4, 
        #         baseline_iteration=20, refinement_iterations=20, unroll_layers = False, unroll_layer_count=3, 
        #         refine_intermediate_bounds =True, optimize_layers_count=2, 
        #         bounds_for_individual_refinement=True, dataloading_seed = 0,
        #         parallelize_executions=False,
        #         lp_threshold=-0.5,
        #         max_linear_apprx=3,
        #         always_correct_cross_execution = False,
        #         result_dir='icml_results', write_file=True)
        #     ver.adptiveRaven(raven_args=args)
        #     eps += 0.5

        net_names = [config.CIFAR_CONV_COLT]
        eps = 2.0
        for _ in range(15):
            args = ver.RavenArgs(raven_mode=RavenMode.UAP, dataset=Dataset.CIFAR10, net_names=net_names,
                count_per_prop=10, prop_count=10, eps=eps/255,
                threshold_execution=10, cross_executional_threshold=4, maximum_cross_execution_count=4, 
                baseline_iteration=20, refinement_iterations=20, unroll_layers = False, unroll_layer_count=3, 
                refine_intermediate_bounds =True, optimize_layers_count=2, 
                bounds_for_individual_refinement=True, dataloading_seed = 0,
                parallelize_executions=False,
                lp_threshold=-0.5,
                max_linear_apprx=3,
                device='cuda:2',
                always_correct_cross_execution = False,
                result_dir='icml_results', write_file=True)
            ver.adptiveRaven(raven_args=args)
            eps += 0.5

        net_names = [config.CIFAR_CONV_DIFFAI]
        eps = 2.0
        for _ in range(15):
            args = ver.RavenArgs(raven_mode=RavenMode.UAP, dataset=Dataset.CIFAR10, net_names=net_names,
                count_per_prop=10, prop_count=10, eps=eps/255,
                threshold_execution=10, cross_executional_threshold=4, maximum_cross_execution_count=4, 
                baseline_iteration=20, refinement_iterations=20, unroll_layers = False, unroll_layer_count=3, 
                refine_intermediate_bounds =True, optimize_layers_count=2, 
                bounds_for_individual_refinement=True, dataloading_seed = 0,
                parallelize_executions=False,
                lp_threshold=-0.5,
                max_linear_apprx=3,
                device='cuda:2',
                always_correct_cross_execution = False,
                result_dir='icml_results', write_file=True)
            ver.adptiveRaven(raven_args=args)
            eps += 0.5

        net_names = [config.CIFAR_CONV_SMALL]
        eps = 0.5
        for _ in range(12):
            args = ver.RavenArgs(raven_mode=RavenMode.UAP, dataset=Dataset.CIFAR10, net_names=net_names,
                count_per_prop=10, prop_count=10, eps=eps/255,
                threshold_execution=10, cross_executional_threshold=4, maximum_cross_execution_count=4, 
                baseline_iteration=20, refinement_iterations=20, unroll_layers = False, unroll_layer_count=3, 
                refine_intermediate_bounds =True, optimize_layers_count=2, 
                bounds_for_individual_refinement=True, dataloading_seed = 0,
                parallelize_executions=False,
                lp_threshold=-0.5,
                max_linear_apprx=3,
                device='cuda:2',
                always_correct_cross_execution = False,
                result_dir='icml_results', write_file=True)
            ver.adptiveRaven(raven_args=args)
            eps += 0.25


        net_names = [config.CIFAR_CONV_SMALL_PGD]
        eps = 0.5
        for _ in range(12):
            args = ver.RavenArgs(raven_mode=RavenMode.UAP, dataset=Dataset.CIFAR10, net_names=net_names,
                count_per_prop=10, prop_count=10, eps=eps/255,
                threshold_execution=10, cross_executional_threshold=4, maximum_cross_execution_count=4, 
                baseline_iteration=20, refinement_iterations=20, unroll_layers = False, unroll_layer_count=3, 
                refine_intermediate_bounds =True, optimize_layers_count=2, 
                bounds_for_individual_refinement=True, dataloading_seed = 0,
                parallelize_executions=False,
                lp_threshold=-0.5,
                max_linear_apprx=3,
                device='cuda:2',
                always_correct_cross_execution = False,
                result_dir='icml_results', write_file=True)
            ver.adptiveRaven(raven_args=args)
            eps += 0.5

    def test_cifar_uap_medium(self):
        net_names = [config.CIFAR_CROWN_IBP_MEDIUM]
        eps = 1.0
        for _ in range(9):
            args = ver.RavenArgs(raven_mode=RavenMode.UAP, dataset=Dataset.CIFAR10, net_names=net_names,
                count_per_prop=10, prop_count=10, eps=eps/255,
                threshold_execution=4, cross_executional_threshold=4, maximum_cross_execution_count=4, 
                baseline_iteration=20, refinement_iterations=20, unroll_layers = False, unroll_layer_count=3, 
                refine_intermediate_bounds =True, optimize_layers_count=2, 
                bounds_for_individual_refinement=True, dataloading_seed = 0,
                parallelize_executions=False,
                lp_threshold=-0.5,
                max_linear_apprx=3,
                device='cuda:1',
                always_correct_cross_execution = False,
                result_dir='icml_results', write_file=True)
            ver.adptiveRaven(raven_args=args)
            eps += 0.25

    def test_cifar_uap_big(self):
        net_names = [config.CIFAR_CONV_BIG]
        eps = 0.5
        for _ in range(9):
            args = ver.RavenArgs(raven_mode=RavenMode.UAP, dataset=Dataset.CIFAR10, net_names=net_names,
                count_per_prop=5, prop_count=20, eps=eps/255,
                threshold_execution=3, cross_executional_threshold=3, maximum_cross_execution_count=3, 
                baseline_iteration=20, refinement_iterations=20, unroll_layers = False, unroll_layer_count=3, 
                refine_intermediate_bounds =True, optimize_layers_count=2, 
                bounds_for_individual_refinement=True, dataloading_seed = 0,
                parallelize_executions=False,
                lp_threshold=-0.5,
                max_linear_apprx=3,
                device='cuda:3',
                always_correct_cross_execution = False,
                result_dir='icml_results', write_file=True)
            ver.adptiveRaven(raven_args=args)
            eps += 0.25
