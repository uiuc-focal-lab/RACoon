from unittest import TestCase
import raven.src.config as config
import src.adaptiveRaven as ver
from src.common import Dataset, RavenMode

class TestRaven(TestCase):
    def test_mnist_citrus(self):
        net_names = [config.MNIST_CONV_SMALL_CITRUS_1]
        eps = 0.15
        for _ in range(1):
            args = ver.RavenArgs(raven_mode=RavenMode.UAP, dataset=Dataset.MNIST, net_names=net_names,
                count_per_prop=5, prop_count=20, eps=eps,
                threshold_execution=6, cross_executional_threshold=5, maximum_cross_execution_count=3, 
                baseline_iteration=20, refinement_iterations=20, unroll_layers = False, unroll_layer_count=3, 
                refine_intermediate_bounds = False, optimize_layers_count=2, 
                bounds_for_individual_refinement=False, dataloading_seed = 0,
                parallelize_executions=False,
                lp_threshold=-1.5,
                max_linear_apprx=3,
                device='cuda:0',
                always_correct_cross_execution = True,
                result_dir='recent_res', write_file=True)
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

    def test_mnist_uap(self):
        # net_names = [config.MNIST_CONV_SMALL_CITRUS_1]
        # eps = 0.15
        # for _ in range(4):
        #     args = ver.RavenArgs(raven_mode=RavenMode.UAP, dataset=Dataset.MNIST, net_names=net_names,
        #         count_per_prop=20, prop_count=10, eps=eps,
        #         threshold_execution=6, cross_executional_threshold=5, maximum_cross_execution_count=3, 
        #         baseline_iteration=20, refinement_iterations=20, unroll_layers = False, unroll_layer_count=3, 
        #         refine_intermediate_bounds = True, optimize_layers_count=2, 
        #         bounds_for_individual_refinement=True, dataloading_seed = 0,
        #         parallelize_executions=False,
        #         lp_threshold=-1.5,
        #         max_linear_apprx=6,
        #         device='cuda:2',
        #         always_correct_cross_execution = True,
        #         result_dir='recent_res', write_file=True)
        #     ver.adptiveRaven(raven_args=args)
        #     eps += 0.005

        net_names = [config.MNIST_CONV_SMALL_DIFFAI]
        eps = 0.14
        for _ in range(1):
            args = ver.RavenArgs(raven_mode=RavenMode.UAP, dataset=Dataset.MNIST, net_names=net_names,
                count_per_prop=5, prop_count=20, eps=eps,
                threshold_execution=10, cross_executional_threshold=4, maximum_cross_execution_count=4, 
                baseline_iteration=20, refinement_iterations=20, unroll_layers = False, unroll_layer_count=3, 
                refine_intermediate_bounds =True, optimize_layers_count=2, 
                bounds_for_individual_refinement=True, dataloading_seed = 0,
                parallelize_executions=False,
                lp_threshold=-1.5,
                max_linear_apprx=3,
                device='cuda:3',
                always_correct_cross_execution = False,
                result_dir='recent_res', write_file=True)
            ver.adptiveRaven(raven_args=args)
            eps += 0.005 
        exit()
        # net_names = [config.MNIST_CROWN_IBP]
        # eps = 0.13
        # for _ in range(1):
        #     args = ver.RavenArgs(raven_mode=RavenMode.UAP, dataset=Dataset.MNIST, net_names=net_names,
        #         count_per_prop=5, prop_count=20, eps=eps,
        #         threshold_execution=10, cross_executional_threshold=4, maximum_cross_execution_count=4, 
        #         baseline_iteration=20, refinement_iterations=40, unroll_layers = False, unroll_layer_count=3, 
        #         refine_intermediate_bounds =False, optimize_layers_count=2, 
        #         bounds_for_individual_refinement=False, dataloading_seed = 0,
        #         parallelize_executions=False,
        #         lp_threshold=-1.5,
        #         max_linear_apprx=3,
        #         device='cuda:2',
        #         always_correct_cross_execution = True,
        #         result_dir='recent_res', write_file=True)
        #     ver.adptiveRaven(raven_args=args)
        #     eps += 0.005 

        # net_names = [config.MNIST_FFN_01]
        # eps = 0.15
        # for _ in range(1):
        #     args = ver.RavenArgs(raven_mode=RavenMode.UAP, dataset=Dataset.MNIST, net_names=net_names,
        #         count_per_prop=10, prop_count=20, eps=eps,
        #         threshold_execution=10, cross_executional_threshold=4, maximum_cross_execution_count=4, 
        #         baseline_iteration=20, refinement_iterations=20, unroll_layers = False, unroll_layer_count=3, 
        #         refine_intermediate_bounds =True, optimize_layers_count=2, 
        #         bounds_for_individual_refinement=False, dataloading_seed = 0,
        #         parallelize_executions=False,
        #         lp_threshold=-1.5,
        #         max_linear_apprx=3,
        #         always_correct_cross_execution = False,
        #         result_dir='recent_res', write_file=True,
        #         device='cuda:2')
        #     ver.adptiveRaven(raven_args=args)
        #     eps += 0.005 

        # net_names = [config.MNIST_CROWN_IBP_MED]
        # eps = 0.2
        # for _ in range(16):
        #     args = ver.RavenArgs(raven_mode=RavenMode.UAP, dataset=Dataset.MNIST, net_names=net_names,
        #         count_per_prop=20, prop_count=10, eps=eps,
        #         threshold_execution=10, cross_executional_threshold=4, maximum_cross_execution_count=4, 
        #         baseline_iteration=20, refinement_iterations=20, unroll_layers = False, unroll_layer_count=3, 
        #         refine_intermediate_bounds =True, optimize_layers_count=2, 
        #         bounds_for_individual_refinement=True, dataloading_seed = 0,
        #         parallelize_executions=False,
        #         lp_threshold=-1.5,
        #         max_linear_apprx=3,
        #         always_correct_cross_execution = False,
        #         result_dir='icml_results_new', write_file=True)
        #     ver.adptiveRaven(raven_args=args)
        #     eps += 0.01

        net_names = [config.MNIST_CONV_SMALL]
        eps = 0.06
        for _ in range(3):
            args = ver.RavenArgs(raven_mode=RavenMode.UAP, dataset=Dataset.MNIST, net_names=net_names,
                count_per_prop=20, prop_count=10, eps=eps,
                threshold_execution=10, cross_executional_threshold=4, maximum_cross_execution_count=4, 
                baseline_iteration=20, refinement_iterations=20, unroll_layers = False, unroll_layer_count=3, 
                refine_intermediate_bounds =True, optimize_layers_count=2, 
                bounds_for_individual_refinement=True, dataloading_seed = 0,
                parallelize_executions=False,
                lp_threshold=-1.5,
                max_linear_apprx=3,
                device='cuda:2',
                always_correct_cross_execution = False,
                result_dir='recent_res', write_file=True)
            ver.adptiveRaven(raven_args=args)
            eps += 0.005 

        net_names = [config.MNIST_CONV_PGD]
        eps = 0.08
        for _ in range(4):
            args = ver.RavenArgs(raven_mode=RavenMode.UAP, dataset=Dataset.MNIST, net_names=net_names,
                count_per_prop=20, prop_count=10, eps=eps,
                threshold_execution=10, cross_executional_threshold=4, maximum_cross_execution_count=4, 
                baseline_iteration=20, refinement_iterations=20, unroll_layers = False, unroll_layer_count=3, 
                refine_intermediate_bounds =True, optimize_layers_count=2, 
                bounds_for_individual_refinement=True, dataloading_seed = 0,
                parallelize_executions=False,
                lp_threshold=-1.5,
                max_linear_apprx=3,
                device='cuda:2',
                always_correct_cross_execution = False,
                result_dir='recent_res', write_file=True)
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
                refine_intermediate_bounds =False, optimize_layers_count=2, 
                bounds_for_individual_refinement=True, dataloading_seed = 0,
                parallelize_executions=False,
                device='cuda:0',
                lp_threshold=-1.5,
                max_linear_apprx=3,
                always_correct_cross_execution = False,
                result_dir='recent_res', write_file=True)
            ver.adptiveRaven(raven_args=args)
            eps += 0.005 



    def test_cifar_uap(self):
        net_names = [config.CIFAR_CONV_SMALL]
        eps = 1.0
        for _ in range(1):
            args = ver.RavenArgs(raven_mode=RavenMode.UAP, dataset=Dataset.CIFAR10, net_names=net_names,
                count_per_prop=20, prop_count=10, eps=eps/255,
                threshold_execution=10, cross_executional_threshold=4, maximum_cross_execution_count=4, 
                baseline_iteration=20, refinement_iterations=20, unroll_layers = False, unroll_layer_count=3, 
                refine_intermediate_bounds =True, optimize_layers_count=2, 
                bounds_for_individual_refinement=True, dataloading_seed = 0,
                parallelize_executions=False,
                lp_threshold=-0.5,
                max_linear_apprx=10,
                device='cuda:2',
                always_correct_cross_execution = False,
                result_dir='recent_res', write_file=True)
            ver.adptiveRaven(raven_args=args)
            eps += 0.5
        

        # net_names = [config.CIFAR_CROWN_IBP]
        # eps = 6.0
        # for _ in range(1):
        #     args = ver.RavenArgs(raven_mode=RavenMode.UAP, dataset=Dataset.CIFAR10, net_names=net_names,
        #         count_per_prop=5, prop_count=20, eps=eps/255,
        #         threshold_execution=10, cross_executional_threshold=4, maximum_cross_execution_count=4, 
        #         baseline_iteration=20, refinement_iterations=20, unroll_layers = False, unroll_layer_count=3, 
        #         refine_intermediate_bounds =True, optimize_layers_count=2, 
        #         bounds_for_individual_refinement=True, dataloading_seed = 0,
        #         parallelize_executions=False,
        #         lp_threshold=-0.5,
        #         max_linear_apprx=3,
        #         device='cuda:0',
        #         always_correct_cross_execution = False,
        #         result_dir='recent_res', write_file=True)
        #     ver.adptiveRaven(raven_args=args)
        #     eps += 0.5

        # net_names = [config.CIFAR_CONV_COLT]
        # eps = 2.0
        # for _ in range(15):
        #     args = ver.RavenArgs(raven_mode=RavenMode.UAP, dataset=Dataset.CIFAR10, net_names=net_names,
        #         count_per_prop=10, prop_count=10, eps=eps/255,
        #         threshold_execution=10, cross_executional_threshold=4, maximum_cross_execution_count=4, 
        #         baseline_iteration=20, refinement_iterations=20, unroll_layers = False, unroll_layer_count=3, 
        #         refine_intermediate_bounds =True, optimize_layers_count=2, 
        #         bounds_for_individual_refinement=True, dataloading_seed = 0,
        #         parallelize_executions=False,
        #         lp_threshold=-0.5,
        #         max_linear_apprx=3,
        #         device='cuda:2',
        #         always_correct_cross_execution = False,
        #         result_dir='icml_results_new', write_file=True)
        #     ver.adptiveRaven(raven_args=args)
        #     eps += 0.5

        net_names = [config.CIFAR_CONV_DIFFAI]
        eps = 8.0
        for _ in range(1):
            args = ver.RavenArgs(raven_mode=RavenMode.UAP, dataset=Dataset.CIFAR10, net_names=net_names,
                count_per_prop=20, prop_count=10, eps=eps/255,
                threshold_execution=10, cross_executional_threshold=4, maximum_cross_execution_count=4, 
                baseline_iteration=20, refinement_iterations=20, unroll_layers = False, unroll_layer_count=3, 
                refine_intermediate_bounds =True, optimize_layers_count=2, 
                bounds_for_individual_refinement=True, dataloading_seed = 0,
                parallelize_executions=False,
                lp_threshold=-0.5,
                max_linear_apprx=6,
                device='cuda:2',
                always_correct_cross_execution = False,
                result_dir='recent_res', write_file=True)
            ver.adptiveRaven(raven_args=args)
            eps += 0.5

        # net_names = [config.CIFAR_CONV_SMALL]
        # eps = 0.5
        # for _ in range(12):
        #     args = ver.RavenArgs(raven_mode=RavenMode.UAP, dataset=Dataset.CIFAR10, net_names=net_names,
        #         count_per_prop=10, prop_count=10, eps=eps/255,
        #         threshold_execution=10, cross_executional_threshold=4, maximum_cross_execution_count=4, 
        #         baseline_iteration=20, refinement_iterations=20, unroll_layers = False, unroll_layer_count=3, 
        #         refine_intermediate_bounds =True, optimize_layers_count=2, 
        #         bounds_for_individual_refinement=True, dataloading_seed = 0,
        #         parallelize_executions=False,
        #         lp_threshold=-0.5,
        #         max_linear_apprx=3,
        #         device='cuda:2',
        #         always_correct_cross_execution = False,
        #         result_dir='icml_results_new', write_file=True)
        #     ver.adptiveRaven(raven_args=args)
        #     eps += 0.25


        net_names = [config.CIFAR_CONV_SMALL_PGD]
        eps = 3.0
        for _ in range(1):
            args = ver.RavenArgs(raven_mode=RavenMode.UAP, dataset=Dataset.CIFAR10, net_names=net_names,
                count_per_prop=20, prop_count=10, eps=eps/255,
                threshold_execution=10, cross_executional_threshold=4, maximum_cross_execution_count=4, 
                baseline_iteration=20, refinement_iterations=20, unroll_layers = False, unroll_layer_count=3, 
                refine_intermediate_bounds =True, optimize_layers_count=2, 
                bounds_for_individual_refinement=True, dataloading_seed = 0,
                parallelize_executions=False,
                lp_threshold=-0.5,
                max_linear_apprx=6,
                device='cuda:2',
                always_correct_cross_execution = False,
                result_dir='recent_res', write_file=True)
            ver.adptiveRaven(raven_args=args)
            eps += 0.5

        net_names = [config.CIFAR_CITRUS_2]
        eps = 2.0
        for _ in range(1):
            args = ver.RavenArgs(raven_mode=RavenMode.UAP, dataset=Dataset.CIFAR10, net_names=net_names,
                count_per_prop=20, prop_count=10, eps=eps/255,
                threshold_execution=10, cross_executional_threshold=4, maximum_cross_execution_count=4, 
                baseline_iteration=20, refinement_iterations=20, unroll_layers = False, unroll_layer_count=3, 
                refine_intermediate_bounds =True, optimize_layers_count=2, 
                bounds_for_individual_refinement=True, dataloading_seed = 0,
                parallelize_executions=False,
                lp_threshold=-0.5,
                max_linear_apprx=6,
                device='cuda:2',
                always_correct_cross_execution = False,
                result_dir='recent_res', write_file=True)
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

    def test_mnist_uap_bound_refinement(self):
        net_names = [config.MNIST_CONV_PGD]
        eps = 0.1
        for _ in range(1):
            args = ver.RavenArgs(raven_mode=RavenMode.UAP, dataset=Dataset.MNIST, net_names=net_names,
                count_per_prop=20, prop_count=10, eps=eps,
                threshold_execution=10, cross_executional_threshold=4, maximum_cross_execution_count=4, 
                baseline_iteration=20, refinement_iterations=20, unroll_layers = False, unroll_layer_count=3, 
                refine_intermediate_bounds =True, optimize_layers_count=2, 
                bounds_for_individual_refinement=True, dataloading_seed = 0,
                parallelize_executions=False,
                lp_threshold=-105,
                max_linear_apprx=300,
                populate_trace=True,
                device='cuda:3',
                always_correct_cross_execution = True,
                result_dir='icml_results', write_file=False)
            ver.adptiveRaven(raven_args=args)
            eps += 0.005 

        net_names = [config.MNIST_CONV_SMALL_DIFFAI]
        eps = 0.12
        for _ in range(1):
            args = ver.RavenArgs(raven_mode=RavenMode.UAP, dataset=Dataset.MNIST, net_names=net_names,
                count_per_prop=20, prop_count=10, eps=eps,
                threshold_execution=10, cross_executional_threshold=4, maximum_cross_execution_count=4, 
                baseline_iteration=20, refinement_iterations=20, unroll_layers = False, unroll_layer_count=3, 
                refine_intermediate_bounds =True, optimize_layers_count=2, 
                bounds_for_individual_refinement=True, dataloading_seed = 0,
                parallelize_executions=False,
                lp_threshold=-100,
                max_linear_apprx=300,
                populate_trace=True,
                device='cuda:3',
                always_correct_cross_execution = True,
                result_dir='icml_results', write_file=False)
            ver.adptiveRaven(raven_args=args)
            eps += 0.005 

    def test_cifar_uap_bound_refinement(self):
        net_names = [config.CIFAR_CONV_DIFFAI]
        eps = 6.0
        for _ in range(1):
            args = ver.RavenArgs(raven_mode=RavenMode.UAP, dataset=Dataset.CIFAR10, net_names=net_names,
                count_per_prop=20, prop_count=10, eps=eps/255,
                threshold_execution=10, cross_executional_threshold=4, maximum_cross_execution_count=4, 
                baseline_iteration=20, refinement_iterations=20, unroll_layers = False, unroll_layer_count=3, 
                refine_intermediate_bounds =True, optimize_layers_count=2, 
                bounds_for_individual_refinement=True, dataloading_seed = 0,
                parallelize_executions=False,
                lp_threshold=-200,
                max_linear_apprx=100,
                populate_trace=True,
                device='cuda:2',
                always_correct_cross_execution = True,
                result_dir='icml_results', write_file=False)
            ver.adptiveRaven(raven_args=args)
            eps += 0.005 


    def test_mnist_uap_diff_k(self):
        net_names = [config.MNIST_CONV_SMALL_DIFFAI]
        eps = 0.1
        for _ in range(11):
            for count_prop in [10, 20, 30, 40, 50]:
                args = ver.RavenArgs(raven_mode=RavenMode.UAP, dataset=Dataset.MNIST, net_names=net_names,
                    count_per_prop=count_prop, prop_count=10, eps=eps,
                    threshold_execution=10, cross_executional_threshold=4, maximum_cross_execution_count=4, 
                    baseline_iteration=20, refinement_iterations=20, unroll_layers = False, unroll_layer_count=3, 
                    refine_intermediate_bounds =True, optimize_layers_count=2, 
                    bounds_for_individual_refinement=True, dataloading_seed = 0,
                    parallelize_executions=False,
                    lp_threshold=-1.5,
                    max_linear_apprx=3,
                    always_correct_cross_execution = False,
                    result_dir='icml_results_diff_k', write_file=True)
                ver.adptiveRaven(raven_args=args)
            eps += 0.005 
        
        net_names = [config.MNIST_CROWN_IBP]
        eps = 0.1
        for _ in range(11):
            for count_prop in [10, 20, 30, 40, 50]:
                args = ver.RavenArgs(raven_mode=RavenMode.UAP, dataset=Dataset.MNIST, net_names=net_names,
                count_per_prop=count_prop, prop_count=10, eps=eps,
                threshold_execution=10, cross_executional_threshold=4, maximum_cross_execution_count=4, 
                baseline_iteration=20, refinement_iterations=20, unroll_layers = False, unroll_layer_count=3, 
                refine_intermediate_bounds =True, optimize_layers_count=2, 
                bounds_for_individual_refinement=True, dataloading_seed = 0,
                parallelize_executions=False,
                lp_threshold=-1.5,
                max_linear_apprx=3,
                always_correct_cross_execution = False,
                result_dir='icml_results_diff_k', write_file=True)
                ver.adptiveRaven(raven_args=args)
            eps += 0.005 

        net_names = [config.MNIST_FFN_01]
        eps = 0.1
        for _ in range(11):
            for count_prop in [10, 20, 30, 40, 50]:
                args = ver.RavenArgs(raven_mode=RavenMode.UAP, dataset=Dataset.MNIST, net_names=net_names,
                count_per_prop=count_prop, prop_count=10, eps=eps,
                threshold_execution=10, cross_executional_threshold=4, maximum_cross_execution_count=4, 
                baseline_iteration=20, refinement_iterations=20, unroll_layers = False, unroll_layer_count=3, 
                refine_intermediate_bounds =True, optimize_layers_count=2, 
                bounds_for_individual_refinement=True, dataloading_seed = 0,
                parallelize_executions=False,
                lp_threshold=-1.5,
                max_linear_apprx=3,
                always_correct_cross_execution = False,
                result_dir='icml_results_diff_k', write_file=True)
                ver.adptiveRaven(raven_args=args)
            eps += 0.005 

        net_names = [config.MNIST_CONV_SMALL]
        eps = 0.05
        for _ in range(15):
            for count_prop in [10, 20, 30, 40, 50]:
                args = ver.RavenArgs(raven_mode=RavenMode.UAP, dataset=Dataset.MNIST, net_names=net_names,
                count_per_prop=count_prop, prop_count=10, eps=eps,
                threshold_execution=10, cross_executional_threshold=4, maximum_cross_execution_count=4, 
                baseline_iteration=20, refinement_iterations=20, unroll_layers = False, unroll_layer_count=3, 
                refine_intermediate_bounds =True, optimize_layers_count=2, 
                bounds_for_individual_refinement=True, dataloading_seed = 0,
                parallelize_executions=False,
                lp_threshold=-1.5,
                max_linear_apprx=3,
                always_correct_cross_execution = False,
                result_dir='icml_results_diff_k', write_file=True)
                ver.adptiveRaven(raven_args=args)
            eps += 0.005 

        net_names = [config.MNIST_CONV_PGD]
        eps = 0.05
        for _ in range(15):
            for count_prop in [10, 20, 30, 40, 50]:
                args = ver.RavenArgs(raven_mode=RavenMode.UAP, dataset=Dataset.MNIST, net_names=net_names,
                count_per_prop=count_prop, prop_count=10, eps=eps,
                threshold_execution=10, cross_executional_threshold=4, maximum_cross_execution_count=4, 
                baseline_iteration=20, refinement_iterations=20, unroll_layers = False, unroll_layer_count=3, 
                refine_intermediate_bounds =True, optimize_layers_count=2, 
                bounds_for_individual_refinement=True, dataloading_seed = 0,
                parallelize_executions=False,
                lp_threshold=-1.5,
                max_linear_apprx=3,
                always_correct_cross_execution = False,
                result_dir='icml_results_diff_k', write_file=True)
                ver.adptiveRaven(raven_args=args)
            eps += 0.005 


    def test_cifar_uap_diff_k(self):
        net_names = [config.CIFAR_CROWN_IBP]
        eps = 2.0
        for _ in range(10):
            for count_prop in [5, 10, 15, 20, 25]:
                args = ver.RavenArgs(raven_mode=RavenMode.UAP, dataset=Dataset.CIFAR10, net_names=net_names,
                    count_per_prop=count_prop, prop_count=10, eps=eps/255,
                    threshold_execution=10, cross_executional_threshold=4, maximum_cross_execution_count=4, 
                    baseline_iteration=20, refinement_iterations=20, unroll_layers = False, unroll_layer_count=3, 
                    refine_intermediate_bounds =True, optimize_layers_count=2, 
                    bounds_for_individual_refinement=True, dataloading_seed = 0,
                    parallelize_executions=False,
                    lp_threshold=-0.5,
                    max_linear_apprx=3,
                    device='cuda:0',
                    always_correct_cross_execution = False,
                    result_dir='icml_results_diff_k', write_file=True)
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
        #         result_dir='icml_results_diff_k', write_file=True)
        #     ver.adptiveRaven(raven_args=args)
        #     eps += 0.5

        net_names = [config.CIFAR_CONV_COLT]
        eps = 2.0
        for _ in range(10):
            for count_prop in [5, 10, 15, 20, 25]:
                args = ver.RavenArgs(raven_mode=RavenMode.UAP, dataset=Dataset.CIFAR10, net_names=net_names,
                    count_per_prop=10, prop_count=10, eps=eps/255,
                    threshold_execution=10, cross_executional_threshold=4, maximum_cross_execution_count=4, 
                    baseline_iteration=20, refinement_iterations=20, unroll_layers = False, unroll_layer_count=3, 
                    refine_intermediate_bounds =True, optimize_layers_count=2, 
                    bounds_for_individual_refinement=True, dataloading_seed = 0,
                    parallelize_executions=False,
                    lp_threshold=-0.5,
                    max_linear_apprx=3,
                    device='cuda:0',
                    always_correct_cross_execution = False,
                    result_dir='icml_results_diff_k', write_file=True)
                ver.adptiveRaven(raven_args=args)
            eps += 0.5

        net_names = [config.CIFAR_CONV_DIFFAI]
        eps = 2.0
        for _ in range(10):
            for count_prop in [5, 10, 15, 20, 25]:
                args = ver.RavenArgs(raven_mode=RavenMode.UAP, dataset=Dataset.CIFAR10, net_names=net_names,
                    count_per_prop=10, prop_count=10, eps=eps/255,
                    threshold_execution=10, cross_executional_threshold=4, maximum_cross_execution_count=4, 
                    baseline_iteration=20, refinement_iterations=20, unroll_layers = False, unroll_layer_count=3, 
                    refine_intermediate_bounds =True, optimize_layers_count=2, 
                    bounds_for_individual_refinement=True, dataloading_seed = 0,
                    parallelize_executions=False,
                    lp_threshold=-0.5,
                    max_linear_apprx=3,
                    device='cuda:0',
                    always_correct_cross_execution = False,
                    result_dir='icml_results_diff_k', write_file=True)
                ver.adptiveRaven(raven_args=args)
            eps += 0.5

        net_names = [config.CIFAR_CONV_SMALL]
        eps = 0.5
        for _ in range(10):
            for count_prop in [5, 10, 15, 20, 25]:
                args = ver.RavenArgs(raven_mode=RavenMode.UAP, dataset=Dataset.CIFAR10, net_names=net_names,
                    count_per_prop=10, prop_count=10, eps=eps/255,
                    threshold_execution=10, cross_executional_threshold=4, maximum_cross_execution_count=4, 
                    baseline_iteration=20, refinement_iterations=20, unroll_layers = False, unroll_layer_count=3, 
                    refine_intermediate_bounds =True, optimize_layers_count=2, 
                    bounds_for_individual_refinement=True, dataloading_seed = 0,
                    parallelize_executions=False,
                    lp_threshold=-0.5,
                    max_linear_apprx=3,
                    device='cuda:0',
                    always_correct_cross_execution = False,
                    result_dir='icml_results_diff_k', write_file=True)
                ver.adptiveRaven(raven_args=args)
            eps += 0.25


        net_names = [config.CIFAR_CONV_SMALL_PGD]
        eps = 0.5
        for _ in range(10):
            for count_prop in [5, 10, 15, 20, 25]:
                args = ver.RavenArgs(raven_mode=RavenMode.UAP, dataset=Dataset.CIFAR10, net_names=net_names,
                    count_per_prop=10, prop_count=10, eps=eps/255,
                    threshold_execution=10, cross_executional_threshold=4, maximum_cross_execution_count=4, 
                    baseline_iteration=20, refinement_iterations=20, unroll_layers = False, unroll_layer_count=3, 
                    refine_intermediate_bounds =True, optimize_layers_count=2, 
                    bounds_for_individual_refinement=True, dataloading_seed = 0,
                    parallelize_executions=False,
                    lp_threshold=-0.5,
                    max_linear_apprx=3,
                    device='cuda:0',
                    always_correct_cross_execution = False,
                    result_dir='icml_results_diff_k', write_file=True)
                ver.adptiveRaven(raven_args=args)
            eps += 0.5


    def test_mnist_hamming(self):
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
                device='cuda:3',
                always_correct_cross_execution = False,
                result_dir='icml_hamming', write_file=True)
            ver.adptiveRaven(raven_args=args)
            eps += (0.005 * 2) 
        
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
                device='cuda:3',
                max_linear_apprx=3,
                always_correct_cross_execution = False,
                result_dir='icml_hamming', write_file=True)
            ver.adptiveRaven(raven_args=args)
            eps += (0.005 * 2) 

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
                device='cuda:3',
                lp_threshold=-1.5,
                max_linear_apprx=3,
                always_correct_cross_execution = False,
                result_dir='icml_hamming', write_file=True)
            ver.adptiveRaven(raven_args=args)
            eps += (0.005 * 2)

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
                device='cuda:3',
                max_linear_apprx=3,
                always_correct_cross_execution = False,
                result_dir='icml_hamming', write_file=True)
            ver.adptiveRaven(raven_args=args)
            eps += (0.01 * 2)

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
                device='cuda:3',
                max_linear_apprx=3,
                always_correct_cross_execution = False,
                result_dir='icml_hamming', write_file=True)
            ver.adptiveRaven(raven_args=args)
            eps += (0.005 * 2) 

        net_names = [config.MNIST_CONV_PGD]
        eps = 0.05
        for _ in range(15):
            args = ver.RavenArgs(raven_mode=RavenMode.UAP, dataset=Dataset.MNIST, net_names=net_names,
                count_per_prop=20, prop_count=10, eps=eps,
                threshold_execution=10, cross_executional_threshold=4, maximum_cross_execution_count=4, 
                baseline_iteration=20, refinement_iterations=20, unroll_layers = False, unroll_layer_count=3, 
                refine_intermediate_bounds =True, optimize_layers_count=2, 
                bounds_for_individual_refinement=True, dataloading_seed = 0,
                device='cuda:3',
                parallelize_executions=False,
                lp_threshold=-1.5,
                max_linear_apprx=3,
                always_correct_cross_execution = False,
                result_dir='icml_hamming', write_file=True)
            ver.adptiveRaven(raven_args=args)
            eps += (0.005 * 2)

    def test_mnist_bound_ration(self):
        net_names = [config.MNIST_CONV_SMALL_DIFFAI]
        eps = 0.12
        for _ in range(1):
            args = ver.RavenArgs(raven_mode=RavenMode.UAP, dataset=Dataset.MNIST, net_names=net_names,
                count_per_prop=20, prop_count=40, eps=eps,
                threshold_execution=10, cross_executional_threshold=4, maximum_cross_execution_count=4, 
                baseline_iteration=20, refinement_iterations=20, unroll_layers = False, unroll_layer_count=3, 
                refine_intermediate_bounds =True, optimize_layers_count=2, 
                bounds_for_individual_refinement=True, dataloading_seed = 0,
                parallelize_executions=False,
                lp_threshold=-1.5,
                max_linear_apprx=3,
                device='cuda:0',
                always_correct_cross_execution = True,
                result_dir='recent_res', write_file=True, bound_improvement_ration=True,
                bound_ration_dir='bound_ration_final')
            ver.adptiveRaven(raven_args=args)
            eps += 0.005 

        net_names = [config.MNIST_CONV_SMALL]
        eps = 0.08
        for _ in range(1):
            args = ver.RavenArgs(raven_mode=RavenMode.UAP, dataset=Dataset.MNIST, net_names=net_names,
                count_per_prop=20, prop_count=40, eps=eps,
                threshold_execution=10, cross_executional_threshold=4, maximum_cross_execution_count=4, 
                baseline_iteration=20, refinement_iterations=20, unroll_layers = False, unroll_layer_count=3, 
                refine_intermediate_bounds =True, optimize_layers_count=2, 
                bounds_for_individual_refinement=True, dataloading_seed = 0,
                parallelize_executions=False,
                lp_threshold=-1.5,
                max_linear_apprx=3,
                device='cuda:0',
                always_correct_cross_execution = True,
                result_dir='recent_res', write_file=True, bound_improvement_ration=True,
                bound_ration_dir='bound_ration_final')
            ver.adptiveRaven(raven_args=args)
            eps += 0.005 

        net_names = [config.MNIST_CONV_PGD]
        eps = 0.1
        for _ in range(1):
            args = ver.RavenArgs(raven_mode=RavenMode.UAP, dataset=Dataset.MNIST, net_names=net_names,
                count_per_prop=20, prop_count=40, eps=eps,
                threshold_execution=10, cross_executional_threshold=4, maximum_cross_execution_count=4, 
                baseline_iteration=20, refinement_iterations=20, unroll_layers = False, unroll_layer_count=3, 
                refine_intermediate_bounds =True, optimize_layers_count=2, 
                bounds_for_individual_refinement=True, dataloading_seed = 0,
                parallelize_executions=False,
                lp_threshold=-1.5,
                max_linear_apprx=3,
                device='cuda:0',
                always_correct_cross_execution = True,
                result_dir='recent_res', write_file=True, bound_improvement_ration=True,
                bound_ration_dir='bound_ration_final')
            ver.adptiveRaven(raven_args=args)
            eps += 0.005 

    def test_mnist_bound_ration(self):
        net_names = [config.MNIST_CONV_SMALL_DIFFAI]
        eps = 0.12
        for _ in range(1):
            args = ver.RavenArgs(raven_mode=RavenMode.UAP, dataset=Dataset.MNIST, net_names=net_names,
                count_per_prop=20, prop_count=40, eps=eps,
                threshold_execution=10, cross_executional_threshold=4, maximum_cross_execution_count=4, 
                baseline_iteration=20, refinement_iterations=20, unroll_layers = False, unroll_layer_count=3, 
                refine_intermediate_bounds =True, optimize_layers_count=2, 
                bounds_for_individual_refinement=True, dataloading_seed = 0,
                parallelize_executions=False,
                lp_threshold=-1.5,
                max_linear_apprx=3,
                device='cuda:0',
                always_correct_cross_execution = True,
                result_dir='recent_res', write_file=True, bound_improvement_ration=True,
                bound_ration_dir='bound_ration_final')
            ver.adptiveRaven(raven_args=args)
            eps += 0.005 

        net_names = [config.MNIST_CONV_SMALL]
        eps = 0.08
        for _ in range(1):
            args = ver.RavenArgs(raven_mode=RavenMode.UAP, dataset=Dataset.MNIST, net_names=net_names,
                count_per_prop=20, prop_count=40, eps=eps,
                threshold_execution=10, cross_executional_threshold=4, maximum_cross_execution_count=4, 
                baseline_iteration=20, refinement_iterations=20, unroll_layers = False, unroll_layer_count=3, 
                refine_intermediate_bounds =True, optimize_layers_count=2, 
                bounds_for_individual_refinement=True, dataloading_seed = 0,
                parallelize_executions=False,
                lp_threshold=-1.5,
                max_linear_apprx=3,
                device='cuda:0',
                always_correct_cross_execution = True,
                result_dir='recent_res', write_file=True, bound_improvement_ration=True,
                bound_ration_dir='bound_ration_final')
            ver.adptiveRaven(raven_args=args)
            eps += 0.005 

        net_names = [config.MNIST_CONV_PGD]
        eps = 0.1
        for _ in range(1):
            args = ver.RavenArgs(raven_mode=RavenMode.UAP, dataset=Dataset.MNIST, net_names=net_names,
                count_per_prop=20, prop_count=40, eps=eps,
                threshold_execution=10, cross_executional_threshold=4, maximum_cross_execution_count=4, 
                baseline_iteration=20, refinement_iterations=20, unroll_layers = False, unroll_layer_count=3, 
                refine_intermediate_bounds =True, optimize_layers_count=2, 
                bounds_for_individual_refinement=True, dataloading_seed = 0,
                parallelize_executions=False,
                lp_threshold=-1.5,
                max_linear_apprx=3,
                device='cuda:0',
                always_correct_cross_execution = True,
                result_dir='recent_res', write_file=True, bound_improvement_ration=True,
                bound_ration_dir='bound_ration_final')
            ver.adptiveRaven(raven_args=args)
            eps += 0.005 

    def test_mnist_citrus_bound_ratio(self):
        net_names = [config.MNIST_CONV_SMALL_CITRUS_1]
        eps = 0.15
        for _ in range(1):
            args = ver.RavenArgs(raven_mode=RavenMode.UAP, dataset=Dataset.MNIST, net_names=net_names,
                count_per_prop=20, prop_count=20, eps=eps,
                threshold_execution=6, cross_executional_threshold=5, maximum_cross_execution_count=4, 
                baseline_iteration=20, refinement_iterations=40, unroll_layers = False, unroll_layer_count=3, 
                refine_intermediate_bounds = True, optimize_layers_count=2, 
                bounds_for_individual_refinement=True, dataloading_seed = 0,
                parallelize_executions=False,
                lp_threshold=-1.5,
                max_linear_apprx=6,
                device='cuda:3',
                always_correct_cross_execution = False,
                result_dir='recent_res', write_file=True, bound_improvement_ration=False,
                bound_ration_dir='bound_ration_final')
            ver.adptiveRaven(raven_args=args)
            eps += 0.005

    def test_cifar_bound_ratio(self):
        # net_names = [config.CIFAR_CONV_SMALL]
        # eps = 1.0
        # for _ in range(1):
        #     args = ver.RavenArgs(raven_mode=RavenMode.UAP, dataset=Dataset.CIFAR10, net_names=net_names,
        #         count_per_prop=20, prop_count=20, eps=eps/255,
        #         threshold_execution=10, cross_executional_threshold=4, maximum_cross_execution_count=4, 
        #         baseline_iteration=20, refinement_iterations=20, unroll_layers = False, unroll_layer_count=3, 
        #         refine_intermediate_bounds =True, optimize_layers_count=2, 
        #         bounds_for_individual_refinement=True, dataloading_seed = 0,
        #         parallelize_executions=False,
        #         lp_threshold=-0.5,
        #         max_linear_apprx=3,
        #         device='cuda:0',
        #         always_correct_cross_execution = False,
        #         result_dir='recent_res', write_file=True, bound_improvement_ration=True,
        #         bound_ration_dir='bound_ration_final')
        #     ver.adptiveRaven(raven_args=args)
        #     eps += 0.5

        net_names = [config.CIFAR_CITRUS_2]
        eps = 2.0
        for _ in range(1):
            args = ver.RavenArgs(raven_mode=RavenMode.UAP, dataset=Dataset.CIFAR10, net_names=net_names,
                count_per_prop=1, prop_count=50, eps=eps/255,
                threshold_execution=10, cross_executional_threshold=4, maximum_cross_execution_count=4, 
                baseline_iteration=20, refinement_iterations=20, unroll_layers = False, unroll_layer_count=3, 
                refine_intermediate_bounds =True, optimize_layers_count=2, 
                bounds_for_individual_refinement=True, dataloading_seed = 0,
                parallelize_executions=False,
                lp_threshold=-0.5,
                max_linear_apprx=3,
                device='cuda:0',
                always_correct_cross_execution = False,
                result_dir='recent_res', write_file=True, bound_improvement_ration=False,
                bound_ration_dir='bound_ration_final')
            ver.adptiveRaven(raven_args=args)
            eps += 0.5

        # net_names = [config.CIFAR_CONV_SMALL_PGD]
        # eps = 2.0
        # for _ in range(12):
        #     args = ver.RavenArgs(raven_mode=RavenMode.UAP, dataset=Dataset.CIFAR10, net_names=net_names,
        #         count_per_prop=20, prop_count=20, eps=eps/255,
        #         threshold_execution=10, cross_executional_threshold=4, maximum_cross_execution_count=4, 
        #         baseline_iteration=20, refinement_iterations=20, unroll_layers = False, unroll_layer_count=3, 
        #         refine_intermediate_bounds =True, optimize_layers_count=2, 
        #         bounds_for_individual_refinement=True, dataloading_seed = 0,
        #         parallelize_executions=False,
        #         lp_threshold=-0.5,
        #         max_linear_apprx=3,
        #         device='cuda:0',
        #         always_correct_cross_execution = False,
        #         result_dir='icml_results_new', write_file=True, bound_improvement_ration=True,
        #         bound_ration_dir='bound_ration_final')
        #     ver.adptiveRaven(raven_args=args)
        #     eps += 0.5
