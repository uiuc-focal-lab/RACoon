import torch
from src.common import Dataset, RavenMode
from src.specLoader import get_specification, get_std
from src.netLoader import get_net
from src.adaptiveRavenBackend import AdaptiveRavenBackend
from src.adaptiveRavenResult import AdaptiveRavenResultList
from raven.src.config import mnist_data_transform

class RavenArgs:
    def __init__(self, raven_mode : RavenMode, dataset : Dataset, net_names,
                count_per_prop=None, prop_count=None, eps=None,
                threshold_execution=5, cross_executional_threshold=4, 
                maximum_cross_execution_count=3, baseline_iteration=10,
                refinement_iterations=30, unroll_layers = False, unroll_layer_count=3,
                optimize_layers_count = None, full_alpha=False,
                bounds_for_individual_refinement=True,
                always_correct_cross_execution=False,
                parallelize_executions = False, lp_threshold=None,
                max_linear_apprx=3,
                device=None,
                refine_intermediate_bounds = False, dataloading_seed = 0, 
                result_dir=None, write_file=True) -> None:
        self.raven_mode = raven_mode
        self.dataset = dataset
        self.net_names = net_names
        assert len(self.net_names) > 0
        if raven_mode in [RavenMode.UAP, RavenMode.UAP_TARGETED]:
            assert len(self.net_names) == 1
        self.count_per_prop = count_per_prop
        self.prop_count = prop_count
        self.eps = eps
        self.threshold_execution = threshold_execution
        self.cross_executional_threshold = cross_executional_threshold
        self.maximum_cross_execution_count = maximum_cross_execution_count
        self.baseline_iteration = baseline_iteration
        self.refinement_iterations = refinement_iterations
        self.bounds_for_individual_refinement=True 
        self.full_alpha = full_alpha
        self.unroll_layers = unroll_layers
        self.unroll_layer_count = unroll_layer_count
        self.always_correct_cross_execution = always_correct_cross_execution
        self.parallelize_executions = parallelize_executions
        self.refine_intermediate_bounds = refine_intermediate_bounds
        self.optimize_layers_count = optimize_layers_count
        self.lp_threshold = lp_threshold
        self.max_linear_apprx = max_linear_apprx
        self.dataloading_seed = dataloading_seed
        self.device = device
        self.result_dir = result_dir
        self.write_file = write_file

class Property:
    def __init__(self, inputs, labels, eps, constraint_matrices, lbs, ubs) -> None:
        self.inputs = inputs
        self.labels = labels
        self.eps = eps
        self.constraint_matrices = constraint_matrices
        self.lbs = lbs
        self.ubs = ubs

def adptiveRaven(raven_args : RavenArgs):
    nets = get_net(net_names = raven_args.net_names, dataset = raven_args.dataset)
    total_input_count = raven_args.prop_count * raven_args.count_per_prop
    images, labels, constraint_matrices, lbs, ubs = get_specification(dataset=raven_args.dataset,
                                                            raven_mode=raven_args.raven_mode, 
                                                            count=total_input_count, nets=nets, eps=raven_args.eps,
                                                            dataloading_seed=raven_args.dataloading_seed,
                                                            net_names=raven_args.net_names)
    assert len(raven_args.net_names) > 0
    assert images.shape[0] == raven_args.count_per_prop * raven_args.prop_count
    assert labels.shape[0] == raven_args.count_per_prop * raven_args.prop_count
    assert constraint_matrices.shape[0] == raven_args.count_per_prop * raven_args.prop_count

    result_list = AdaptiveRavenResultList(args=raven_args)
    data_transform = mnist_data_transform(dataset=raven_args.dataset, net_name=raven_args.net_names[0])

    print(f'net name {raven_args.net_names[0]} data transform {data_transform}')
    for i in range(raven_args.prop_count):
        start = i * raven_args.count_per_prop
        end = start + raven_args.count_per_prop
        prop_images, prop_labels, prop_constraint_matrices = images[start:end], labels[start:end], constraint_matrices[start:end]
        prop_lbs, prop_ubs = lbs[start:end], ubs[start:end]
        prop = Property(inputs=prop_images, labels=prop_labels, 
                        eps=raven_args.eps / get_std(dataset=raven_args.dataset, transform=data_transform),
                        constraint_matrices=prop_constraint_matrices, lbs=prop_lbs, ubs=prop_ubs)
        verifier = AdaptiveRavenBackend(prop=prop, nets=nets, args=raven_args)
        result = verifier.verify()
        result_list.add_res(res=result)
    
    result_list.analyze()
    

