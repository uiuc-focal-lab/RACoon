import torch
from src.common import Dataset, RavenMode
from src.specLoader import get_specification
from src.netLoader import get_net
from src.adaptiveRavenBackend import AdaptiveRavenBackend
from src.adaptiveRavenResult import AdaptiveRavenResultList

class RavenArgs:
    def __init__(self, raven_mode : RavenMode, dataset : Dataset, net_names,
                count_per_prop=None, prop_count=None, eps=None,
                threshold_execution=5, baseline_iteration=10,
                refinement_iterations=30, unroll_layers = False, unroll_layer_count=3, 
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
        self.baseline_iteration = baseline_iteration
        self.refinement_iterations = refinement_iterations
        self.unroll_layers = unroll_layers
        self.unroll_layer_count = unroll_layer_count
        self.refine_intermediate_bounds = refine_intermediate_bounds
        self.dataloading_seed = dataloading_seed
        self.result_dir = result_dir
        self.write_file = write_file

class Property:
    def __init__(self, inputs, labels, eps, constraint_matrices) -> None:
        self.inputs = inputs
        self.labels = labels
        self.eps = eps
        self.constraint_matrices = constraint_matrices

def adptiveRaven(raven_args : RavenArgs):
    nets = get_net(net_names = raven_args.net_names, dataset = raven_args.dataset)
    total_input_count = raven_args.prop_count * raven_args.count_per_prop
    images, labels, constraint_matrices = get_specification(dataset=raven_args.dataset,
                                                            raven_mode=raven_args.raven_mode, 
                                                            count=total_input_count, nets=nets,
                                                            dataloading_seed=raven_args.dataloading_seed)
    assert images.shape[0] == raven_args.count_per_prop * raven_args.prop_count
    assert labels.shape[0] == raven_args.count_per_prop * raven_args.prop_count
    assert constraint_matrices.shape[0] == raven_args.count_per_prop * raven_args.prop_count

    result_list = AdaptiveRavenResultList()

    for i in range(raven_args.prop_count):
        start = i * raven_args.count_per_prop
        end = start + raven_args.count_per_prop
        prop_images, prop_labels, prop_constraint_matrices = images[start:end], labels[start:end], constraint_matrices[start:end]
        prop = Property(inputs=prop_images, labels=prop_labels, eps=raven_args.eps, constraint_matrices=prop_constraint_matrices)
        verifier = AdaptiveRavenBackend(prop=prop, nets=nets, args=raven_args)
        result = verifier.verify()
        result_list.add_res(res=result)
    

