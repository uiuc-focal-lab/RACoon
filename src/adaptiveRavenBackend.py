import torch
import numpy as np

from src.adaptiveRavenResult import Result, AdaptiveRavenResult
from raven.src.network_conversion_helper import get_pytorch_net
import raven.src.config as config
from auto_LiRPA.operators import BoundLinear, BoundConv
from auto_LiRPA import BoundedModule, PerturbationLpNorm, BoundedTensor
from src.common import RavenMode
from src.gurobi_certifier import RavenLPtransformer
import src.util as util
import time

class AdaptiveRavenBackend:
    def __init__(self, prop, nets, args) -> None:
        self.prop = prop
        self.nets = nets
        self.args = args
        # Converted torch models.
        self.torch_models = []
        self.bounded_models = []
        self.ptb = None
        self.input_names = []
        self.final_names = []
        self.devices = config.GPU_DEVICE_LIST
        self.device = 'cuda:3' if torch.cuda.is_available else 'cpu'
        self.base_method = 'CROWN'
        self.optimized_method = 'CROWN-Optimized'
        self.individual_res = None
        self.individual_verified = None
        self.baseline_res = None
        self.individual_refinement_res = None
        self.cross_executional_refinement_res = None
        self.final_res = None
        self.baseline_lowerbound = None
        self.number_of_class = 10
        self.final_layer_weights = []
        self.final_layer_biases = []
        self.refinement_indices = None
        self.cross_executional_indices = None
        # self.indices_for_2 = None
        # self.indices_for_3 = None
        # self.indices_for_4 = None
        self.indices_for = {}
        self.lb_coef_dict = {}
        self.lb_bias_dict = {}

    def populate_names(self):
        for model in self.bounded_models:
            i = 0
            last_name = None
            for node_name, node in model._modules.items():
                if i == 0:
                    self.input_names.append(node_name)
                i += 1
                if type(node) in [BoundLinear, BoundConv]:
                    last_name = node_name
            assert last_name is not None
            self.final_names.append(node_name)


    def initialize_models(self):
        for net in self.nets:
            self.torch_models.append(get_pytorch_net(model=net, remove_last_layer=False, all_linear=False))
            self.torch_models[-1] = self.torch_models[-1].to(self.device)
            self.final_layer_weights.append(net[-1].weight.to(self.device))
            self.final_layer_biases.append(net[-1].bias.to(self.device))
            self.bounded_models.append(BoundedModule(self.torch_models[-1], (self.prop.inputs)))
        self.ptb = PerturbationLpNorm(norm = np.inf, eps = self.prop.eps)
        self.populate_names()

    def shift_to_device(self, device, indices=None):
        # print(f'device {device}')
        self.prop.inputs = self.prop.inputs.to(device)
        self.prop.labels = self.prop.labels.to(device)
        self.prop.constraint_matrices = self.prop.constraint_matrices.to(device)
        if indices is not None:
            indices = indices.to(device)
        # print(f'input device {self.prop.inputs.device}')
        for i, model in enumerate(self.bounded_models):
            model = model.to(device) 
            # self.final_layer_weights[i] = self.final_layer_weights[i].to(device)
            # self.final_layer_biases[i].to(device)
    
    @torch.no_grad()
    def get_coef_bias_baseline(self, override_device=None):
        if override_device is not None:
            self.shift_to_device(device=override_device)
        else:
            self.shift_to_device(device=self.device)

        bounded_images = BoundedTensor(self.prop.inputs, self.ptb)
        coef_dict = { self.final_names[0]: [self.input_names[0]]}
        for model in self.bounded_models:
            result = model.compute_bounds(x=(bounded_images,), method=self.base_method, C=self.prop.constraint_matrices,
                                           bound_upper=False, return_A=True, needed_A_dict=coef_dict, 
                                           multiple_execution=False, execution_count=None, ptb=self.ptb, 
                                           unperturbed_images = self.prop.inputs)
            lower_bnd, upper, A_dict = result
            self.baseline_lowerbound = lower_bnd
            lA = A_dict[self.final_names[0]][self.input_names[0]]['lA']
            lbias = A_dict[self.final_names[0]][self.input_names[0]]['lbias']
            lA = torch.reshape(lA,(self.args.count_per_prop, self.number_of_class-1,-1))
        return lA, lbias, lower_bnd
    

    def select_indices(self, lower_bound, threshold=None):
        min_logit_diff = lower_bound.detach().cpu().min(axis=1)[0]
        min_logit_diff_sorted = min_logit_diff.sort(descending=True)
        # print(f'sorted logit diff {min_logit_diff_sorted[0]}')
        indices = min_logit_diff_sorted[1][(min_logit_diff_sorted[0] < 0.0)]
        length = indices.shape[0]
        if threshold is not None:
            indices = indices[:min(length, threshold)]
        # print(f'filtered min_indices {min_logit_diff[indices]}')
        return indices

    def populate_cross_indices(self, cross_executional_indices, count):
        if count > 4:
            raise ValueError(f'Execution number of {count} is not supported.')
        indices = cross_executional_indices[:min(len(cross_executional_indices), self.args.cross_executional_threshold)]
        final_indices = util.generate_indices(indices=indices, threshold=self.args.threshold_execution, count=count)
        return final_indices

    def run_refinement(self, indices, device, multiple_execution=False, execution_count=None, iteration=None):
        self.shift_to_device(device=device, indices=indices)
        filtered_inputs = self.prop.inputs[indices]
        bounded_images = BoundedTensor(filtered_inputs, self.ptb)
        filtered_constraint_matrices = self.prop.constraint_matrices[indices]
        coef_dict = {self.final_names[0]: [self.input_names[0]]}
        for model in self.bounded_models:
            result = model.compute_bounds(x=(bounded_images,), method=self.optimized_method, C=filtered_constraint_matrices,
                                           bound_upper=False, return_A=True, needed_A_dict=coef_dict, 
                                           multiple_execution=multiple_execution, execution_count=execution_count, ptb=self.ptb, 
                                           unperturbed_images = filtered_inputs, iteration=iteration)
            lower_bnd, _, A_dict = result
            lA = A_dict[self.final_names[0]][self.input_names[0]]['lA']
            lbias = A_dict[self.final_names[0]][self.input_names[0]]['lbias']
            lA = torch.reshape(lA,(filtered_inputs.shape[0], self.number_of_class-1,-1))
        
        self.populate_coef_and_bias(indices=indices, lb_coef=lA, lb_bias=lbias)
        return lA, lbias, lower_bnd      

    def run_cross_executional_refinement(self, count):
        if count not in self.indices_for.keys() or self.indices_for[count] is None:
            return None, None, None
        # print(f'Refinement indices {self.indices_for[count]}')
        return self.run_refinement(indices=self.indices_for[count], device=self.device,
                                    multiple_execution=True, execution_count=count, iteration=self.args.refinement_iterations)
        

    def cross_executional_refinement(self, cross_executional_indices):
        length = cross_executional_indices.shape[0]
        indices = cross_executional_indices.detach().cpu().numpy()
        if length > 1 and 2 <= self.args.maximum_cross_execution_count:
            self.indices_for[2] = self.populate_cross_indices(cross_executional_indices=indices, count=2)
            lA, lbias, lower_bnd = self.run_cross_executional_refinement(count=2)
            # if lower_bnd is not None:
            #     print(f'count 2 lower bound {lower_bnd.detach().cpu().min(axis=1)[0]}')
        
        if length > 2 and 3 <= self.args.maximum_cross_execution_count:
            self.indices_for[3] = self.populate_cross_indices(cross_executional_indices=indices, count=3)
            lA, lbias, lower_bnd = self.run_cross_executional_refinement(count=3)
            # if lower_bnd is not None:
            #     print(f'count 3 lower bound {lower_bnd.detach().cpu().min(axis=1)[0]}')
        
        if length > 3 and 4 <= self.args.maximum_cross_execution_count:
            self.indices_for[4] = self.populate_cross_indices(cross_executional_indices=indices, count=4)
            lA, lbias, lower_bnd = self.run_cross_executional_refinement(count=4)
            # if lower_bnd is not None:
            #     print(f'count 4 lower bound {lower_bnd.detach().cpu().min(axis=1)[0]}')

    
    def populate_coef_and_bias(self, indices, lb_coef, lb_bias):
        assert len(indices) == lb_coef.shape[0]
        assert len(indices) == lb_bias.shape[0]
        lb_coef = lb_coef.detach()
        lb_bias = lb_bias.detach()
        for i, ind in enumerate(indices):
            if ind not in self.lb_bias_dict.keys():
                self.lb_coef_dict[ind] = []
                self.lb_bias_dict[ind] = []
            self.lb_bias_dict[ind].append(lb_bias[i])
            self.lb_coef_dict[ind].append(lb_coef[i])


    def get_coef_bias_with_refinement(self, override_device=None):
        self.refinement_indices = self.select_indices(lower_bound=self.baseline_lowerbound, threshold=self.args.threshold_execution)
        if override_device is not None:
            self.shift_to_device(device=override_device)
        else:
            self.shift_to_device(device=self.device)
        lA, lbias, lower_bnd = self.run_refinement(indices=self.refinement_indices, device=self.device, iteration=self.args.baseline_iteration)
        print(f'individual refinement lower bound {lower_bnd.detach().cpu().min(axis=1)[0]}')
        
        self.cross_executional_indices = self.select_indices(lower_bound=lower_bnd, threshold=self.args.cross_executional_threshold)
        self.cross_executional_indices = self.refinement_indices[self.cross_executional_indices]
        self.cross_executional_refinement(cross_executional_indices=self.cross_executional_indices)

        return lA, lbias, lower_bnd

    def get_verified_count(self, lower_bnd):
        verified_count = torch.sum(lower_bnd.detach().cpu().min(axis=1)[0] > 0).numpy() 
        return verified_count

    def get_baseline_res(self):
        start_time = time.time()
        lA, lbias, lower_bnd = self.get_coef_bias_baseline()
        lA, lbias, lower_bnd = lA.detach(), lbias.detach(), lower_bnd.detach() 
        self.individual_verified = self.get_verified_count(lower_bnd=lower_bnd)
        individual_ceritified_accuracy = self.individual_verified / self.args.count_per_prop * 100
        individual_time = time.time() - start_time
        print(f'lower bound {lower_bnd.min(axis=1)[0]}')
        print(f'Individual certified accuracy {individual_ceritified_accuracy}')
        all_indices = [i for i in range(self.prop.inputs.shape[0])]
        self.populate_coef_and_bias(indices=all_indices, lb_coef=lA, lb_bias=lbias)
        milp_verifier = RavenLPtransformer(eps=self.prop.eps, inputs=self.prop.inputs, batch_size=self.args.count_per_prop,
                                         roll_indices=None, lb_coef=lA, lb_bias=lbias,
                                         lb_coef_dict=self.lb_coef_dict, lb_bias_dict=self.lb_bias_dict, non_verified_indices=None,
                                         lb_penultimate_coef=None, lb_penultimate_bias=None, ub_penultimate_coef=None,
                                         ub_penultimate_bias=None, lb_penult=None, ub_penult=None,
                                         constraint_matrices=self.prop.constraint_matrices, disable_unrolling=True)
        baseline_accuracy = milp_verifier.formulate_constriants_from_dict(final_weight=self.final_layer_weights[0],
                                                        final_bias=self.final_layer_biases[0]).solv_MILP() / self.args.count_per_prop * 100
        print(f'Baseline certified accuracy {baseline_accuracy}')
        baseline_time = time.time() - start_time
        self.individual_res = Result(final_result=individual_ceritified_accuracy, final_time=individual_time)
        self.baseline_res = Result(final_result=baseline_accuracy, final_time=baseline_time)
        # Populate the coeficients and biases.
        
    def get_post_refinement_unverified_indices(self, refined_lower_bound):
        all_cross_execution_indices = self.select_indices(lower_bound=refined_lower_bound.detach().cpu(), threshold=None)
        all_cross_execution_indices = all_cross_execution_indices.detach().cpu()
        refine_indices = self.refinement_indices.detach().cpu()
        all_cross_execution_indices = refine_indices[all_cross_execution_indices]
        all_unverified = self.select_indices(lower_bound=self.baseline_lowerbound.detach().cpu(), threshold=None)
        final_indices = []
        for i in all_unverified:
            if i not in refine_indices:
                final_indices.append(i)
            elif i in all_cross_execution_indices:
                final_indices.append(i)
        return torch.tensor(final_indices, device='cpu')

    def get_refined_res(self):
        torch.cuda.empty_cache()
        start_time = time.time()
        lA, lbias, lower_bnd = self.get_coef_bias_with_refinement()
        lA, lbias, lower_bnd = lA.detach(), lbias.detach(), lower_bnd.detach()
        individual_refinement_count = self.get_verified_count(lower_bnd=lower_bnd) + self.individual_verified
        individual_refinement_accuracy = individual_refinement_count / self.args.count_per_prop * 100
        individual_refinement_time = time.time() - start_time
        print(f'lower bound {lower_bnd.min(axis=1)[0]}')
        print(f'Individual refinement certified accuracy {individual_refinement_accuracy}')
        milp_verifier = RavenLPtransformer(eps=self.prop.eps, inputs=self.prop.inputs, batch_size=self.args.count_per_prop,
                                         roll_indices=None, lb_coef=lA, lb_bias=lbias, lb_coef_dict=self.lb_coef_dict, lb_bias_dict=self.lb_bias_dict,
                                         non_verified_indices=self.get_post_refinement_unverified_indices(refined_lower_bound=lower_bnd),
                                         lb_penultimate_coef=None, lb_penultimate_bias=None, ub_penultimate_coef=None,
                                         ub_penultimate_bias=None, lb_penult=None, ub_penult=None,
                                         constraint_matrices=self.prop.constraint_matrices, disable_unrolling=True)
        final_count = milp_verifier.formulate_constriants_from_dict(final_weight=self.final_layer_weights[0],
                                                        final_bias=self.final_layer_biases[0]).solv_MILP()
        print(f'final count {final_count}')
        final_count += individual_refinement_count
        final_accuracy = final_count / self.args.count_per_prop * 100.0
        print(f'Final certified accuracy {final_accuracy}')
        final_time = time.time() - start_time
        self.individual_refinement_res = Result(final_result=individual_refinement_accuracy, final_time=individual_refinement_time)
        self.final_res = Result(final_result=final_accuracy, final_time=final_time)

    def verify(self) -> AdaptiveRavenResult:
        self.initialize_models()
        if self.args.raven_mode != RavenMode.UAP:
            raise NotImplementedError(f'Currently {self.args.raven_mode} is not supported')
        assert len(self.bounded_models) == 1
        # Get the baseline results.
        self.get_baseline_res()
        # Get the final results.
        self.get_refined_res()