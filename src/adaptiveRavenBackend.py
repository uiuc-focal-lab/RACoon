import torch
import numpy as np

from src.adaptiveRavenResult import Result, AdaptiveRavenResult
from raven.src.network_conversion_helper import get_pytorch_net
import raven.src.config as config
from auto_LiRPA.operators import BoundLinear, BoundConv
from auto_LiRPA import BoundedModule, PerturbationLpNorm, BoundedTensor
from src.common import RavenMode
from src.gurobi_certifier import RavenLPtransformer
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
        self.device = 'cuda:2' if torch.cuda.is_available else 'cpu'
        self.base_method = 'CROWN'
        self.optimized_method = 'CROWN-Optimized'
        self.individual_res = None
        self.baseline_res = None
        self.individual_refinement_res = None
        self.cross_executional_refinement_res = None
        self.final_res = None
        self.number_of_class = 10
        self.final_layer_weights = []
        self.final_layer_biases = []

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

    def shift_to_device(self, device):
        # print(f'device {device}')
        self.prop.inputs = self.prop.inputs.to(device)
        self.prop.labels = self.prop.labels.to(device)
        self.prop.constraint_matrices = self.prop.constraint_matrices.to(device)
        # print(f'input device {self.prop.inputs.device}')
        for i, model in enumerate(self.bounded_models):
            model = model.to(device) 
            # self.final_layer_weights[i] = self.final_layer_weights[i].to(device)
            # self.final_layer_biases[i].to(device)

    
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
            lA = A_dict[self.final_names[0]][self.input_names[0]]['lA']
            lbias = A_dict[self.final_names[0]][self.input_names[0]]['lbias']
            lA = torch.reshape(lA,(self.args.count_per_prop, self.number_of_class-1,-1))
        return lA, lbias, lower_bnd
    
    def get_coef_bias_with_refinement(self, override_device=None):
        if override_device is not None:
            self.shift_to_device(device=override_device)
        else:
            self.shift_to_device(device=self.device)

        bounded_images = BoundedTensor(self.prop.inputs, self.ptb)
        coef_dict = { self.final_names[0]: [self.input_names[0]]}
        for model in self.bounded_models:
            result = model.compute_bounds(x=(bounded_images,), method=self.optimized_method, C=self.prop.constraint_matrices,
                                           bound_upper=False, return_A=True, needed_A_dict=coef_dict, 
                                           multiple_execution=False, execution_count=None, ptb=self.ptb, 
                                           unperturbed_images = self.prop.inputs)
            lower_bnd, upper, A_dict = result
            lA = A_dict[self.final_names[0]][self.input_names[0]]['lA']
            lbias = A_dict[self.final_names[0]][self.input_names[0]]['lbias']
            lA = torch.reshape(lA,(self.args.count_per_prop, self.number_of_class-1,-1))
        return lA, lbias, lower_bnd

    def get_verified_percentages(self, lower_bnd):
        verified_accuracy = torch.sum(lower_bnd.detach().cpu().min(axis=1)[0] > 0).numpy() / self.args.count_per_prop * 100
        return verified_accuracy

    def get_baseline_res(self):
        start_time = time.time()
        lA, lbias, lower_bnd = self.get_coef_bias_baseline()
        individual_ceritified_accuracy = self.get_verified_percentages(lower_bnd=lower_bnd)
        individual_time = time.time() - start_time
        print(f'lower bound {lower_bnd.min(axis=1)[0]}')
        print(f'Individual certified accuracy {individual_ceritified_accuracy}')
        milp_verifier = RavenLPtransformer(eps=self.prop.eps, inputs=self.prop.inputs, batch_size=self.args.count_per_prop,
                                         roll_indices=None, lb_coef=lA, lb_bias=lbias, non_verified_indices=None,
                                         lb_penultimate_coef=None, lb_penultimate_bias=None, ub_penultimate_coef=None,
                                         ub_penultimate_bias=None, lb_penult=None, ub_penult=None,
                                         constraint_matrices=self.prop.constraint_matrices, disable_unrolling=True)
        baseline_accuracy = milp_verifier.formulate_constriants(final_weight=self.final_layer_weights[0],
                                                        final_bias=self.final_layer_biases[0]).solv_MILP() * 100
        print(f'Baseline certified accuracy {baseline_accuracy}')
        baseline_time = time.time() - start_time
        self.individual_res = Result(final_result=individual_ceritified_accuracy, final_time=individual_time)
        self.baseline_res = Result(final_result=baseline_accuracy, final_time=baseline_time)
        
        

    def get_refined_res(self):
        start_time = time.time()
        lA, lbias, lower_bnd = self.get_coef_bias_with_refinement()
        individual_refinement_accuracy = self.get_verified_percentages(lower_bnd=lower_bnd)
        individual_refinement_time = time.time() - start_time
        print(f'lower bound {lower_bnd.min(axis=1)[0]}')
        print(f'Individual refinement certified accuracy {individual_refinement_accuracy}')
        milp_verifier = RavenLPtransformer(eps=self.prop.eps, inputs=self.prop.inputs, batch_size=self.args.count_per_prop,
                                         roll_indices=None, lb_coef=lA, lb_bias=lbias, non_verified_indices=None,
                                         lb_penultimate_coef=None, lb_penultimate_bias=None, ub_penultimate_coef=None,
                                         ub_penultimate_bias=None, lb_penult=None, ub_penult=None,
                                         constraint_matrices=self.prop.constraint_matrices, disable_unrolling=True)
        final_accuracy = milp_verifier.formulate_constriants(final_weight=self.final_layer_weights[0],
                                                        final_bias=self.final_layer_biases[0]).solv_MILP() * 100
        print(f'Final certified accuracy {final_accuracy}')
        final_time = time.time() - start_time
        self.individual_refinement_res = Result(final_result=individual_refinement_accuracy, final_time=individual_refinement_time)
        self.baseline_res = Result(final_result=final_accuracy, final_time=final_time)

    def verify(self) -> AdaptiveRavenResult:
        self.initialize_models()
        if self.args.raven_mode != RavenMode.UAP:
            raise NotImplementedError(f'Currently {self.args.raven_mode} is not supported')
        assert len(self.bounded_models) == 1
        # Get the baseline results.
        self.get_baseline_res()
        # Get the final results.
        self.get_refined_res()