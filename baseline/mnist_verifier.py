import torch
import numpy as np
import random

device = 'cuda:3' if torch.cuda.is_available else 'cpu'
from auto_LiRPA import BoundedModule, PerturbationLpNorm, BoundedTensor
from auto_LiRPA.utils import get_spec_matrix
from cert_util import min_correct_with_eps, load_data, DeltaWrapper
from auto_LiRPA.operators import BoundLinear, BoundConv

from model_defs import mnist_cnn_4layer,mnist_conv_small,mnist_conv_big
import sys
sys.path.append('/home/debangshu/uap-robustness/')
sys.path.insert(0, '../')
import src.util as util
from gurobi_certifier import UAPLPtransformer as Verifier


def extract_penultimate_layer(bounded_model):
    prev_name = None
    curr_name = None
    for node_name, node in bounded_model._modules.items():

        if type(node) in [BoundLinear, BoundConv]:
            if curr_name is not None:
                prev_name = curr_name
            curr_name = node_name
    return prev_name
    
def bounded_results(new_image, eps, C, bounded_model, eval_num, delta, num_cls):
    ptb = PerturbationLpNorm(norm = np.inf, eps = eps)
    # print(bounded_model)
    # bounded_delta = BoundedTensor(delta, ptb)
    bounded_images = BoundedTensor(new_image, ptb)
    final_name = bounded_model.final_name
    input_name = '/input.1' 
    BASE_METHOD = 'CROWN'
    OPTIMIZED_METHOD = 'CROWN-Optimized'
    bounded_model.set_input_node_name(input_node_name=input_name)
    penulitmate_layer_name = extract_penultimate_layer(bounded_model=bounded_model)
    if penulitmate_layer_name is not None:
        coef_dict = { final_name: [input_name], penulitmate_layer_name: [input_name]}
    else:
        coef_dict = { final_name: [input_name]}
    result = bounded_model.compute_bounds(
        x=(bounded_images,), method=OPTIMIZED_METHOD, C=C, bound_upper=False,
        return_A=True, 
        needed_A_dict=coef_dict, multiple_execution=True, execution_count=1, ptb=ptb, unperturbed_images = new_image)
    lower, upper, A_dict = result
    lA = A_dict[final_name][input_name]['lA']
    lbias = A_dict[final_name][input_name]['lbias']
    lA_penult = A_dict[penulitmate_layer_name][input_name]['lA']
    uA_penult = A_dict[penulitmate_layer_name][input_name]['uA']
    lbias_penult = A_dict[penulitmate_layer_name][input_name]['lbias']
    ubias_penult = A_dict[penulitmate_layer_name][input_name]['ubias']
    lower_penult = bounded_model[penulitmate_layer_name].lower
    upper_penult = bounded_model[penulitmate_layer_name].upper
    lb_bias_penult = lower_penult - ptb.concretize(new_image, lA_penult, sign=-1)
    ub_bias_penult = upper_penult - ptb.concretize(new_image, uA_penult, sign=1)
    lb = lower - ptb.concretize(new_image, lA, sign=-1)
    # ub = upper - ptb.concretize(new_image, uA, sign=1)

    lA = torch.reshape(lA,(eval_num, num_cls-1,-1))
    return lA, lbias, lower, [lA_penult, lbias_penult, uA_penult, ubias_penult], [lower_penult, upper_penult] 


def get_unrolling_indices(result, theshold=10):
    min_logit_diff = result.detach().cpu().min(axis=1)[0]
    min_logit_diff = min_logit_diff.sort(descending=True)
    print(f'sorted logit diff {min_logit_diff[0]}')
    indices = min_logit_diff[1][(min_logit_diff[0] < 0.0)]
    length = indices.shape[0]
    roll_indices = indices[:min(length, theshold)]
    return roll_indices, indices


#   (/input.1): BoundInput(name="/input.1")
#   (/1): BoundParams(name="/1")
#   (/2): BoundParams(name="/2")
#   (/3): BoundParams(name="/3")
#   (/4): BoundParams(name="/4")
#   (/5): BoundParams(name="/5")
#   (/6): BoundParams(name="/6")
#   (/7): BoundParams(name="/7")
#   (/8): BoundParams(name="/8")
#   (/input): BoundConv(name="/input")
#   (/input.4): BoundRelu(name="/input.4")
#   (/input.8): BoundConv(name="/input.8")
#   (/12): BoundRelu(name="/12")
#   (/13): BoundFlatten(name="/13")
#   (/input.12): BoundLinear(name="/input.12")
#   (/15): BoundRelu(name="/15")
#   (/16): BoundLinear(name="/16")




def main():
    # Setting parameters.
    my_seed = 2232
    torch.cuda.empty_cache()
    torch.manual_seed(my_seed)
    random.seed(my_seed)
    np.random.seed(my_seed)
    eval_num = 20
    num_cls = 10
    model_name = "mnist_conv_small" 
    net = eval(model_name)()
    net.load_state_dict(torch.load('./'+model_name+'.pth'))

    # Extract the final layer weight and bias
    final_layer_weight, final_layer_bias = net[-1].weight, net[-1].bias

    # Loading data
    new_image, new_label = load_data(num_imgs=eval_num, random=True, dataset='MNIST')
    # new_image, new_label = new_image[[1, 3, 5]], new_label[[1, 3, 5]]
    # eval_num = 3
    new_image = new_image.to(device)
    C = get_spec_matrix(new_image,new_label.long(), 10)

    # Model conversion and eps
    eps = 0.045
    delta = torch.zeros_like(new_image[0]).unsqueeze(0) + eps
    dummy_input = (new_image)
    model = net.to(device)
    bounded_model = BoundedModule(model, dummy_input)
    bounded_model.eval()
    # final_name = bounded_model.final_name

    # new_image_temp = new_image
    # C_temp = C


    alpha, beta, result, penult_coef, penult_bound = bounded_results(new_image, eps, C, bounded_model, eval_num, delta, num_cls)
    samp_ACC = torch.sum(result.detach().cpu().min(axis=1)[0] > 0).numpy()  
    print('Samp-wise Cert-ACC: {}%'.format(samp_ACC/ eval_num * 100.0))

    roll_indices, non_verified_indices = get_unrolling_indices(result=result, theshold=10)


    ver = Verifier(eps=eps, inputs=new_image, batch_size=eval_num, roll_indices=roll_indices, 
                 lb_coef=alpha, lb_bias=beta, non_verified_indices=non_verified_indices,
                 lb_penultimate_coef=penult_coef[0], lb_penultimate_bias=penult_coef[1], 
                 ub_penultimate_coef=penult_coef[2], ub_penultimate_bias=penult_coef[3],
                 lb_penult=penult_bound[0], ub_penult=penult_bound[1], constraint_matrices=C,
                 disable_unrolling=True)
    cert_ACC = ver.formulate_constriants(final_weight=final_layer_weight, final_bias=final_layer_bias).solv_MILP()
    # print('UP-based LB: {}%'.format(cert_ACC))
    print('UP-based Cert-ACC: {}%'.format(((samp_ACC + cert_ACC * non_verified_indices.shape[0]) / eval_num) * 100.0))

if __name__ == "__main__":
    main()