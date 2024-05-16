import torch 

def get_net_format(net_name):
    net_format = None
    if 'pt' in net_name:
        net_format = 'pt'
    if 'onnx' in net_name:
        net_format = 'onnx'
    return net_format


def get_net_helper(net_name, dataset):
    net_format = get_net_format(net_name)
    if net_format == 'pt':
        # Load the model
        with torch.no_grad():
            net_torch = get_torch_net(net_name, dataset)
            net = parse.parse_torch_layers(net_torch)

    elif net_format == 'onnx':
        net_onnx = onnx.load(net_name)
        net = parse.parse_onnx_layers(net_onnx)
    else:
        raise ValueError("Unsupported net format!")

    net.net_name = net_name
    net.net_format = net_format
    return net