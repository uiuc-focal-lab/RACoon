import torch

def convert_tensor(tuple_list):
    t = torch.tensor(tuple_list)
    return t.T.reshape(-1)

def generate_indices(indices, threshold, count):
    entries = threshold // count
    tuple_list = []
    if entries <= 0:
        return None
    if count == 2:
        for i, x  in enumerate(indices):
            for j in range(i+1, len(indices)):
                tuple_list.append((x, indices[j]))
                if len(tuple_list) >= entries:
                    # print(f'original indices {indices}')
                    # print(f'tuple list {tuple_list} cross ex indices {convert_tensor(tuple_list=tuple_list)}')
                    return convert_tensor(tuple_list=tuple_list), tuple_list
    elif count == 3:
        for i, x  in enumerate(indices):
            for j in range(i+1, len(indices)):
                for k in range(j+1, len(indices)):
                    tuple_list.append((x, indices[j], indices[k]))
                    if len(tuple_list) >= entries:
                        return convert_tensor(tuple_list=tuple_list), tuple_list
    elif count == 4:
        for i, x  in enumerate(indices):
            for j in range(i+1, len(indices)):
                for k in range(j+1, len(indices)):
                    for m in range(k+1, len(indices)):
                        tuple_list.append((x, indices[j], indices[k], indices[m]))
                        if len(tuple_list) >= entries:
                            return convert_tensor(tuple_list=tuple_list), tuple_list
    else:
        raise ValueError(f"We don't support cross executions of {count}")
    
    if len(tuple_list) > 0:
        return convert_tensor(tuple_list=tuple_list), tuple_list
    else:
        return None 