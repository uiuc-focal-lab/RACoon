from src.common import Dataset
class Result:
    def __init__(self, final_result, final_time,
                result_trace=None, time_trace=None) -> None:
        self.final_result = final_result
        self.final_time = final_time
        self.result_trace = result_trace
        self.time_trace = time_trace

class AdaptiveRavenResult:
    def __init__(self, individual_res : Result, baseline_res : Result, 
                 individual_refinement_res : Result, 
                 individual_refinement_milp_res : Result,
                 cross_executional_refinement_res : Result, 
                 final_res : Result) -> None:
        self.individual_res = individual_res
        self.baseline_res = baseline_res
        self.individual_refinement_res = individual_refinement_res
        self.individual_refinement_milp_res = individual_refinement_milp_res
        self.cross_executional_refinement_res = cross_executional_refinement_res
        self.final_res = final_res

class AdaptiveRavenResultList:
    def __init__(self, args) -> None:
        self.res_list = []
        self.args = args
    
    def add_res(self, res : AdaptiveRavenResult):
        self.res_list.append(res)
    
    def get_file(self):
        eps = self.args.eps if self.args.dataset == Dataset.MNIST else self.args.eps * 255
        filename = self.args.result_dir + '/' + f'{self.args.net_names[0]}_{self.args.count_per_prop}_{eps}.dat'
        file = open(filename, 'a+')
        return file

    def analyze(self):
        individual_acc = 0
        baseline_acc = 0
        individual_refinement_acc = 0
        individual_refinement_milp_acc = 0
        cross_ex_acc = 0
        final_acc = 0

        individual_time = 0
        baseline_time = 0
        individual_refinement_time = 0
        individual_refinement_milp_time = 0
        cross_ex_time = 0
        final_time = 0
        count = 0
        
        def populate(res : Result, acc, time):
            if res.final_result is not None:
                acc += res.final_result
            if res.final_time is not None:
                time += res.final_time
            return acc, time

        for res in self.res_list:
            individual_acc, individual_time = populate(res.individual_res, individual_acc, individual_time)
            baseline_acc, baseline_time = populate(res.baseline_res, baseline_acc, baseline_time)
            individual_refinement_acc, individual_refinement_time = populate(res.individual_refinement_res, 
                                                                             individual_refinement_acc, individual_refinement_time)
            
            individual_refinement_milp_acc, individual_refinement_milp_time = populate(res.individual_refinement_milp_res, 
                                                                             individual_refinement_milp_acc, 
                                                                             individual_refinement_milp_time)
            cross_ex_acc, cross_ex_time = populate(res.cross_executional_refinement_res, cross_ex_acc, cross_ex_time)
            final_acc, final_time = populate(res.final_res, final_acc, final_time)
        assert len(self.res_list) > 0
        scale = 1/ len(self.res_list)
        if self.args.write_file:
            file = self.get_file()
            file.write(f'individual acc {individual_acc*scale}\n')
            file.write(f'baseline acc {baseline_acc*scale}\n')
            file.write(f'IndivRefine acc {individual_refinement_acc*scale}\n')
            file.write(f'IndivRefineMILP acc {individual_refinement_milp_acc*scale}\n')            
            file.write(f'CrossEx acc {cross_ex_acc*scale}\n')
            file.write(f'final acc {final_acc*scale}\n')

            file.write(f'individual time {individual_time*scale}\n')
            file.write(f'baseline time {baseline_time*scale}\n')
            file.write(f'IndivRefine time {individual_refinement_time*scale}\n')
            file.write(f'IndivRefineMILP time {individual_refinement_milp_time*scale}\n')
            file.write(f'CrossEx time {cross_ex_time*scale}\n')        
            file.write(f'final time {final_time*scale}\n')

            file.close()