class Result:
    def __init__(self, final_result, final_time,
                result_trace=None, time_trace=None) -> None:
        self.final_result = final_result
        self.final_time = final_time
        self.result_trace = result_trace
        self.time_trace = time_trace

class AdaptiveRavenResult:
    def __init__(self, individual_res : Result, baseline_res : Result, 
                 individual_refinement_res : Result, cross_executional_refinement_res : Result, 
                 final_res : Result) -> None:
        self.individual_res = individual_res
        self.baseline_res = baseline_res
        self.individual_refinement_res = individual_refinement_res
        self.cross_executional_refinement_res = cross_executional_refinement_res
        self.final_res = final_res

class AdaptiveRavenResultList:
    def __init__(self) -> None:
        self.res_list = []
    
    def add_res(self, res : AdaptiveRavenResult):
        self.res_list.append(res)
