class IntermediateInfos(object):
    def __init__(self):
        self.infos = {
            'task_acc':[],
            'pred_classes':[],
            'ys':[],
            'reps':[],
            'pred_reps':[],
            'lowest_img_infos':[]
        }
    def add_infos(self, info, info_name):
        if info_name in list(self.infos.keys()):
            self.infos[info_name].append(info)
        else:
            raise ValueError("Wrong kind of information")
        
    def get_infos(self, info_name):
        if info_name in list(self.infos.keys()):
            return self.infos[info_name]
        else:
            raise ValueError("Wrong kind of information")
        
    def __len__(self, info_name='task_acc'):
        if info_name in list(self.infos.keys()):
            return len(self.infos[info_name])
        else:
            raise ValueError("Wrong kind of information")