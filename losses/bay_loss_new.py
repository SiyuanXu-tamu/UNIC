from torch.nn.modules import Module
import torch

class Bay_Loss(Module):
    def __init__(self, use_background, device):
        super(Bay_Loss, self).__init__()
        self.device = device
        self.use_bg = use_background

    def forward(self, prob_list, target_list, pre_density , epoch):  ##target_list = prior
        loss = 0
        #print(pre_density.shape)
        for idx, prob in enumerate(prob_list):  # iterative through each sample
            if prob is None:  # image contains no annotation points
                pre_count = torch.sum(pre_density[idx])
                target = torch.zeros((1,), dtype=torch.float32, device=self.device)
                
                loss += torch.sum(torch.abs(target - pre_count))
                
            else:
                N = len(prob)
                n_sample = target_list[0].shape[1]
                if self.use_bg:
                    #print(target_list[idx].shape) ##15 1024
                    target = torch.zeros((N, n_sample), dtype=torch.float32, device=self.device)
                    target[:-1] = target_list[idx][:-2]#*10
                    

                    pre_count = pre_density[idx].reshape((1, -1)).unsqueeze(0) * prob

                    tt = target_list[idx][-1].view((1, -1))
                    
                    
                     
                    
                else:
                    target = target_list[idx][:-1]
                    
                    pre_count = pre_density[idx].view((1, -1)).unsqueeze(0) * prob
                    

            

            

            loss += torch.nn.MSELoss(reduction = 'sum')(target*10, pre_count[0])



            
        loss = loss / len(prob_list)
        return loss



