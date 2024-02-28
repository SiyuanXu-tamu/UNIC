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
                    
                    #bg_map = target_list[idx][-1:].unsqueeze(0) ## [1,1,1024]
                    #print('bg_map', bg_map.shape, pre_density[idx].view((1, -1)).unsqueeze(0).shape, prob.shape)
                    
                    #pre_density_ = pre_density[idx].view((2, -1))
                    #density1 = pre_density_[0:1]
                    #density2 = pre_density_[1:]
                    #print(density1.shape, density2.shape) ##[1 1024]
                    
                    #pre_count = (density1.unsqueeze(0) + density2.unsqueeze(0)) * prob.unsqueeze(0)
                    #print(pre_count.shape)  #[1 26 1024]
                    #pre_count[0][-1] = density1[0]*prob[-1]
                    
                    
                    #
                    #print(prob.shape)  [26 1024]
                    #pre_count = (pre_density[idx].view((1, -1)).unsqueeze(0) + bg_map)  * prob
                    #pre_count = pre_density[idx].view((1, -1)).unsqueeze(0) * prob
                    #print(pre_density[idx,:,::2,::2].shape)
                    #pre_count = pre_density[idx,:,::2,::2].reshape((1, -1)).unsqueeze(0) * prob
                    pre_count = pre_density[idx].reshape((1, -1)).unsqueeze(0) * prob
                    #pre_count1 = pre_density[idx].view((1, -1)).unsqueeze(0) * prob
                    #pre_count = pre_density[idx].unsqueeze(0).unsqueeze(0) * prob
                    #print(pre_density.shape)
                    tt = target_list[idx][-1].view((1, -1))
                    
                    
                     
                    
                else:
                    target = target_list[idx][:-1]
                    
                    pre_count = pre_density[idx].view((1, -1)).unsqueeze(0) * prob
                    
                    
                    
                    
                #change to no sum
                #pre_count = torch.sum(pre_density[idx].view((1, -1)) * prob, dim=1)  # flatten into vector###view:reshape
                #pre_count = pre_density[idx].view((1, -1)).unsqueeze(0) * prob ##pre_density[idx]
                #pre_count = pre_density[idx].unsqueeze(0) * prob ##pre_density[idx].view((1, -1))
        
            #change to no sum
            #print(target.shape, pre_count.shape) # torch.Size([47, 1024]) torch.Size([1, 47, 1024])
            #loss += torch.sum(torch.abs(target[:-1] - pre_count[0][:-1]))
            #loss += torch.sum(torch.abs(target - pre_count[0]))
            #loss += torch.sum(torch.abs(torch.sum(target[:-1], dim = 1) - torch.sum(pre_count[0][:-1], dim = 1)))
            #loss += torch.sum(torch.abs(torch.sum(target[-1:], dim = 1) - torch.sum(pre_count[0][-1:], dim = 1)))
            #loss += torch.sum(torch.abs(torch.sum(target, dim = 1) - torch.sum(pre_count[0], dim = 1)))
             
            #loss += torch.nn.L1Loss(reduction = 'sum')(tt*100, pre_count)
            #loss += torch.nn.MSELoss(reduction = 'sum')(tt, pre_count)
            
            if True:
                p = (2000-epoch)/2000*0.5+1
            
                if epoch < 3000:
                    loss += torch.nn.MSELoss(reduction = 'sum')(target*10, pre_count[0])
                    #loss += torch.nn.MSELoss(reduction = 'sum')(target[:-1]*10, pre_count[0][:-1])*1
                    #loss += torch.nn.MSELoss(reduction = 'sum')(target[-1:]*10, pre_count[0][-1:])*2
                    #loss += torch.sum(torch.abs(torch.sum(target[-1]*10, dim = 1) - torch.sum(pre_count[0][-1], dim = 1)))*
                    #loss += torch.nn.MSELoss(reduction = 'sum')(target*10, pre_count[0])
                    
                    #loss += torch.sum(torch.abs(torch.sum(target*10, dim = 1) - torch.sum(pre_count[0], dim = 1)))
                    #loss +=  torch.sum(torch.abs(torch.sum(target*10, dim = 1) - torch.sum(pre_count[0], dim = 1)))*p
                    
                    #loss += torch.nn.KLDivLoss(reduction = 'sum')(torch.log(torch.squeeze(pre_count[0])), torch.squeeze(target))
                    
                else:
                    #loss += torch.nn.MSELoss(reduction = 'sum')(target*10, pre_count[0])*p 
                    loss += torch.nn.MSELoss(reduction = 'sum')(target*10, pre_count[0])*p + torch.sum(torch.abs(torch.sum(target*10, dim = 1) - torch.sum(pre_count[0], dim = 1)))*0.2
                    #loss += torch.sum(torch.abs(torch.sum(target*10, dim = 1) - torch.sum(pre_count[0], dim = 1)))*p
            
            else:
                #loss += torch.sum(torch.abs(torch.sum(target*10, dim = 1) - torch.sum(pre_count[0], dim = 1)))#+ 0.1 * (torch.sum(target*10) - torch.sum(pre_count[0]))
                #loss += torch.sum(torch.abs(torch.sum(target*10, dim = 1) - torch.sum(pre_count[0], dim = 1)))
                
                loss += torch.nn.MSELoss(reduction = 'sum')(tt*10, pre_count) #+ 0.1 * (torch.sum(target*10) - torch.sum(pre_count[0]))
                #loss += torch.nn.MSELoss(reduction = 'sum')(tt, pre_count)
            
            #loss += torch.sum(torch.abs(torch.sum(target, dim = 1) - torch.sum(pre_count[0], dim = 1)))
            #print(target.shape, pre_count.shape)
            #loss += torch.sum(torch.abs(torch.sum(target) - torch.sum(pre_count)))
            
            #loss += 1.0 * torch.sum(density2)
            
            
            #loss += torch.sum(torch.abs(torch.sum(target, dim = 1) - torch.sum(pre_count[0], dim = 1)))
            #
            
        loss = loss / len(prob_list)
        return loss



