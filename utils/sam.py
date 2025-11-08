import torch
import torch.nn as nn


class SAM(torch.optim.Optimizer):
    def __init__(self, params, base_optimizer, rho=0.05, adaptive=False, **kwargs):
        assert rho >= 0.0, f"Invalid rho, should be non-negative: {rho}"

        defaults = dict(rho=rho, adaptive=adaptive, **kwargs)
        super(SAM, self).__init__(params, defaults)

        self.base_optimizer = base_optimizer(self.param_groups, **kwargs)
        self.param_groups = self.base_optimizer.param_groups
        self.defaults.update(self.base_optimizer.defaults)

    @torch.no_grad()
    def first_step(self, zero_grad=False):
        grad_norm = self._grad_norm()
        for group in self.param_groups:
            scale = group["rho"] / (grad_norm + 1e-12)

            for p in group["params"]:
                if p.grad is None:
                    continue
                self.state[p]["old_p"] = p.data.clone()
                e_w = (
                    (torch.pow(p, 2) if group["adaptive"] else 1.0)
                    * p.grad
                    * scale.to(p)
                )
                p.add_(e_w)  # climb to the local maximum "w + e(w)"

        if zero_grad:
            self.zero_grad()

    @torch.no_grad()
    def second_step(self, zero_grad=False):
        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is None:
                    continue
                p.data = self.state[p]["old_p"]  # get back to "w" from "w + e(w)"

        self.base_optimizer.step()  # do the actual "sharpness-aware" update

        if zero_grad:
            self.zero_grad()

    @torch.no_grad()
    def step(self, closure=None):
        assert (
            closure is not None
        ), "Sharpness Aware Minimization requires closure, but it was not provided"
        closure = torch.enable_grad()(
            closure
        )  # the closure should do a full forward-backward pass

        self.first_step(zero_grad=True)
        closure()
        self.second_step()

    def _grad_norm(self):
        shared_device = self.param_groups[0]["params"][
            0
        ].device  # put everything on the same device, in case of model parallelism
        norm = torch.norm(
            torch.stack(
                [
                    ((torch.abs(p) if group["adaptive"] else 1.0) * p.grad)
                    .norm(p=2)
                    .to(shared_device)
                    for group in self.param_groups
                    for p in group["params"]
                    if p.grad is not None
                ]
            ),
            p=2,
        )
        return norm

    def load_state_dict(self, state_dict):
        super().load_state_dict(state_dict)
        self.base_optimizer.param_groups = self.param_groups


class FriendlySAM(torch.optim.Optimizer):
    def __init__(self, params, base_optimizer, rho=0.05, sigma=1, lmbda=0.9, adaptive=False, **kwargs):
        assert rho >= 0.0, f"Invalid rho, should be non-negative: {rho}"

        defaults = dict(rho=rho, adaptive=adaptive, **kwargs)
        super(FriendlySAM, self).__init__(params, defaults)

        self.base_optimizer = base_optimizer(self.param_groups, **kwargs)
        self.param_groups = self.base_optimizer.param_groups
        self.defaults.update(self.base_optimizer.defaults)
        self.sigma = sigma
        self.lmbda = lmbda

    @torch.no_grad()
    def first_step(self, zero_grad=False):

        for group in self.param_groups:
            for p in group["params"]:      
                if p.grad is None: continue       
                grad = p.grad.clone()
                if not "momentum" in self.state[p]:
                    self.state[p]["momentum"] = grad
                else:
                    p.grad -= self.state[p]["momentum"] * self.sigma
                    self.state[p]["momentum"] = self.state[p]["momentum"] * self.lmbda + grad * (1 - self.lmbda)
            
        grad_norm = self._grad_norm()
        for group in self.param_groups:
            scale = group["rho"] / (grad_norm + 1e-12)

            for p in group["params"]:
                if p.grad is None: continue
                self.state[p]["old_p"] = p.data.clone()
                e_w = (torch.pow(p, 2) if group["adaptive"] else 1.0) * p.grad * scale.to(p)
                p.add_(e_w)  # climb to the local maximum "w + e(w)"

        if zero_grad: self.zero_grad()

    @torch.no_grad()
    def second_step(self, zero_grad=False):
        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is None: continue
                p.data = self.state[p]["old_p"]  # get back to "w" from "w + e(w)"

        self.base_optimizer.step()  # do the actual "sharpness-aware" update

        if zero_grad: self.zero_grad()

    @torch.no_grad()
    def step(self, closure=None):
        assert closure is not None, "Sharpness Aware Minimization requires closure, but it was not provided"
        closure = torch.enable_grad()(closure)  # the closure should do a full forward-backward pass

        self.first_step(zero_grad=True)
        closure()
        self.second_step()

    def _grad_norm(self):
        shared_device = self.param_groups[0]["params"][0].device  # put everything on the same device, in case of model parallelism
        norm = torch.norm(
                    torch.stack([
                        ((torch.abs(p) if group["adaptive"] else 1.0) * p.grad).norm(p=2).to(shared_device)
                        for group in self.param_groups for p in group["params"]
                        if p.grad is not None
                    ]),
                    p=2
               )
        return norm

    def load_state_dict(self, state_dict):
        super().load_state_dict(state_dict)
        self.base_optimizer.param_groups = self.param_groups


class FisherSAM(torch.optim.Optimizer):
    def __init__(self, params, base_optimizer, model, rho=0.05, keep_ratio=0.1, 
                 mask_update_freq=100, adaptive=False, **kwargs):
        assert rho >= 0.0, f"Invalid rho, should be non-negative: {rho}"
        assert 0.0 < keep_ratio <= 1.0, f"Invalid keep_ratio, should be in (0, 1]: {keep_ratio}"

        defaults = dict(rho=rho, adaptive=adaptive, **kwargs)
        super(FisherSAM, self).__init__(params, defaults)

        self.base_optimizer = base_optimizer(self.param_groups, **kwargs)
        self.param_groups = self.base_optimizer.param_groups
        self.defaults.update(self.base_optimizer.defaults)
        
        self.model = model
        self.keep_ratio = keep_ratio
        self.mask_update_freq = mask_update_freq
        self.mask = {}
        self.iteration = 0

    def init_mask(self):
        for name, param in self.model.named_parameters():
            self.mask[name] = torch.ones_like(param, dtype=torch.float32, requires_grad=False).to(param.device)
        
        self._remove_based_partial('bias')
        self._remove_based_partial('embed')
        self._remove_based_nntype(nn.BatchNorm1d)
        self._remove_based_nntype(nn.BatchNorm2d)

    def _remove_weight(self, name):
        if name in list(self.mask.keys()):
            #print(f'Removing `{name}` (size: {self.mask[name].shape}; params: {self.mask[name].numel()})')
            self.mask.pop(name)

    def _remove_based_nntype(self, nn_type):
        for name, module in self.model.named_modules():
            if isinstance(module, nn_type):
                self._remove_weight(name)
                self._remove_weight(name + '.weight')
                self._remove_weight(name + '.bias')

    def _remove_based_partial(self, partial_name):
        for name in list(self.mask.keys()):
            if partial_name in name:
                print(f'Removing `{name}` (size: {self.mask[name].shape}; params: {self.mask[name].numel()})')
                self.mask.pop(name)

    def set_fisher_mask(self):
        fisher_dict = {}
        for name, param in self.model.named_parameters():
            if name in self.mask:
                fisher_dict[name] = torch.zeros_like(param, requires_grad=False).to(param.device)

        for name, param in self.model.named_parameters():
            if name in self.mask and param.grad is not None:
                fisher_dict[name] += torch.square(param.grad).data
        
        self.model.zero_grad()
        
        param_shape = {}
        fisher_value = []
        all_param_size = 0
        
        for name, fisher_info in fisher_dict.items():
            if name in self.mask:
                param_shape[name] = fisher_info.shape
                fisher_value.append(fisher_info.view(-1))
                all_param_size += fisher_info.numel()
        
        fisher_value = torch.cat(fisher_value, 0)
        
        keep_num = int(all_param_size * self.keep_ratio)
        param_to_be_update = torch.sort(fisher_value, descending=True)[1][:keep_num]
        
        mask_position = torch.zeros_like(fisher_value, dtype=torch.float, requires_grad=False).to(fisher_value.device)
        mask_position[param_to_be_update] = 1

        start_idx = 0
        for name, shape in param_shape.items():
            end_idx = start_idx + torch.prod(torch.tensor(shape))
            self.mask[name] = mask_position[start_idx:end_idx].reshape(shape)
            self.mask[name].requires_grad = False
            start_idx = end_idx

    def mask_info(self):
        all_param = 0
        zero_param = 0
        nonzero_param = 0
        
        for name, mask_value in self.mask.items():
            all_param += mask_value.numel()
            nonzero_param += torch.sum(mask_value).item()
            zero_param += mask_value.numel() - torch.sum(mask_value).item()
        
        sparse_ratio = zero_param / float(all_param) if all_param > 0 else 0.0
        
        info = (f'Mask has {all_param/1024./1024.:.3f}MB params to choose, '
                f'{nonzero_param/1024./1024.:.3f}MB params active, '
                f'{zero_param/1024./1024.:.3f}MB params frozen, '
                f'sparse ratio: {sparse_ratio:.3f}')
        
        return info, all_param, nonzero_param, zero_param, sparse_ratio

    @torch.no_grad()
    def first_step(self, zero_grad=False):
        grad_norm = self._grad_norm()
        
        for group in self.param_groups:
            scale = group["rho"] / (grad_norm + 1e-7)
            
            for name, p in self.model.named_parameters():
                p.requires_grad = True
                if p.grad is None:
                    continue
                
                e_w = p.grad * scale.to(p)
                
                if name in self.mask:
                    e_w = e_w * self.mask[name].to(p.device)
                
                p.add_(e_w)
                self.state[p]["e_w"] = e_w

        if zero_grad:
            self.zero_grad()

    @torch.no_grad()
    def second_step(self, zero_grad=False):
        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is None:
                    continue

                if "e_w" in self.state[p]:
                    p.sub_(self.state[p]["e_w"])

        self.base_optimizer.step()

        if zero_grad:
            self.zero_grad()

    @torch.no_grad()
    def step(self, closure=None):
        assert closure is not None, "Sharpness Aware Minimization requires closure"
        closure = torch.enable_grad()(closure)

        closure()

        if self.iteration % self.mask_update_freq == 0:
            self.set_fisher_mask()
            info, _, _, _, _ = self.mask_info()
            #print(f"[Iteration {self.iteration}] {info}")
        
        self.iteration += 1

        self.first_step(zero_grad=True)
        
        closure()
        
        self.second_step()

    def _grad_norm(self):
        shared_device = self.param_groups[0]["params"][0].device
        norm = torch.norm(
            torch.stack([
                ((torch.abs(p) if group["adaptive"] else 1.0) * p.grad).norm(p=2).to(shared_device)
                for group in self.param_groups
                for p in group["params"]
                if p.grad is not None
            ]),
            p=2
        )
        return norm

    def load_state_dict(self, state_dict):
        super().load_state_dict(state_dict)
        self.base_optimizer.param_groups = self.param_groups