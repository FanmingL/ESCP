import torch
import copy
import time
import numpy as np

class ContrastiveLoss:
    def __init__(self, dim, device=torch.device('cpu')):
        self.device = device
        # self.W = torch.rand((dim, dim), requires_grad=True, device=device)
        self.W = torch.eye(dim, requires_grad=True, device=device)
        self.w_optim = torch.optim.Adam([self.W], lr=1e-2)
        self.loss_func = torch.nn.CrossEntropyLoss()

    def get_loss_meta(self, y, need_w_grad=False):
        if need_w_grad:
            proj_k = (self.W + self.W.t()).matmul(y.t())
        else:
            proj_k = (self.W + self.W.t()).detach().matmul(y.t())
        logits = y.matmul(proj_k)
        # print(logits.max(dim=1, keepdim=True).values)
        logits = logits - logits.max(dim=1, keepdim=True).values
        labels = torch.arange(logits.shape[0])
        # print(logits)
        loss = self.loss_func(logits, labels)
        return loss

    def get_loss(self, y):
        loss_w = self.get_loss_meta(y.detach(), True)
        self.w_optim.zero_grad()
        loss_w.backward()
        self.w_optim.step()
        return self.get_loss_meta(y)

    def __call__(self, *args, **kwargs):
        return self.get_loss(*args, **kwargs)


class constraint:
    def __init__(self, alg_name='dpp', w_consistency=1.0, w_diverse=1.0):
        self.alg_name = alg_name
        self.w_consistency = w_consistency
        self.w_diverse = w_diverse
        self.contrastive_loss = ContrastiveLoss() if alg_name == 'contrastive' else None

    @staticmethod
    def get_loss_dpp(y):
        y = y / torch.clamp_min(y.pow(2).mean(dim=1, keepdim=True).sqrt(), 1e-5)
        K = (y.matmul(y.t()) - 1).exp() + torch.eye(y.shape[0], device=y.device) * 1e-3
        loss = -torch.logdet(K)
        if torch.isnan(loss).any():
            print(K)
        return loss

    def get_loss_contrastive(self, y):
        pass

    def get_loss(self, predicted_env_vector, tasks, valid):
        tasks = tasks[..., 0, 0]
        all_tasks = torch.unique(tasks).detach().cpu().numpy().tolist()
        if len(all_tasks) <= 1:
            return None, None, None
        total_trasition_num = valid.sum()
        all_predicted_env_vectors = []
        all_valids = []
        mean_vector = []
        var_vector = []
        valid_num_list = []
        masks_list = []
        real_all_tasks = []

        for item in all_tasks:
            if item == 0:
                continue
            masks = tasks == item
            valid_it = valid[masks]
            if valid_it.sum() == 0:
                continue
            masks_list.append(masks)
            all_valids.append(valid_it)
            real_all_tasks.append(item)
        if len(all_tasks) <= 1:
            return None, None, None
        all_tasks = real_all_tasks
        for ind, item in enumerate(all_tasks):
            masks = masks_list[ind]
            valid_it = all_valids[ind]
            env_vector_it = predicted_env_vector[masks]
            all_predicted_env_vectors.append(env_vector_it)
            point_num = valid_it.sum()
            assert point_num > 0, 'trajectory should not be empty!!!!'
            # print(env_vector_it.shape, valid_it.shape)
            repre_it = (env_vector_it * valid_it).sum(1, keepdim=True).sum(0, keepdim=True) / point_num
            mean_vector.append(repre_it)
            var_it = ((env_vector_it - repre_it.detach()) * valid_it).pow(2).sum() / point_num / \
                     predicted_env_vector.shape[-1]
            var_vector.append(var_it)
            valid_num_list.append(point_num)
        ##### consistency loss
        consistency_loss = sum([a1 * a2 for a1, a2 in zip(var_vector, valid_num_list)]) / total_trasition_num
        ##### use DPP loss
        repres = [item.reshape(1, -1) for item in mean_vector]
        repre_tensor = torch.cat(repres, 0)
        if self.alg_name == 'dpp':
            diverse_loss = self.get_loss_dpp(repre_tensor)
        elif self.alg_name == 'contrastive':
            diverse_loss = self.contrastive_loss(repre_tensor)
        else:
            raise NotImplementedError(f'{self.alg_name} has not been implemented!!!')

        constraint_loss = self.w_consistency * consistency_loss + diverse_loss * self.w_diverse
        # print(consistency_loss, dpp_loss)

        return constraint_loss, consistency_loss, diverse_loss


def get_rbf_matrix(data, centers, alpha):
    out_shape = torch.Size([data.shape[0], centers.shape[0], data.shape[-1]])
    data = data.unsqueeze(1).expand(out_shape)
    centers = centers.unsqueeze(0).expand(out_shape)
    mtx = (-(centers - data).pow(2) * alpha).sum(dim=-1, keepdim=False).exp()
    # mtx = (-(centers - data).pow(2) * alpha).exp().mean(dim=-1, keepdim=False)
    # mtx = mtx.clamp_min(mtx.min().item() * 1)
    return mtx


def get_loss_dpp(y, kernel='rbf', rbf_radius=3000.0):
    # K = (y.matmul(y.t()) - 1).exp() + torch.eye(y.shape[0]) * 1e-3
    if kernel == 'rbf':
        K = get_rbf_matrix(y, y, alpha=rbf_radius) + torch.eye(y.shape[0], device=y.device) * 1e-5
    elif kernel == 'inner':
        # y = y / y.pow(2).sum(dim=-1, keepdim=True).sqrt()
        K = y.matmul(y.t()).exp()
        # K = torch.softmax(K, dim=0)
        K = K + torch.eye(y.shape[0], device=y.device) * 1e-4
        print(K)
        # print('k shape: ', K.shape, ', y_mtx shape: ', y_mtx.shape)
    else:
        assert False
    loss = -torch.logdet(K)
    # loss = -(y.pinverse().t().detach() * y).sum()
    return loss


def get_loss_cov(y):
    cov = (y - y.mean(dim=0, keepdim=True)).pow(2).mean()
    return -torch.log(cov + 1e-4)

class RMDMLoss:
    def __init__(self, tau=0.995, target_consistency_metric=-4.0, target_diverse_metric=None, max_env_len=40):
        self.mean_vector = {}
        self.tau = tau
        self.target_consistency_metric = target_consistency_metric
        self.target_diverse_metric = target_diverse_metric
        self.lst_tasks = []
        self.max_env_len = max_env_len
        self.current_env_mean = None
        self.history_env_mean = None

    def construct_loss(self, consistency_loss, diverse_loss, consis_w, diverse_w, std):
        consis_w_loss = None
        divers_w_loss = None
        if isinstance(consis_w, torch.Tensor):
            rmdm_loss_it = consis_w.detach() * consistency_loss + diverse_loss * diverse_w.detach()
            if std >= 1e-1:
                rmdm_loss_it = consis_w.detach() * consistency_loss
            # alpha_loss = (alpha[0] * (target - current).detach()).mean()

            if self.target_consistency_metric is not None:
                consis_w_loss = consis_w * ((self.target_consistency_metric - consistency_loss.detach()).detach().mean())
            if self.target_diverse_metric is not None:
                divers_w_loss = diverse_w * ((self.target_diverse_metric - diverse_loss.detach()).detach().mean())
                pass
        else:
            rmdm_loss_it = consis_w * consistency_loss + diverse_loss * diverse_w
            if std >= 1e-1:
                rmdm_loss_it = consis_w * consistency_loss
        return rmdm_loss_it, consis_w_loss, divers_w_loss

    def rmdm_loss(self, predicted_env_vector, tasks, valid, consis_w, diverse_w, need_all_repre=False, need_parameter_loss=False, rbf_radius=3000.0):
        tasks = torch.max(tasks[..., 0, 0], tasks[..., -1, 0])
        all_tasks = torch.unique(tasks).detach().cpu().numpy().tolist()
        if len(all_tasks) <= 1:
            print(f'current task num: {len(all_tasks)}, {all_tasks}')
            return None, None, None, 0
        total_trasition_num = valid.sum()
        all_predicted_env_vectors = []
        all_valids = []
        mean_vector = []
        var_vector = []
        valid_num_list = []
        masks_list = []
        real_all_tasks = []

        for item in all_tasks:
            if item == 0:
                continue
            masks = tasks == item
            valid_it = valid[masks]
            if valid_it.sum() == 0:
                continue
            masks_list.append(masks)
            all_valids.append(valid_it)
            real_all_tasks.append(item)
        if len(all_tasks) <= 1:
            print(f'current task num: {len(all_tasks)}, {all_tasks}')
            return None, None, None, 0
        # print(f'task num: {len(all_tasks)}, env_vector: {predicted_env_vector.shape}')
        all_tasks = real_all_tasks
        self.lst_tasks = copy.deepcopy(real_all_tasks)
        dpp_inner = []
        use_dpp_inner = False
        for ind, item in enumerate(all_tasks):
            masks = masks_list[ind]
            valid_it = all_valids[ind]
            env_vector_it = predicted_env_vector[masks]
            all_predicted_env_vectors.append(env_vector_it)
            point_num = valid_it.sum()
            assert point_num > 0, 'trajectory should not be empty!!!!'
            repre_it = (env_vector_it * valid_it).sum(1, keepdim=True).sum(0, keepdim=True) / point_num
            if item not in self.mean_vector:
                self.mean_vector[item] = repre_it.detach()
            else:
                self.mean_vector[item] = ((repre_it.detach() * (1-self.tau)) + self.mean_vector[item] * self.tau).detach()
            mean_vector.append(repre_it)
            var_it = ((env_vector_it - self.mean_vector[item]) * valid_it).pow(2).sum() / point_num / predicted_env_vector.shape[-1]
            var_vector.append(var_it)
            valid_num_list.append(point_num)
        ##### consistency loss
        var = sum([a1 * a2 for a1, a2 in zip(var_vector, valid_num_list)]) / total_trasition_num
        stds = var.sqrt()
        consistency_loss = stds  #  + 1e-4)
        if stds < 1e-3:
            consistency_loss = consistency_loss.detach()
        ##### use DPP loss
        repres = [item.reshape(1, -1) for item in mean_vector]
        for item in self.mean_vector:
            if item not in all_tasks:
                repres.append(self.mean_vector[item].reshape(1, -1))
        repre_tensor = torch.cat(repres, 0)
        dpp_loss = get_loss_dpp(repre_tensor, rbf_radius=rbf_radius)
        rmdm_loss_it, consis_w_loss, diverse_w_loss = self.construct_loss(consistency_loss, dpp_loss, consis_w, diverse_w, stds.item())
        # rmdm_loss_it = dpp_loss + consistency_loss
        # print(consistency_loss, dpp_loss)
        if need_parameter_loss:
            if need_all_repre:
                return rmdm_loss_it, consistency_loss, dpp_loss, len(all_tasks), consis_w_loss, diverse_w_loss, all_predicted_env_vectors, all_valids
            return rmdm_loss_it, consistency_loss, dpp_loss, len(all_tasks), consis_w_loss, diverse_w_loss

        if need_all_repre:
            return rmdm_loss_it, consistency_loss, dpp_loss, len(all_tasks), all_predicted_env_vectors, all_valids
        return rmdm_loss_it, consistency_loss, dpp_loss, len(all_tasks)

    def rmdm_loss_timing(self, predicted_env_vector, tasks, valid,
                         consis_w, diverse_w, need_all_repre=False,
                         need_parameter_loss=False, cum_time=[], rbf_radius=3000.0):
        if self.current_env_mean is None:
            self.current_env_mean = torch.zeros((self.max_env_len, 1, predicted_env_vector.shape[-1]), device=predicted_env_vector.device)
            self.history_env_mean = torch.zeros((self.max_env_len, 1, predicted_env_vector.shape[-1]), device=predicted_env_vector.device)
        tasks = tasks[..., -1, 0] # torch.max(tasks[..., 0, 0], )
        tasks_sorted, indices = torch.sort(tasks)
        tasks_sorted_np = tasks_sorted.detach().cpu().numpy().reshape((-1))
        task_ind_map = {}
        tasks_sorted_np_idx = np.where(np.diff(tasks_sorted_np))[0] + 1
        last_ind = 0
        for i, item in enumerate(tasks_sorted_np_idx):
            task_ind_map[tasks_sorted_np[item-1]] = [last_ind, item]
            last_ind = item
            if i == len(tasks_sorted_np_idx) - 1:
                task_ind_map[tasks_sorted_np[-1]] = [last_ind, len(tasks_sorted_np)]
        predicted_env_vector = predicted_env_vector[indices]
        # remove the invalid data
        if 0 in task_ind_map:
            predicted_env_vector = predicted_env_vector[task_ind_map[0][1]:]
            start_ind = task_ind_map[0][1]
            task_ind_map.pop(0)
            for k in task_ind_map:
                task_ind_map[k][0] -= start_ind
                task_ind_map[k][1] -= start_ind
        # finish preprocess the data
        # def update_cum_time(time_last, time_count, cum_time):
        #     # if len(cum_time) < time_count + 1:
        #     #     cum_time.append(time.time() - time_last)
        #     # else:
        #     #     cum_time[time_count] += time.time() - time_last
        #     # time_last = time.time()
        #     # time_count += 1
        #     return time_last, time_count
        if len(task_ind_map) <= 1:
            print(f'current task num: {len(task_ind_map)}, {task_ind_map}')
            return None, None, None, 0
        total_trasition_num = predicted_env_vector.shape[0]
        all_valids, mean_vector, valid_num_list, all_predicted_env_vectors = [], [], [], []
        real_all_tasks = sorted(list(task_ind_map.keys()))
        all_tasks, self.lst_tasks = real_all_tasks, real_all_tasks
        use_history_mean = True
        for ind, item in enumerate(all_tasks):
            env_vector_it = predicted_env_vector[task_ind_map[item][0]:task_ind_map[item][1]]
            if need_all_repre:
                all_predicted_env_vectors.append(env_vector_it)
            point_num = env_vector_it.shape[0]
            repre_it = env_vector_it.mean(dim=0, keepdim=True)
            if item not in self.mean_vector:
                with torch.no_grad():
                    self.history_env_mean[int(item-1)] = repre_it
            self.current_env_mean[int(item-1)] = repre_it
            mean_vector.append(repre_it)
            valid_num_list.append(point_num)
        valid_num_tensor = torch.from_numpy(np.array(valid_num_list)).to(device=valid.device,
                                                                         dtype=torch.get_default_dtype()).reshape((-1, 1, 1))
        task_set = set(all_tasks)
        with torch.no_grad():
            for k in self.mean_vector:
                if k not in task_set:
                    self.current_env_mean[int(k-1)] = self.history_env_mean[int(k-1)]
        self.current_env_mean = self.current_env_mean.detach()
        self.history_env_mean = self.history_env_mean * self.tau + (1-self.tau) * self.current_env_mean
        for item in all_tasks:
            if item not in self.mean_vector:
                self.mean_vector[item] = 1
        ##### use DPP loss
        repres = [item[0] for item in mean_vector]
        valid_repres_len = len(repres)
        for item in self.mean_vector:
            if item not in task_set:
                repres.append(self.history_env_mean[int(item-1)])
        repre_tensor = torch.cat(repres, 0)
        dpp_loss = get_loss_dpp(repre_tensor, rbf_radius=rbf_radius)
        ##### consistency loss
        # total minus outter
        if not use_history_mean:
            with torch.no_grad():
                total_mean = ((repre_tensor[:valid_repres_len] * valid_num_tensor).sum(dim=0, keepdim=True) / total_trasition_num)
            total_outter_var = ((repre_tensor[:valid_repres_len] - total_mean).pow(2) * valid_num_tensor).sum(dim=0, keepdim=True) / total_trasition_num
            total_var = (predicted_env_vector - total_mean).pow(2).mean(dim=0, keepdim=True)
            var = max(total_var.mean() - total_outter_var.mean(), 0)
        ######################
        # summation of inner
        else:
            total_var = 0
            for ind, item in enumerate(all_tasks):
                mean_vector = self.history_env_mean[int(item-1)]
                if need_all_repre:
                    env_vector_it = all_predicted_env_vectors[ind]
                else:
                    env_vector_it = predicted_env_vector[task_ind_map[item][0]:task_ind_map[item][1]]
                var_it = (env_vector_it - mean_vector.detach()).pow(2).sum(dim=0, keepdim=True).mean()
                total_var = total_var + var_it
            var = total_var / total_trasition_num
        #####################

        stds = var.sqrt()
        consistency_loss = stds  #  + 1e-4)
        if stds < 1e-3:
            consistency_loss = consistency_loss.detach()
        rmdm_loss_it, consis_w_loss, diverse_w_loss = self.construct_loss(consistency_loss, dpp_loss, consis_w,
                                                                          diverse_w, stds.item())
        # rmdm_loss_it = consistency_loss + dpp_loss
        if need_parameter_loss:
            if need_all_repre:
                return rmdm_loss_it, consistency_loss, dpp_loss, len(all_tasks), consis_w_loss, diverse_w_loss, all_predicted_env_vectors, all_valids
            return rmdm_loss_it, consistency_loss, dpp_loss, len(all_tasks), consis_w_loss, diverse_w_loss
        if need_all_repre:
            return rmdm_loss_it, consistency_loss, dpp_loss, len(all_tasks), all_predicted_env_vectors, all_valids
        return rmdm_loss_it, consistency_loss, dpp_loss, len(all_tasks)




