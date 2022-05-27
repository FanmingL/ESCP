import torch
import copy
import time
import numpy as np


class ContrastiveLoss:
    def __init__(self, c_dim, max_env_len=40, ep=None):
        self.max_env_len = max_env_len
        self.query_ep = ep
        self.c_dim = c_dim
        self.W = torch.rand((c_dim, c_dim), requires_grad=True)
        self.w_optim = torch.optim.Adam([self.W], lr=1e-1)
        self.loss_func = torch.nn.CrossEntropyLoss()

    def to(self, *arg, **kwargs):
        self.query_ep.to(*arg, **kwargs)
        self.W = self.W.to(*arg, **kwargs)

    def get_query_tensor(self, state, last_action):
        hidden = self.query_ep.make_init_state(state.shape[0], device=state.device)
        ep, h, full_hidden = self.query_ep.meta_forward(torch.cat((state, last_action), dim=-1),
                                                        hidden, require_full_hidden=True)
        return ep

    def get_loss_meta(self, y_key, y_query, need_w_grad=False):
        if need_w_grad:
            proj_k = self.W.matmul(y_key.t())
        else:
            proj_k = self.W.detach().matmul(y_key.t())
        # proj_k = 30 * torch.eye((self.W + self.W.t()).shape[0]).detach().matmul(y.t())
        # proj_k = y.t()
        logits = y_query.matmul(proj_k)
        # print(logits.max(dim=1, keepdim=True).values)
        logits = logits - logits.max(dim=1, keepdim=True).values
        # logits = get_rbf_matrix(y, y, alpha=1)
        labels = torch.arange(logits.shape[0]).to(device=y_key.device)
        # print(logits)
        loss = self.loss_func(logits, labels)
        return loss

    def update_ep(self, origin_ep):
        self.query_ep.copy_weight_from(origin_ep, tau=0.99)

    def contrastive_loss(self, predicted_env_vector, predicted_env_vector_query, tasks):
        tasks = tasks[..., -1, 0]  # torch.max(tasks[..., 0, 0], )
        tasks_sorted, indices = torch.sort(tasks)
        tasks_sorted_np = tasks_sorted.detach().cpu().numpy().reshape((-1))
        task_ind_map = {}
        tasks_sorted_np_idx = np.where(np.diff(tasks_sorted_np))[0] + 1
        last_ind = 0
        for i, item in enumerate(tasks_sorted_np_idx):
            task_ind_map[tasks_sorted_np[item - 1]] = [last_ind, item]
            last_ind = item
            if i == len(tasks_sorted_np_idx) - 1:
                task_ind_map[tasks_sorted_np[-1]] = [last_ind, len(tasks_sorted_np)]
        predicted_env_vector = predicted_env_vector[indices]
        predicted_env_vector_query = predicted_env_vector_query[indices]
        if 0 in task_ind_map:
            predicted_env_vector = predicted_env_vector[task_ind_map[0][1]:]
            predicted_env_vector_query = predicted_env_vector_query[task_ind_map[0][1]:]
            start_ind = task_ind_map[0][1]
            task_ind_map.pop(0)
            for k in task_ind_map:
                task_ind_map[k][0] -= start_ind
                task_ind_map[k][1] -= start_ind
        all_queries_ind = []
        all_key_ind = []
        all_tasks = sorted(list(task_ind_map.keys()))
        for ind, item in enumerate(all_tasks):
            all_queries_ind.append(task_ind_map[item][0])
            all_key_ind.append(task_ind_map[item][1]-1)
        queries = predicted_env_vector_query[all_queries_ind, 0]
        key = predicted_env_vector[all_key_ind, 0]
        loss_w = self.get_loss_meta(y_key=key.detach(), y_query=queries.detach(), need_w_grad=True)
        self.w_optim.zero_grad()
        loss_w.backward()
        self.w_optim.step()
        return self.get_loss_meta(y_key=key, y_query=queries.detach(), need_w_grad=False)



