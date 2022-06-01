import yaml
from parameter.private_config import *

config = {"session_name": "run-all-8359y", "windows": []}
base_path = get_base_path()
docker_path = "/root/policy_adaptation"
path = docker_path
tb_port = 6006
docker_template = f'docker run --rm -it --shm-size 50gb --gpus all -v {base_path}:{docker_path} sanluosizhou/selfdl:ml -c '
docker_template = None
if docker_template is None or len(docker_template) == 0:
    path = base_path
template = 'export CUDA_VISIBLE_DEVICES={0} && cd {1} ' \
           '&& python main.py --env_name {3}  --rnn_fix_length {4}' \
           ' --seed {5}  --task_num {6} --max_iter_num {7} --varying_params  {8}  --test_task_num {9} --ep_dim {10}' \
           ' --name_suffix {11} --rbf_radius {12} --kernel_type {13} --diversity_loss_weight {14} '
seeds = [8]
GPUS = [0, 1]
envs = ['HalfCheetah-v2', 'Hopper-v2', 'Walker2d-v2', 'Ant-v2']
count_it = 0
algs = ['sac']
task_common_num = 40
test_task_num = 40
max_iter_num = 2000
rnn_fix_length = 16
ep_dim = 2
sac_mini_batch_size = 256
test_dof = False
diversity_loss_weight = 0.004
if test_dof:
    varying_params = [
        # ' gravity ',
        ' gravity dof_damping_1_dim ',
    ]
    kernel_type = "rbf"
    name_suffix = 'RMDM_REPLAY_DOF'
    rbf_kernel_size = 80
else:
    varying_params = [
        ' gravity ',
    ]
    kernel_type = "rbf_element_wise"
    name_suffix = 'RMDM_REPLAY'
    rbf_kernel_size = 3000
use_rmdm = True
stop_pg_for_ep = True
bottle_neck = True
imitate_update_interval = 50
for seed in seeds:
    for env in envs:
        panes_list = []
        for varying_param in varying_params:
            for alg in algs:
                    script_it = template.format(GPUS[count_it % len(GPUS)], path, alg, env, rnn_fix_length,
                                                seed, task_common_num, max_iter_num, varying_param, test_task_num,
                                                ep_dim, name_suffix, rbf_kernel_size, kernel_type, diversity_loss_weight)
                    if use_rmdm:
                        script_it += ' --use_rmdm '
                    if stop_pg_for_ep:
                        script_it += ' --stop_pg_for_ep '
                    if bottle_neck:
                        script_it += ' --bottle_neck '
                    if docker_template is not None and len(docker_template) > 0:
                        script_it = docker_template + '"{}"'.format(script_it)

                    print('{}-{}'.format(env, seed), ': ', script_it)

                    panes_list.append(script_it)
                    count_it = count_it + 1

        config["windows"].append({
            "window_name": "{}-{}".format(env, seed),
            "panes": panes_list,
            "layout": "tiled"
        })

yaml.dump(config, open("run_all.yaml", "w"), default_flow_style=False)
