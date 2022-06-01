import os, sys
import argparse
import json
import socket
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from parameter.private_config import *
from datetime import datetime

class Parameter:
    def __init__(self, config_path=None, debug=False, information=None):
        self.base_path = self.get_base_path()
        self.debug = debug
        self.experiment_target = EXPERIMENT_TARGET
        self.DEFAULT_CONFIGS = global_configs()
        self.arg_names = []
        self.host_name = 'localhost'
        self.ip = '127.0.0.1'
        self.exec_time = datetime.now().strftime('%Y-%m-%d_%H:%M:%S')
        self.commit_id = self.get_commit_id()
        self.log_func = None
        self.json_name = 'parameter.json'
        self.txt_name = 'parameter.txt'
        self.information = information
        if config_path:
            self.config_path = config_path
        else:
            self.info('use default config path')
            self.config_path = osp.join(get_base_path(), 'parameter')
        self.info(f'json path is {os.path.join(self.config_path, self.json_name)}, '
                  f'txt path is {os.path.join(self.config_path, self.txt_name)}')
        if config_path:
            self.load_config()
        else:
            self.args = self.parse()
            self.apply_vars(self.args)

    def set_log_func(self, log_func):
        self.log_func = log_func

    def info(self, info):
        if self.log_func is not None:
            self.log_func(info)
        else:
            print(info)

    @staticmethod
    def get_base_path():
        return get_base_path()

    def set_config_path(self, config_path):
        self.config_path = config_path

    @staticmethod
    def important_configs():
        res = ['env_name', 'use_true_parameter', 'use_rmdm',
               'use_uposi', 'share_ep', "rnn_fix_length",
               'uniform_sample', 'enhance_ep', 'bottle_neck', 'stop_pg_for_ep', "ep_dim", 'seed']
        return res

    def apply_vars(self, args):
        for name in self.arg_names:
            setattr(self, name, getattr(args, name))

    def make_dict(self):
        res = {}
        for name in self.arg_names:
            res[name] = getattr(self, name)
        res['description'] = self.experiment_target
        res['exec_time'] = self.exec_time
        res['commit_id'] = self.commit_id
        return res

    def parse(self):
        parser = argparse.ArgumentParser(description=EXPERIMENT_TARGET)

        self.env_name = "Hopper-v2"
        parser.add_argument('--env_name', default=self.env_name, metavar='G',
                            help='name of the environment to run')
        self.register_param('env_name')

        self.model_path = ""
        parser.add_argument('--model_path', metavar='G',
                            help='path of pre-trained model')
        self.register_param("model_path")

        self.render = False
        parser.add_argument('--render', action='store_true', default=self.render,
                            help='render the environment')
        self.register_param("render")

        self.log_std = -1.5
        parser.add_argument('--log_std', type=float, default=self.log_std, metavar='G',
                            help='log std for the policy (default: -0.0)')
        self.register_param('log_std')

        self.gamma = 0.99
        parser.add_argument('--gamma', type=float, default=self.gamma, metavar='G',
                            help='discount factor (default: 0.99)')
        self.register_param('gamma')

        self.learning_rate = 3e-4
        parser.add_argument('--learning_rate', type=float, default=self.learning_rate, metavar='G',
                            help='learning rate (default: 3e-4)')
        self.register_param('learning_rate')

        self.value_learning_rate = 1e-3
        parser.add_argument('--value_learning_rate', type=float, default=self.value_learning_rate, metavar='G',
                            help='learning rate (default: 1e-3)')
        self.register_param('value_learning_rate')

        self.num_threads = 4
        parser.add_argument('--num_threads', type=int, default=self.num_threads, metavar='N',
                            help='number of threads for agent (default: 1)')
        self.register_param('num_threads')

        self.seed = 1
        parser.add_argument('--seed', type=int, default=self.seed, metavar='N',
                            help='random seed (default: 1)')
        self.register_param('seed')

        self.random_num = 4000
        parser.add_argument('--random_num', type=int, default=self.random_num, metavar='N',
                            help='sample random_num fully random samples,')
        self.register_param('random_num')

        self.start_train_num = 20000
        parser.add_argument('--start_train_num', type=int, default=self.start_train_num, metavar='N',
                            help='after reach start_train_num, training start')
        self.register_param('start_train_num')

        self.test_sample_num = 4000
        parser.add_argument('--test_sample_num', type=int, default=self.test_sample_num, metavar='N',
                            help='sample num in test phase')
        self.register_param('test_sample_num')

        self.sac_update_time = 1000
        parser.add_argument('--sac_update_time', type=int, default=self.sac_update_time, metavar='N',
                            help='update time after sampling a batch data')
        self.register_param('sac_update_time')

        self.sac_replay_size = 1e6
        parser.add_argument('--sac_replay_size', type=int, default=self.sac_replay_size, metavar='N',
                            help='update time after sampling a batch data')
        self.register_param('sac_replay_size')

        self.min_batch_size = 1200
        parser.add_argument('--min_batch_size', type=int, default=self.min_batch_size, metavar='N',
                            help='minimal sample number per iteration')
        self.register_param('min_batch_size')

        if FC_MODE:
            self.sac_mini_batch_size = 256
            parser.add_argument('--sac_mini_batch_size', type=int, default=self.sac_mini_batch_size, metavar='N',
                                help='update time after sampling a batch data')
            self.register_param('sac_mini_batch_size')

            self.sac_inner_iter_num = 1
            parser.add_argument('--sac_inner_iter_num', type=int, default=self.sac_inner_iter_num, metavar='N',
                                help='after sample several trajectories from replay buffer, '
                                     'sac_inner_iter_num mini-batch will be sampled from the batch, '
                                     'and model will be optimized for sac_inner_iter_num times.')
            self.register_param('sac_inner_iter_num')
        else:
            self.sac_mini_batch_size = 256
            parser.add_argument('--sac_mini_batch_size', type=int, default=self.sac_mini_batch_size, metavar='N',
                                help='sac_mini_batch_size trajectories will be sampled from the replay buffer.')
            self.register_param('sac_mini_batch_size')

            self.sac_inner_iter_num = 1
            parser.add_argument('--sac_inner_iter_num', type=int, default=self.sac_inner_iter_num, metavar='N',
                                help='after sample several trajectories from replay buffer, '
                                     'sac_inner_iter_num mini-batch will be sampled from the batch, '
                                     'and model will be optimized for sac_inner_iter_num times.')
            self.register_param('sac_inner_iter_num')

        self.rnn_sample_max_batch_size = 3e5
        parser.add_argument('--rnn_sample_max_batch_size', type=int, default=self.rnn_sample_max_batch_size, metavar='N',
                            help='max point num sampled from replay buffer per time')
        self.register_param('rnn_sample_max_batch_size')

        self.rnn_slice_num = 16
        parser.add_argument('--rnn_slice_num', type=int, default=self.rnn_slice_num, metavar='N',
                            help='gradient clip steps')
        self.register_param('rnn_slice_num')

        self.sac_tau = 0.995
        parser.add_argument('--sac_tau', type=float, default=self.sac_tau, metavar='N',
                            help='ratio of coping value net to target value net')
        self.register_param('sac_tau')

        self.sac_alpha = 0.2
        parser.add_argument('--sac_alpha', type=float, default=self.sac_alpha, metavar='N',
                            help='sac temperature coefficient')
        self.register_param('sac_alpha')

        self.reward_scale = 1.0
        parser.add_argument('--reward_scale', type=float, default=self.reward_scale, metavar='N',
                            help='sac temperature coefficient')
        self.register_param('reward_scale')

        self.max_iter_num = 10000
        parser.add_argument('--max_iter_num', type=int, default=self.max_iter_num, metavar='N',
                            help='maximal number of main iterations (default: 500)')
        self.register_param('max_iter_num')

        self.save_model_interval = 5
        parser.add_argument('--save_model_interval', type=int, default=self.save_model_interval, metavar='N',
                            help="interval between saving model (default: 5, means don't save)")
        self.register_param('save_model_interval')

        self.std_learnable = 1
        parser.add_argument('--std_learnable', type=int, default=self.std_learnable, metavar='N',
                            help="standard dev can be learned")
        self.register_param('std_learnable')

        self.update_interval = 1
        parser.add_argument('--update_interval', type=int, default=self.update_interval, metavar='N',
                            help="standard dev can be learned")
        self.register_param('update_interval')

        self.ep_pretrain_path_suffix = 'None'   # '-use_rmdm-rnn_len_32-ep_dim_2-1-debug'
        parser.add_argument('--ep_pretrain_path_suffix', type=str, default=self.ep_pretrain_path_suffix, metavar='N',
                            help="environment probing pretrain model path")
        self.register_param('ep_pretrain_path_suffix')

        self.name_suffix = 'None'  # '-use_rmdm-rnn_len_32-ep_dim_2-1-debug'
        parser.add_argument('--name_suffix', type=str, default=self.name_suffix, metavar='N',
                            help="name suffix of the experiment")
        self.register_param('name_suffix')

        self.ep_apply_tau = 0.99
        parser.add_argument('--ep_apply_tau', type=float, default=self.ep_apply_tau, metavar='N',
                            help="tau used to apply ep")
        self.register_param('ep_apply_tau')

        self.target_entropy_ratio = 1.5
        parser.add_argument('--target_entropy_ratio', type=float, default=self.target_entropy_ratio, metavar='N',
                            help="target entropy")
        self.register_param('target_entropy_ratio')

        self.history_length = 0
        parser.add_argument('--history_length', type=int, default=self.history_length, metavar='N',
                            help="interval between saving model (default: 0, means don't save)")
        self.register_param('history_length')

        self.task_num = 0
        parser.add_argument('--task_num', type=int, default=self.task_num, metavar='N',
                            help="interval between saving model (default: 0, means don't save)")
        self.register_param('task_num')

        self.test_task_num = 0
        parser.add_argument('--test_task_num', type=int, default=self.test_task_num, metavar='N',
                            help="number of tasks for testing")
        self.register_param('test_task_num')

        self.use_true_parameter = False
        parser.add_argument('--use_true_parameter', action='store_true')
        self.register_param("use_true_parameter")

        self.bottle_neck = False
        parser.add_argument('--bottle_neck', action='store_true')
        self.register_param("bottle_neck")

        self.transition_learn_aux = False
        parser.add_argument('--transition_learn_aux', action='store_true')
        self.register_param("transition_learn_aux")

        self.bottle_sigma = 1e-2
        parser.add_argument('--bottle_sigma', type=float, default=self.bottle_sigma, metavar='N',
                            help="std of the noise injected to ep while inference (information bottleneck)")
        self.register_param('bottle_sigma')

        self.l2_norm_for_ep = 0.0
        parser.add_argument('--l2_norm_for_ep', type=float, default=self.l2_norm_for_ep, metavar='N',
                            help="L2 norm added to EP module")
        self.register_param('l2_norm_for_ep')

        self.policy_max_gradient = 10
        parser.add_argument('--policy_max_gradient', type=float, default=self.policy_max_gradient, metavar='N',
                            help="maximum gradient of policy")
        self.register_param('policy_max_gradient')


        self.use_rmdm = False
        parser.add_argument('--use_rmdm', action='store_true',
                            help="use Relational Matrix Determinant Maximization or not")
        self.register_param('use_rmdm')

        self.use_uposi = False
        parser.add_argument('--use_uposi', action='store_true')
        self.register_param('use_uposi')

        self.uniform_sample = False
        parser.add_argument('--uniform_sample', action='store_true')
        self.register_param('uniform_sample')

        self.share_ep = False
        parser.add_argument('--share_ep', action='store_true')
        self.register_param('share_ep')

        self.enhance_ep = False
        parser.add_argument('--enhance_ep', action='store_true')
        self.register_param('enhance_ep')

        self.stop_pg_for_ep = False
        parser.add_argument('--stop_pg_for_ep', action='store_true')
        self.register_param('stop_pg_for_ep')

        self.use_contrastive = False
        parser.add_argument('--use_contrastive', action='store_true')
        self.register_param('use_contrastive')

        self.rmdm_update_interval = -1
        parser.add_argument('--rmdm_update_interval', type=int, default=self.rmdm_update_interval, metavar='N',
                            help="update interval of rmdm")
        self.register_param('rmdm_update_interval')

        self.rnn_fix_length = 0
        parser.add_argument('--rnn_fix_length', type=int, default=self.rnn_fix_length, metavar='N',
                            help="fix the rnn memory length to rnn_fix_length")
        self.register_param('rnn_fix_length')


        self.minimal_repre_rp_size = 1e5
        parser.add_argument('--minimal_repre_rp_size', type=float, default=self.minimal_repre_rp_size, metavar='N',
                            help="after minimal_repre_rp_size, start training EP module")
        self.register_param('minimal_repre_rp_size')


        # self.ep_start_num = 150000
        self.ep_start_num = 0
        parser.add_argument('--ep_start_num', type=int, default=self.ep_start_num, metavar='N',
                            help="only when the size of the replay buffer is larger than ep_start_num"
                                 ", ep can be learned")
        self.register_param('ep_start_num')

        self.kernel_type = 'rbf_element_wise'
        parser.add_argument('--kernel_type', default=self.kernel_type, metavar='G',
                            help='kernel type for DPP loss computing (rbf/rbf_element_wise/inner)')
        self.register_param('kernel_type')

        self.rmdm_ratio = 1.0
        parser.add_argument('--rmdm_ratio', type=float, default=self.rmdm_ratio, metavar='N',
                            help="gradient ratio of rmdm")
        self.register_param('rmdm_ratio')

        self.test_variable = 1.0
        parser.add_argument('--test_variable', type=float, default=self.test_variable, metavar='N',
                            help="variable for testing variable")
        self.register_param('test_variable')

        self.rmdm_tau = 0.995
        parser.add_argument('--rmdm_tau', type=float, default=self.rmdm_tau, metavar='N',
                            help="smoothing ratio of the representation")
        self.register_param('rmdm_tau')

        self.repre_loss_factor = 1.0
        parser.add_argument('--repre_loss_factor', type=float, default=self.repre_loss_factor, metavar='N',
                            help="size of the representation loss")
        self.register_param('repre_loss_factor')

        self.ep_smooth_factor = 0.0
        parser.add_argument('--ep_smooth_factor', type=float, default=self.ep_smooth_factor, metavar='N',
                            help="smooth  factor for ep module, 0.0 for apply concurrently")
        self.register_param('ep_smooth_factor')

        self.rbf_radius = 80.0
        parser.add_argument('--rbf_radius', type=float, default=self.rbf_radius, metavar='N',
                            help="radius of the rbf kerel")
        self.register_param('rbf_radius')

        self.env_default_change_range = 3.0
        parser.add_argument('--env_default_change_range', type=float, default=self.env_default_change_range, metavar='N',
                            help="environment default change range")
        self.register_param('env_default_change_range')

        self.env_ood_change_range = 4.0
        parser.add_argument('--env_ood_change_range', type=float, default=self.env_ood_change_range,
                            metavar='N',
                            help="environment OOD change range")
        self.register_param('env_ood_change_range')

        self.consistency_loss_weight = 50.0
        parser.add_argument('--consistency_loss_weight', type=float, default=self.consistency_loss_weight, metavar='N',
                            help="loss ratio of the consistency loss")
        self.register_param('consistency_loss_weight')

        self.diversity_loss_weight = 0.025
        parser.add_argument('--diversity_loss_weight', type=float, default=self.diversity_loss_weight, metavar='N',
                            help="loss ratio of the DPP loss")
        self.register_param('diversity_loss_weight')

        self.varying_params = ['gravity', 'body_mass']
        parser.add_argument('--varying_params', nargs='+', type=str, default=self.varying_params)
        self.register_param('varying_params')

        self.up_hidden_size = [128, 64]
        parser.add_argument('--up_hidden_size', nargs='+', type=int, default=self.up_hidden_size,
                            help="architecture of the hidden layers of Universe Policy")
        self.register_param('up_hidden_size')

        self.up_activations = ['leaky_relu', 'leaky_relu', 'linear']
        parser.add_argument('--up_activations', nargs='+', type=str,
                            default=self.up_activations,
                            help="activation of each layer of Universe Policy")
        self.register_param('up_activations')

        self.up_layer_type = ['fc', 'fc', 'fc']
        parser.add_argument('--up_layer_type', nargs='+', type=str,
                            default=self.up_layer_type,
                            help="net type of Universe Policy")
        self.register_param('up_layer_type')

        # self.ep_hidden_size = [128, 64, 32] # [128, 64, 32]
        self.ep_hidden_size = [128, 64] # [256, 128]
        parser.add_argument('--ep_hidden_size', nargs='+', type=int, default=self.ep_hidden_size,
                            help="architecture of the hidden layers of Environment Probing Net")
        self.register_param('ep_hidden_size')

        if FC_MODE:
            self.ep_activations = ['leaky_relu', 'leaky_relu', 'leaky_relu', 'tanh']
            parser.add_argument('--ep_activations', nargs='+', type=str,
                                default=self.ep_activations,
                                help="activation of each layer of Environment Probing Net")
            self.register_param('ep_activations')

            self.ep_layer_type = ['fc', 'fc', 'fc', 'fc']
            parser.add_argument('--ep_layer_type', nargs='+', type=str,
                                default=self.ep_layer_type,
                                help="net type of Environment Probing Net")
            self.register_param('ep_layer_type')
        else:
            # original RNN architecture
            self.ep_activations = ['leaky_relu', 'linear', 'tanh']
            parser.add_argument('--ep_activations', nargs='+', type=str,
                                default=self.ep_activations,
                                help="activation of each layer of Environment Probing Net")
            self.register_param('ep_activations')
            self.ep_layer_type = ['fc', 'gru', 'fc']
            parser.add_argument('--ep_layer_type', nargs='+', type=str,
                                default=self.ep_layer_type,
                                help="net type of Environment Probing Net")
            self.register_param('ep_layer_type')
            # fc architecture
            # self.ep_activations = ['leaky_relu', 'tanh', 'tanh']
            # parser.add_argument('--ep_activations', nargs='+', type=str,
            #                     default=self.ep_activations,
            #                     help="activation of each layer of Environment Probing Net")
            # self.register_param('ep_activations')
            # self.ep_layer_type = ['fc', 'fc', 'fc']
            # parser.add_argument('--ep_layer_type', nargs='+', type=str,
            #                     default=self.ep_layer_type,
            #                     help="net type of Environment Probing Net")
            # self.register_param('ep_layer_type')

        self.ep_dim = 2
        parser.add_argument('--ep_dim', type=int, default=self.ep_dim, metavar='N',
                            help="dimension of environment features")
        self.register_param('ep_dim')

        self.value_hidden_size = [128, 64]
        parser.add_argument('--value_hidden_size', nargs='+', type=int, default=self.value_hidden_size,
                            help="architecture of the hidden layers of value")
        self.register_param('value_hidden_size')

        self.value_activations = ['leaky_relu', 'leaky_relu', 'linear']
        parser.add_argument('--value_activations', nargs='+', type=str,
                            default=self.value_activations,
                            help="activation of each layer of value")
        self.register_param('value_activations')

        self.value_layer_type = ['fc', 'fc', 'fc']
        parser.add_argument('--value_layer_type', nargs='+', type=str,
                            default=self.value_layer_type,
                            help="net type of value")
        self.register_param('value_layer_type')

        return parser.parse_args()

    def register_param(self, name):
        self.arg_names.append(name)

    def get_experiment_description(self):
        description = f"本机{self.host_name}, ip为{self.ip}\n"
        description += f"目前实验目的为{self.experiment_target}\n"
        description += f"实验简称: {self.short_name}\n"
        description += f"commit id: {self.commit_id}\n"
        vars = ''
        important_config = self.important_configs()
        for name in self.arg_names:
            if name in important_config:
                vars += f'**{name}**: {getattr(self, name)}\n'
            else:
                vars += f'{name}: {getattr(self, name)}\n'
        for name in self.DEFAULT_CONFIGS:
            vars += f'{name}: {self.DEFAULT_CONFIGS[name]}\n'
        return description + vars

    def __str__(self):
        return self.get_experiment_description()

    def clear_local_file(self):
        cmd = f'rm -f {os.path.join(self.config_path, self.json_name)} {os.path.join(self.config_path, self.txt_name)}'
        system(cmd)

    def save_config(self):
        self.info(f'save json config to {os.path.join(self.config_path, self.json_name)}')
        if not os.path.exists(self.config_path):
            os.makedirs(self.config_path)
        with open(os.path.join(self.config_path, self.json_name), 'w') as f:
            things = self.make_dict()
            ser = json.dumps(things)
            f.write(ser)
        self.info(f'save readable config to {os.path.join(self.config_path, self.txt_name)}')
        with open(os.path.join(self.config_path, self.txt_name), 'w') as f:
            print(self, file=f)

    def load_config(self):
        self.info(f'load json config from {os.path.join(self.config_path, self.json_name)}')
        with open(os.path.join(self.config_path, self.json_name), 'r') as f:
            ser = json.load(f)
        for k, v in ser.items():
            if not k == 'description':
                setattr(self, k, v)
                self.register_param(k)
        self.experiment_target = ser['description']

    @property
    def differences(self):
        if not os.path.exists(os.path.join(self.config_path, self.json_name)):
            return None
        with open(os.path.join(self.config_path, self.json_name), 'r') as f:
            ser = json.load(f)
        differences = []
        for k, v in ser.items():
            if not hasattr(self, k):
                differences.append(k)
            else:
                v2 = getattr(self, k)
                if not v2 == v:
                    differences.append(k)
        return differences

    def check_identity(self, need_decription=False, need_exec_time=False):
        if not os.path.exists(os.path.join(self.config_path, self.json_name)):
            self.info(f'{os.path.join(self.config_path, self.json_name)} not exists')
            return False
        with open(os.path.join(self.config_path, self.json_name), 'r') as f:
            ser = json.load(f)
        flag = True
        for k, v in ser.items():
            if not k == 'description' and not k == 'exec_time':
                if not hasattr(self, k):
                    flag = False
                    return flag
                v2 = getattr(self, k)
                if not v2 == v:
                    flag = False
                    return flag
        if need_decription:
            if not self.experiment_target == ser['description']:
                flag = False
                return flag
        if need_exec_time:
            if not self.exec_time == ser['exec_time']:
                flag = False
                return flag
        return flag

    @property
    def short_name(self):
        name = ''
        for item in self.important_configs():
            value = getattr(self, item)
            if value:
                if item == 'env_name':
                    name += value
                elif item == 'seed':
                    name += f'-{value}'
                elif item == 'rnn_fix_length':
                    name += f"-rnn_len_{value}"
                elif item == 'ep_dim':
                    name += f"-ep_dim_{value}"
                else:
                    name += f'-{item}'
        if self.debug:
            name += '-debug'
        if self.information is not None:
            name += '-{}'.format(self.information)
        if hasattr(self, 'name_suffix') and not self.name_suffix == 'None':
            name += f'_{self.name_suffix}'
        elif not len(SHORT_NAME_SUFFIX) == 0:
            name += f'_{SHORT_NAME_SUFFIX}'
        return name

    def get_commit_id(self):
        base_path = get_base_path()
        cmd = f'cd {base_path} && git log'
        commit_id = None
        try:
            with os.popen(cmd) as f:
                line = f.readline()
                words = line.split(' ')
                commit_id = words[-1][:-1]
        except Exception as e:
            self.info(f'Error occurs while fetching commit id!!! {e}')
        return commit_id


if __name__ == '__main__':
    parameter = Parameter()
    parameter.get_commit_id()
    print(parameter)


