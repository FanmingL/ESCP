import os.path as osp
import os


def get_base_path():
    return osp.dirname(osp.dirname(osp.abspath(__file__)))


def system(cmd, print_func=None):
    if print_func is None:
        print(cmd)
    else:
        print_func(cmd)
    os.system(cmd)

EXPERIMENT_TARGET = "RELEASE"
MAIN_MACHINE_IP = "114.212.22.189"
SKIP_MAX_LEN_DONE = True
FC_MODE = False
ENV_DEFAULT_CHANGE = 3.0
USE_TQDM = False
NON_STATIONARY_PERIOD = 100
NON_STATIONARY_INTERVAL = 10
SHORT_NAME_SUFFIX = 'N'


def get_global_configs(things):
    res = dict()
    for k, v in things:
        if not k.startswith('__') and not hasattr(v, '__call__') and 'module' not in str(type(v)):
            res[k] = v
    return res

def global_configs(things=[*locals().items()]):
    return get_global_configs(things)

# ALL_CONFIGS = get_global_configs([*locals().items()])

