from parameter.Parameter import Parameter
from log_util.logger_base import LoggerBase
from parameter.private_config import *
import os
import numpy as np
import time
import copy


class Logger(LoggerBase):
    def __init__(self, log_to_file=True, parameter=None, force_backup=False):
        if parameter:
            self.parameter = parameter
        else:
            self.parameter = Parameter()
        self.output_dir = os.path.join(get_base_path(), 'log_file', self.parameter.short_name)
        if log_to_file:
            if not os.path.exists(self.output_dir):
                os.makedirs(self.output_dir)
            if os.path.exists(os.path.join(self.output_dir, 'log.txt')):
                system(f'mv {os.path.join(self.output_dir, "log.txt")} {os.path.join(self.output_dir, "log_back.txt")}')
            self.log_file = open(os.path.join(self.output_dir, 'log.txt'), 'w')
        else:
            self.log_file = None
            # super(Logger, self).set_log_file(self.log_file)
        super(Logger, self).__init__(self.output_dir, log_file=self.log_file)
        # self.parameter.set_log_func(lambda x: self.log(x))
        self.current_data = {}
        self.logged_data = set()

        self.model_output_dir = self.get_model_output_path(self.parameter)
        self.log(f"my output path is {self.output_dir}")

        self.parameter.set_config_path(self.output_dir)
        if not os.path.exists(self.output_dir):
            self.log(f'directory {self.output_dir} does not exist, create it...')
        else:
            self.log(f'directory {self.output_dir} exists, checking identity...')
            if (self.parameter.check_identity(need_decription=True) or self.parameter.differences is None) \
                    and (not force_backup):
                self.log(f'config is completely same, file will be overwrited anyway...')
            else:
                self.log(f'config is not same, file will backup first...')
                diffs = self.parameter.differences
                self.log(f'difference appears in {diffs}')
                backup_dir = os.path.join(get_base_path(), "log_file", f"backup_{self.parameter.exec_time}")
                if not os.path.exists(backup_dir):
                    os.makedirs(backup_dir)
                system(f"cp -r {self.output_dir} {backup_dir}", lambda x: self.log(x))
        self.parameter.save_config()
        self.init_tb()
        self.backup_code()
        self.tb_header_dict = {}
        # self.output_dir = os.path.join(get_base_path(), "log_file")

    @staticmethod
    def get_model_output_path(parameter):
        output_dir = os.path.join(get_base_path(), 'log_file', parameter.short_name)
        return os.path.join(output_dir, 'model')

    @staticmethod
    def get_replay_buffer_path(parameter):
        output_dir = os.path.join(get_base_path(), 'log_file', parameter.short_name)
        return os.path.join(output_dir, 'replay_buffer.pkl')

    def backup_code(self):
        base_path = get_base_path()
        things = []
        for item in os.listdir(base_path):
            p = os.path.join(base_path, item)
            if not item.startswith('.') and not item.startswith('__') and not item == 'log_file' and not item == 'baselines':
                things.append(p)
        code_path = os.path.join(self.output_dir, 'codes')
        if not os.path.exists(code_path):
            os.makedirs(code_path)
        for item in things:
            system(f'cp -r {item} {code_path}', lambda x: self.log(f'backing up: {x}'))

    def log(self, *args, color=None, bold=True):
        super(Logger, self).log(*args, color=color, bold=bold)

    def log_dict(self, color=None, bold=False, **kwargs):
        for k, v in kwargs.items():
            super(Logger, self).log('{}: {}'.format(k, v), color=color, bold=bold)

    def log_dict_single(self, data, color=None, bold=False):
        for k, v in data.items():
            super(Logger, self).log('{}: {}'.format(k, v), color=color, bold=bold)

    def __call__(self, *args, **kwargs):
        self.log(*args, **kwargs)

    def save_config(self):
        self.parameter.save_config()

    def log_tabular(self, key, val=None, tb_prefix=None, with_min_and_max=False, average_only=False, no_tb=False):
        if val is not None:
            super(Logger, self).log_tabular(key, val, tb_prefix, no_tb=no_tb)
        else:
            if key in self.current_data:
                self.logged_data.add(key)
                super(Logger, self).log_tabular(key if average_only else "Average"+key, np.mean(self.current_data[key]), tb_prefix, no_tb=no_tb)
                if not average_only:
                    super(Logger, self).log_tabular("Std" + key,
                                                    np.std(self.current_data[key]), tb_prefix, no_tb=no_tb)
                    if with_min_and_max:
                        super(Logger, self).log_tabular("Min" + key, np.min(self.current_data[key]), tb_prefix, no_tb=no_tb)
                        super(Logger, self).log_tabular('Max' + key, np.max(self.current_data[key]), tb_prefix, no_tb=no_tb)

    def add_tabular_data(self, tb_prefix=None, **kwargs):
        for k, v in kwargs.items():
            if tb_prefix is not None and k not in self.tb_header_dict:
                self.tb_header_dict[k] = tb_prefix
            if k not in self.current_data:
                self.current_data[k] = []
            if not isinstance(v, list):
                self.current_data[k].append(v)
            else:
                self.current_data[k] += v

    def update_tb_header_dict(self, tb_header_dict):
        self.tb_header_dict.update(tb_header_dict)

    def dump_tabular(self):
        for k in self.current_data:
            if k not in self.logged_data:
                if k in self.tb_header_dict:
                    self.log_tabular(k, tb_prefix=self.tb_header_dict[k], average_only=True)
                else:
                    self.log_tabular(k, average_only=True)
        self.logged_data.clear()
        self.current_data.clear()
        super(Logger, self).dump_tabular()


if __name__ == '__main__':
    logger = Logger()
    logger.log(122, '22', color='red', bold=False)
    data = {'a': 10, 'b': 11, 'c': 13}
    for i in range(100):
        for _ in range(10):
            for k in data:
                data[k] += 1
            logger.add_tabular_data(**data)
        logger.log_tabular('a')
        logger.dump_tabular()
        time.sleep(1)





