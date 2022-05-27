import torch
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.cm as cm

def merge_data(data, valid):
    res = []
    for ind, item in enumerate(data):
        sp = item.shape
        # print(ind, valid[ind].shape, sp, item[0][valid[ind][0].squeeze()==1], item[0][valid[ind][0].squeeze(-1)==1])
        if sp[1] == 1:
            res.append(item.squeeze(1).detach().cpu().numpy())
        else:
            data_list = []
            for i in range(sp[0]):
                if len(valid) > 0:
                    valid_it = valid[ind][i]
                    data_it = item[i][valid_it.squeeze(-1)==1]
                else:
                    data_it = item[i]
                data_list.append(data_it)
            res.append(torch.cat(data_list, dim=0).detach().cpu().numpy())
    # assert False
    return res

def to_scalar(param_vector):
    if isinstance(param_vector, float) or isinstance(param_vector, int):
        return param_vector
    else:
        return param_vector[-1]
    pass


def to_2dim_vector(param_vector):

    return [param_vector[0], param_vector[-1]]


def visualize_repre(data, valid, output_file, real_param_dict=None, tasks=None):
    data = merge_data(data, valid)
    fig = plt.figure(0)
    plt.cla()
    cmap = plt.get_cmap('Spectral')
    min_ = 10000
    max_ = -10000
    if real_param_dict is not None:
        for k, v in real_param_dict.items():
            v = to_scalar(v)
            if min_ > v:
                min_ = v
            if max_ < v:
                max_ = v
    norm = plt.Normalize(vmin=min_, vmax=max_)
    for ind, item in enumerate(data):
        x = item[:, 0]
        y = item[:, 1]
        if real_param_dict is None:
            plt.scatter(x, y)
        else:
            task_num = tasks[ind]
            plt.scatter(x, y, c=to_scalar(real_param_dict[task_num]) * np.ones_like(x), cmap=cmap, norm=norm)
    plt.colorbar()
    plt.savefig(output_file)
    plt.xlim(left=-1.1, right=1.1)
    plt.ylim(bottom=-1.1, top=1.1)
    #######
    fig2 = plt.figure(1)
    ax = plt.gca()
    circle_list = []
    for item in data:
        x = item[:, 0]
        y = item[:, 1]
        x_mean = np.mean(x)
        y_mean = np.mean(y)
        std = (np.std(x) + np.std(y)) / 2
        plt.scatter(x_mean, y_mean, marker='+', linewidths=2)
        circle_list.append(plt.Circle((x_mean, y_mean), std / 2, color='r', fill=False))
    for circle in circle_list:
        ax.add_artist(circle)
    plt.xlim(left=-1.1, right=1.1)
    plt.ylim(bottom=-1.1, top=1.1)
    # print(f'saving fig to {output_file}')
    return fig, fig2

def visualize_repre_real_param(data, valid, tasks, real_param_dict):
    data = merge_data(data, valid)
    cmap = plt.get_cmap('Spectral')
    min_ = 10000
    max_ = -10000
    min1_ = 10000
    max1_ = -10000
    min2_ = 10000
    max2_ = -10000
    for k, v in real_param_dict.items():
        v1, v2 = to_2dim_vector(v)

        v = to_scalar(v)

        if min_ > v:
            min_ = v
        if max_ < v:
            max_ = v
        min1_ = min1_ if min1_ < v1 else v1
        max1_ = max1_ if max1_ > v1 else v1

        min2_ = min2_ if min2_ < v2 else v2
        max2_ = max2_ if max2_ > v2 else v2
    norm = plt.Normalize(vmin=min_, vmax=max_)
    norm1 = plt.Normalize(vmin=min1_, vmax=max1_)
    norm2 = plt.Normalize(vmin=min2_, vmax=max2_)

    fig2 = plt.figure(3)
    figsize = (3.2 * 2, 2.24 * 2 * 1.5)
    f, axarr = plt.subplots(2, 1, sharex=True, squeeze=False, figsize=figsize)

    pts = None
    means = []
    colors = []
    for i in range(2):
        ax = axarr[i][0]
        for ind, item in enumerate(data):
            task_num = tasks[ind]
            real_param = real_param_dict[task_num]
            vector_real_param = to_2dim_vector(real_param)
            v1, v2 = to_2dim_vector(real_param)
            vs = [v1, v2]
            v1_norm = norm1(v1)
            v2_norm = norm2(v2)
            norms = [norm1, norm2]
            normalized_color = cmap(v1_norm)
            lightness = (vector_real_param[0] + 1) / 2
            # light_color = [lightness * item for item in normalized_color]
            normalized_color = [normalized_color[0],
                                normalized_color[1],
                                normalized_color[2],
                                v2_norm]
            light_color = normalized_color
            # light_color = [0, v2_norm, v1_norm]
            x = item[:, 0]
            y = item[:, 1]
            x_mean = np.mean(x)
            y_mean = np.mean(y)
            means.append([x_mean, y_mean])
            colors.append(light_color)
            ax.scatter([x_mean], [y_mean], marker='o', linewidths=3, c=[vs[i]], cmap=cmap, norm=norms[i])
            ax.set_xlim(left=-1.1, right=1.1)
            ax.set_ylim(bottom=-1.1, top=1.1)
            # ax.scatter([x_mean], [y_mean], marker='.', linewidths=1, c=[v1], cmap=cmap, norm=norm1)
    # means = np.array(means)

    # plt.scatter(means[:, 0], means[:, 1], marker='+', linewidths=2, c=colors)
    mapple = cm.ScalarMappable(norm=norm1, cmap=cmap)
    mapple.set_array([])
    plt.colorbar(mapple, ax=[axarr[0][0], axarr[1][0]])
    # plt.xlim(left=-1.1, right=1.1)
    # plt.ylim(bottom=-1.1, top=1.1)
    # print(f'saving fig to {output_file}')
    return f


def visualize_repre_real_param_legacy(data, valid, tasks, real_param_dict):
    data = merge_data(data, valid)

    fig2 = plt.figure(3)
    for ind, item in enumerate(data):
        task_num = tasks[ind]
        real_param = real_param_dict[task_num]
        x = item[:, 0]
        y = item[:, 1]
        x_mean = np.mean(x)
        y_mean = np.mean(y)
        plt.scatter(x_mean, y_mean, marker='+', linewidths=2)
        plt.text(x_mean, y_mean, '{:.2f}'.format(to_scalar(real_param)))
    plt.xlim(left=-1.1, right=1.1)
    plt.ylim(bottom=-1.1, top=1.1)
    # print(f'saving fig to {output_file}')
    return fig2

def xy_filter(x, y, ratio):
    x_mean = np.mean(x)
    y_mean = np.mean(y)
    distance = np.sqrt(np.square(x - x_mean) + np.square(y - y_mean))
    distance_sorted = np.sort(distance)
    distance_threshold = distance_sorted[int(distance_sorted.shape[0] * ratio)]
    x_res = x[distance < distance_threshold]
    y_res = y[distance < distance_threshold]
    return x_res, y_res

def visualize_repre_filtered(data, valid, ratio=0.8):
    data = merge_data(data, valid)
    fig = plt.figure(7)
    plt.cla()
    for item in data:
        x = item[:, 0]
        y = item[:, 1]
        x, y = xy_filter(x, y, ratio)
        plt.scatter(x, y)
    plt.xlim(left=-1.1, right=1.1)
    plt.ylim(bottom=-1.1, top=1.1)

    fig2 = plt.figure(1)
    ax = plt.gca()
    circle_list = []
    for item in data:
        x = item[:, 0]
        y = item[:, 1]
        x, y = xy_filter(x, y, ratio)
        x_mean = np.mean(x)
        y_mean = np.mean(y)
        std = (np.std(x) + np.std(y)) / 2
        plt.scatter(x_mean, y_mean, marker='+', linewidths=2)
        circle_list.append(plt.Circle((x_mean, y_mean), std / 2, color='r', fill=False))
    for circle in circle_list:
        ax.add_artist(circle)
    plt.xlim(left=-1.1, right=1.1)
    plt.ylim(bottom=-1.1, top=1.1)
    # print(f'saving fig to {output_file}')
    return fig, fig2


"""
fig = plt.figure(0)
ax = plt.gca()
disk1 = plt.Circle((0, 0), 0.3, color='r', fill=False)
disk2 = plt.Circle((0, 0.5), 0.3, color='r', fill=False)
ax.set_xlim((-1, 1))
ax.set_ylim((-1, 1))
ax.add_artist(disk1)
ax.add_artist(disk2)
plt.show()
"""