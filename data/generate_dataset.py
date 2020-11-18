import numpy as np
import random
import itertools
import copy
from data.simulator import ControlSimulator
from data.scenarios import FrictionSliding, AirFall, FrictionlessSHO


def save_ds(data, edge, name):
    np.save('data/datasets/feat_'+name+'.npy', data)
    np.save('data/datasets/edges_'+name+'.npy', edge)


def merge_inputs_targets_onehot(inputs, targets):
    data = np.concatenate((inputs, targets), axis=1)
    outputs = np.zeros(
        (data.shape[0], data.shape[1], data.shape[2], 1+data.shape[1]))
    outputs[:, :, :, 0:1] = data
    # Add one hot encoding
    for i in range(1, outputs.shape[-1]):
        outputs[:, i-1, :, i] = 1
    return outputs


def get_noise_std(num_inputs, data, noise_ratio):
    std = np.std(data[:, num_inputs:, :, 0], 0)
    noise = std*noise_ratio
    print('Noise to be added for all datasets',
          noise_ratio, noise, noise.shape)
    return noise


def split_data(scenario, lows, highs, train_num_variations, interpolation_num_variations):
    """
    Generate 2 sets of data values from the same ranges. Valid data's values are guaranteed to be the interpolation of train data's values.
    """
    assert len(lows) == len(
        highs) and scenario.num_inputs == len(lows)
    train_values, interpolation_values = [], []
    for i in range(scenario.num_inputs):
        candidates = np.linspace(
            lows[i], highs[i], train_num_variations[i]+interpolation_num_variations[i])

        extremes = [candidates[0], candidates[-1]]
        middle_values = candidates[1:-1]
        np.random.shuffle(middle_values)
        train_values.append(
            extremes+middle_values[:(train_num_variations[i]-2)].tolist())
        interpolation_values.append(
            middle_values[(train_num_variations[i]-2):].tolist())
    return train_values, interpolation_values


def generate_extrapolate_data(lows, highs, min_ratio, max_ratio, low_num_variations, high_num_variations):
    assert len(lows) == len(highs)
    values = []
    for i in range(len(lows)):
        value_range = highs[i]-lows[i]
        min_range = min_ratio*value_range
        max_range = max_ratio*value_range
        # +1 to exclude the given low/high boundary
        low_extra = np.linspace(
            lows[i]-max_range, lows[i]-min_range, low_num_variations+1)
        high_extra = np.linspace(
            highs[i]+min_range, highs[i]+max_range, high_num_variations+1)
        values.append(np.concatenate([low_extra[:-1], high_extra[1:]]))
    # import pdb
    # pdb.set_trace()
    return values


def generate_dataset_discrete(values, scenario, permute):
    num_inputs = scenario.num_inputs
    num_outputs = scenario.num_outputs
    if permute:
        permutations = np.array(list(itertools.product(*values)))
    else:
        permutations = np.array(values)
    import pdb
    pdb.set_trace()
    inputs, outputs = scenario.rollout_func(permutations)
    data = merge_inputs_targets_onehot(inputs, outputs)

    for i in range(scenario.num_inputs):
        print(set(data[:, i, 0, 0]))
    print('\n')

    noise_data = None
    if scenario.add_noise is not None:
        noise_data = copy.deepcopy(data)
        for node in range(num_inputs, num_inputs+num_outputs):
            print(scenario.add_noise[node-num_inputs])
            noise_data[:, node, :, 0] += np.random.normal(
                0, scenario.add_noise[node-num_inputs], data[:, node, :, 0].shape)
    return data, noise_data


def generate_dataset_continuous(simulator, lows, highs, num_variations, dataset_size):
    assert len(lows) == len(
        highs) and simulator.scenario.num_inputs == len(lows)
    simulator.lows = lows
    simulator.highs = highs
    num_inputs = simulator.scenario.num_inputs
    num_outputs = simulator.scenario.num_outputs

    data = np.zeros((dataset_size, num_inputs+num_outputs,
                     simulator.scenario.trajectory_len, num_inputs+simulator.scenario.num_outputs+1))
    # Number of data for each controlled input variable
    temp = int(dataset_size/num_inputs)
    for k in range(num_inputs):
        for j in range(int(temp/num_variations)):
            inputs, targets = simulator.simulate(k, num_variations)
            out = merge_inputs_targets_onehot(inputs, targets)
            data[k*temp+j*num_variations: k*temp +
                 (j+1)*num_variations, :, :, :] = out

    noise_data = None
    if simulator.scenario.add_noise is not None:
        noise_data = copy.deepcopy(data)
        for node in range(num_inputs, num_inputs+num_outputs):
            print('cont', simulator.scenario.add_noise[node-num_inputs])
            data[:, node, :, 0] += np.random.normal(
                0, simulator.scenario.add_noise[node-num_inputs], data[:, node, :, 0].shape)

    return data, noise_data


def extrapolation_dataset_discrete(scenario, edge, lows, highs, suffix, noise_suffix, extra_low_ranges, extra_interval, extra_num_low, extra_num_high):
    for l in extra_low_ranges:
        h = l+extra_interval
        extrapolate_values = generate_extrapolate_data(
            lows, highs, l, h, extra_num_low, extra_num_high)
        data, noise_data = generate_dataset_discrete(
            extrapolate_values, scenario)
        save_ds(data, edge, '_'.join(
            ['valid_causal_vel', suffix, 'extrapolation', str(l), str(h)]))
        if scenario.add_noise is not None:
            save_ds(noise_data, edge, '_'.join(
                ['valid_causal_vel', noise_suffix, 'extrapolation', str(l), str(h)]))
        print('Saved extrapolation of', suffix, l, h)


def extrapolation_dataset_continuous(simulator, edge, lows, highs, suffix, extra_low_ranges, extra_interval, num_variations):
    for l in extra_low_ranges:
        h = l+extra_interval
        interval = highs-lows
        data = generate_dataset_continuous(
            simulator, highs+interval*l+1e-5, highs+interval*h, num_variations)
        save_ds(data, edge, '_'.join(
            ['valid_causal_vel', suffix, 'extrapolation', str(l), str(h)]))
        print('Saved extrapolation of', suffix, l, h)


def main(scenario_name):
    """
    SHO_spaced_wall_onestep_interpolation.npy
    SHO_spaced_wall_onestep.npy
    SHO_spaced_wall_onestep_cont.npy

    feat_train_causal_vel_SHO_spaced_wall.npy
    {1.0, 2.0, 3.0, 6.0, 7.0, 10.0}
    {2.0, 4.0, 5.0, 7.0, 8.0, 10.0}
    {0.0, 1.0, 2.0, 5.0, 6.0, 9.0}
    {32.0, 36.0, 38.0, 40.0, 42.0, 50.0}
    {11.0, 12.0, 15.0, 18.0, 19.0, 20.0}
    {21.0, 23.0, 25.0, 26.0, 27.0, 30.0}

    feat_valid_causal_vel_SHO_spaced_wall_interpolation.npy
    {8.0, 9.0, 4.0, 5.0}
    {1.0, 3.0, 6.0, 9.0}
    {8.0, 3.0, 4.0, 7.0}
    {48.0, 34.0, 44.0, 46.0}
    {16.0, 17.0, 13.0, 14.0}
    {24.0, 28.0, 29.0, 22.0}


    airfall_numerical_onestep_c_spaced_interpolation.npy
    airfall_numerical_onestep_c_spaced.npy
    airfall_numerical_onestep_cont_c.npy

    feat_train_causal_vel_airfall_numerical_onestep_c.npy
    {0.1, 1.0, 2.3, 3.3, 4.0, 0.7}
    {1.0, 2.0, 3.0, 6.0, 7.0, 9.0}
    {8.94427190999916, 9.0, 10.88888, 9.6, 12.566370614359172, 15.0}
    {2.0, 4.0, 5.0, 7.0, 10.0, 12.0}
    {100.0, 101.0, 109.0, 111.0, 118.3, 120.0}
    {100.0, 74.0, 80.0, 50.0, 90.0, 60.0}

    feat_valid_causal_vel_airfall_numerical_onestep_c_interpolation.npy
    {0.5, 1.3, 2.9, 3.6}
    {8.0, 4.0, 5.0, 7.5}
    {11.0, 12.0, 13.0, 14.0}
    {8.0, 9.0, 3.0, 6.0}
    {105.0, 114.88888, 117.0, 103.0}
    {56.0, 67.0, 78.0, 95.0}


    x0s_interpolation.npy
    grouped_46656_x0s_spaced.npy
    46656_x0s_cont.npy

    feat_train_causal_vel_grouped_46656_x0s_manual.npy
    {1.0, 2.0, 3.0, 6.0, 9.0, 10.0}
    {1.0, 3.0, 5.0, 7.0, 9.0, 11.0}
    {0.3, 0.56, 0.22, 0.1, 0.28, 0.01}
    {0.7853981633974483, 1.0471975511965976, 0.9817477042468103, 1.2566370614359172, 0.6283185307179586, 0.5235987755982988}
    {2.0, 3.5, 5.0, 8.0, 9.0, 10.0}
    {3.0, 5.0, 10.0, 11.0, 17.0, 20.0}

    valid_causal_vel_x0s_interpolation
    {8.0, 4.0, 5.0, 7.0}
    {8.0, 2.0, 4.0, 6.0}
    {0.18, 0.26, 0.48, 0.5}
    {0.8490790955648089, 0.6981317007977318, 0.5711986642890533, 1.1635528346628863}
    {2.7, 3.6, 2.3, 7.0}
    {8.0, 12.0, 5.9, 14.0}
    """
    trajectory_len = 40
    num_outputs = 2
    num_inputs = 6
    train_variations = [6]*num_inputs
    valid_variations = [4]*num_inputs
    delta = False
    permute = True
    train_multiples = 1  # Should not be other values.
    valid_multiples = 4
    # This var is for continuous datasets only. Must be divisible by num_variations*num_inputs
    train_dataset_size = int(np.prod(train_variations))
    val_dataset_size = 4800
    extra_num_low = 0
    extra_num_high = 4
    extra_low_ranges = [0, 0.2, 0.4, 0.6]
    extra_interval = 0.2
    seed = 1
    np.random.seed(seed)
    random.seed(seed)

    # The added gaussian noise for a node has mean=0, std=slide_noise*np.mean(np.std(outputs[node]))
    scenario_noise_ratios = {'sliding': 0.1, 'SHO': 0.1, 'airfall': 0.1}
    scenario_intervals = {'sliding': 0.1, 'SHO': 0.1, 'airfall': 0.1}
    scenario_lows = {'sliding': np.array([0, 0, 0, 0.53, 0, 0]), 'SHO': np.zeros(
        6)+0.2, 'airfall': np.array([0, 0, 0, 0, 2, 0])}
    scenario_highs = {'sliding': np.array([1, 1, 0.56, 1.5, 1, 1]), 'SHO': np.ones(
        6), 'airfall': np.array([0.4, 1, 1, 1, 3, 1])}
    scenario_edges = {'sliding': [50, 51, 58, 59, 62], 'SHO': [
        50, 51, 52, 54, 55, 58, 59, 60, 62, 63], 'airfall': [48, 52, 54, 62]}
    scenarios = {'sliding': FrictionSliding,
                 'SHO': FrictionlessSHO, 'airfall': AirFall}

    lows = scenario_lows[scenario_name]
    highs = scenario_highs[scenario_name]
    scenario = scenarios[scenario_name](
        num_inputs, num_outputs, scenario_intervals[scenario_name], trajectory_len, delta, None)
    simulator = ControlSimulator(scenario, lows, highs)
    train_discrete_values, valid_discrete_values = split_data(
        scenario, lows, highs, train_variations, valid_variations)

    # Uniform the std for all datasets to be the same as discrete_train's.
    train_discrete_spaced, train_discrete_spaced_noise = generate_dataset_discrete(
        train_discrete_values, scenario, permute)
    scenario.add_noise = get_noise_std(
        num_inputs, train_discrete_spaced, scenario_noise_ratios[scenario_name])
    import pdb
    pdb.set_trace()
    # valid_discrete_spaced_interpolation, valid_discrete_spaced_interpolation_noise = generate_dataset_discrete(
    #     valid_discrete_values, scenario)

    # train_cont, train_cont_noise = generate_dataset_continuous(
    #     simulator, lows, highs, train_variations, train_dataset_size)
    # valid_interpolation, valid_interpolation_noise = generate_dataset_continuous(
    #     simulator, lows, highs, valid_variations, val_dataset_size)

    edge = np.zeros((1, 1, (num_outputs+num_inputs)**2, 2))
    # for analytical  for wall[51, 52, 53, 55, 62] for nowall [51, 52, 55, 62]
    for i in scenario_edges[scenario_name]:
        edge[:, :, i, 1] = 1
    edge[:, :, :, 0] = 1.0 - edge[:, :, :, 1]

    if scenario.add_noise is not None:
        n = 'fixedgaussian'+str(scenario_noise_ratios[scenario_name])
    #     save_ds(train_discrete_spaced_noise, edge,
    #             'train_causal_vel_'+scenario_name+'_spaced_'+n+'_new')
    #     save_ds(train_cont_noise, edge,
    #             'train_causal_vel_'+scenario_name+'_cont_'+n+'_new')
    #     save_ds(valid_discrete_spaced_interpolation_noise, edge,
    #             'valid_causal_vel_'+scenario_name+'_spaced_'+n+'_new_interpolation')
    #     save_ds(valid_interpolation_noise, edge,
    #             'valid_causal_vel_'+scenario_name+'_cont_'+n+'_new_interpolation')

    # save_ds(train_discrete_spaced, edge,
    #         'train_causal_vel_'+scenario_name+'_spaced_new')
    # save_ds(train_cont, edge,
    #         'train_causal_vel_'+scenario_name+'_cont_new')
    # save_ds(valid_discrete_spaced_interpolation, edge,
    #         'valid_causal_vel_'+scenario_name+'_spaced_new_interpolation')
    # save_ds(valid_interpolation, edge,
    #         'valid_causal_vel_'+scenario_name+'_cont_new_interpolation')

    # extrapolation_dataset_discrete(scenario, edge, lows, highs, scenario_name+'_spaced_new', scenario_name+'_spaced_'+n+'_new',
    #                                extra_low_ranges, extra_interval, extra_num_low, extra_num_high)
    # extrapolation_dataset_continuous(SHO_simulator, SHO_edge, SHO_lows,
    #                                  SHO_highs, 'SHO_new', extra_low_ranges, extra_interval, valid_variations)
    print(scenario_name, 'finished.')


if __name__ == "__main__":
    main('sliding')
