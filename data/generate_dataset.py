import math
import numpy as np
import itertools
from data.simulator import ControlSimulator
from envs.scenarios import FrictionSliding, AirFall, FrictionlessSHO


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


def generate_dataset_discrete(scenario, values):
    permutations = np.array(list(itertools.product(*values)))

    inputs, outputs = scenario.rollout_func(permutations)
    data = merge_inputs_targets_onehot(inputs, outputs)
    return data


def generate_dataset_discrete_spaced(scenario, lows, highs, num_variations):
    assert len(lows) == len(
        highs) and scenario.num_inputs == len(lows)

    values = []
    for i in range(scenario.num_inputs):
        values.append(np.linspace(lows[i], highs[i], num_variations).tolist())

    data = generate_dataset_discrete(scenario, values)
    return data


def generate_dataset_continuous(simulator, lows, highs, num_variations):
    assert len(lows) == len(
        highs) and simulator.scenario.num_inputs == len(lows)
    num_inputs = simulator.scenario.num_inputs

    dataset_size = num_variations**num_inputs
    data = np.zeros((dataset_size, num_inputs+simulator.scenario.num_outputs,
                     simulator.scenario.trajectory_len, num_inputs+simulator.scenario.num_outputs+1))
    # Number of data for each controlled input variable
    temp = int(dataset_size/num_inputs)
    for k in range(num_inputs):
        for j in range(int(temp/num_variations)):
            inputs, targets = simulator.simulate(k, num_variations)
            out = merge_inputs_targets_onehot(inputs, targets)
            data[k*temp+j*num_variations: k*temp +
                 (j+1)*num_variations, :, :, :] = out
    return data


def main():
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
    trajectory_len = 19
    num_outputs = 2
    num_inputs = 6
    train_variations = 6
    valid_variations = 4
    SHO_interval = 1
    airfall_interval = 0.1
    slide_interval = 1
    delta = False

    # SHO_lows = [1, 2, 0, 32, 11, 21]
    # SHO_highs = [10, 10, 9, 50, 20, 30]
    # SHO_scenario = FrictionlessSHO(
    #     num_inputs, num_outputs, SHO_interval, trajectory_len, delta)
    # SHO_simulator = ControlSimulator(
    #     SHO_scenario, SHO_lows, SHO_highs)

    # SHO_train_discrete_spaced = generate_dataset_discrete_spaced(
    #     SHO_scenario, SHO_lows, SHO_highs, train_variations)
    # SHO_train_cont = generate_dataset_continuous(
    #     SHO_simulator, SHO_lows, SHO_highs, train_variations)
    # SHO_valid_interpolation = generate_dataset_continuous(
    #     SHO_simulator, SHO_lows, SHO_highs, valid_variations)

    # SHO_edge = np.zeros((1, 1, (num_outputs+num_inputs)**2, 2))
    # for i in [51, 52, 55, 58, 61, 62]:
    #     SHO_edge[:, :, i, 1] = 1
    # SHO_edge[:, :, :, 0] = 1.0 - SHO_edge[:, :, :, 1]

    # save_ds(SHO_train_discrete_spaced, SHO_edge,
    #         'train_causal_vel_SHO_spaced_wall_onestep_new')
    # save_ds(SHO_train_cont, SHO_edge,
    #         'train_causal_vel_SHO_cont_wall_onestep_new')
    # save_ds(SHO_valid_interpolation, SHO_edge,
    #         'valid_causal_vel_SHO_wall_onestep_new_interpolation')
    # print('SHO finished.')

    # airfall_lows = [0.1, 1, 8.94427190999916, 2, 100, 50]
    # airfall_highs = [4, 9, 16, 12, 120, 100]
    # airfall_scenario = AirFall(
    #     num_inputs, num_outputs, airfall_interval, trajectory_len, delta)
    # airfall_simulator = ControlSimulator(
    #     airfall_scenario, airfall_lows, airfall_highs)

    # airfall_train_discrete_spaced = generate_dataset_discrete_spaced(
    #     airfall_scenario, airfall_lows, airfall_highs, train_variations)
    # airfall_train_cont = generate_dataset_continuous(
    #     airfall_simulator, airfall_lows, airfall_highs, train_variations)
    # airfall_valid_interpolation = generate_dataset_continuous(
    #     airfall_simulator, airfall_lows, airfall_highs, valid_variations)

    # airfall_edge = np.zeros((1, 1, (num_outputs+num_inputs)**2, 2))
    # for i in [48, 52, 54, 62]:
    #     airfall_edge[:, :, i, 1] = 1
    # airfall_edge[:, :, :, 0] = 1.0 - airfall_edge[:, :, :, 1]

    # save_ds(airfall_train_discrete_spaced, airfall_edge,
    #         'train_causal_vel_airfall_spaced_onestep_c_new')
    # save_ds(airfall_train_cont, airfall_edge,
    #         'train_causal_vel_airfall_cont_onestep_c_new')
    # save_ds(airfall_valid_interpolation, airfall_edge,
    #         'valid_causal_vel_airfall_onestep_new_c_interpolation')
    # print('Airfall finished.')

    slide_lows = [1, 1, 0.01, 0.5235987755982988, 2, 3]
    slide_highs = [10, 11, 0.56, 1.2566370614359172, 10, 20]
    slide_scenario = FrictionSliding(
        num_inputs, num_outputs, slide_interval, trajectory_len, delta)
    slide_simulator = ControlSimulator(
        slide_scenario, slide_lows, slide_highs)

    slide_train_discrete_spaced = generate_dataset_discrete_spaced(
        slide_scenario, slide_lows, slide_highs, train_variations)
    slide_train_cont = generate_dataset_continuous(
        slide_simulator, slide_lows, slide_highs, train_variations)
    slide_valid_cont_interpolation = generate_dataset_continuous(
        slide_simulator, slide_lows, slide_highs, valid_variations)

    values = []
    for i in range(slide_scenario.num_inputs):
        # print(np.linspace(
        #     slide_lows[i], slide_highs[i], train_variations))
        temp = np.linspace(
            slide_lows[i], slide_highs[i], train_variations*2)[1:-1]
        np.random.shuffle(temp)
        values.append(temp[:valid_variations].tolist())
    # print(values)
    slide_valid_discrete_spaced_interpolation = generate_dataset_discrete(
        slide_scenario, values)

    slide_edge = np.zeros((1, 1, (num_outputs+num_inputs)**2, 2))
    for i in [50, 51, 58, 59, 62]:
        slide_edge[:, :, i, 1] = 1
    slide_edge[:, :, :, 0] = 1.0 - slide_edge[:, :, :, 1]

    save_ds(slide_train_discrete_spaced, slide_edge,
            'train_causal_vel_sliding_spaced_new')
    save_ds(slide_train_cont, slide_edge,
            'train_causal_vel_sliding_cont_new')
    save_ds(slide_valid_discrete_spaced_interpolation, slide_edge,
            'valid_causal_vel_sliding_new_spaced_interpolation')
    save_ds(slide_valid_cont_interpolation, slide_edge,
            'valid_causal_vel_sliding_new_cont_interpolation')
    print('Sliding cube finished.')


if __name__ == "__main__":
    main()
