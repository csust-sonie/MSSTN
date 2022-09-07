
from importlib.resources import path
import numpy as np
import h5py
import pandas as pd

link_of_data = {
    'metr-la': './dataset/METR-LA/metrla.npy',
    'pems-bay': './dataset/PEMS-BAY/pems-bay.h5',
    'pems04': './dataset/PEMS04/pems04.npz',
    'pems08': './dataset/PEMS08/pems08.npz',
    'pemsD7': './dataset/PEMSD7/PeMSD7_V_228.csv',
    'npems03': './dataset/NEWPEMS/PEMS03/PEMS03.npz',
    'npems04': './dataset/NEWPEMS/PEMS04/PEMS04.npz',
    'npems07': './dataset/NEWPEMS/PEMS07/PEMS07.npz',
    'npems08': './dataset/NEWPEMS/PEMS08/PEMS08.npz'
}

link_of_adj = {
    'metr-la': './dataset/METR-LA/metrla-adj.npy',
    'pems-bay': './dataset/PEMS-BAY/distance.csv',
    'pems04': './dataset/PEMS04/distance.csv',
    'pems08': './dataset/PEMS08/distance.csv',
    'pemsD7': './dataset/PEMSD7/distance.csv',
    'npems03': './dataset/NEWPEMS/PEMS03/PEMS03.csv',
    'npems04': './dataset/NEWPEMS/PEMS04/PEMS04.csv',
    'npems07': './dataset/NEWPEMS/PEMS07/PEMS07.csv',
    'npems08': './dataset/NEWPEMS/PEMS08/PEMS08.csv'
}


def read_distance_csv(file, num_sensors, epsilon=0.1):
    try:
        distance = pd.read_csv(file)
    except FileNotFoundError:
        print(f'ERROR: input file was not found in  {file}')

    dist_mx = np.zeros((num_sensors, num_sensors), dtype=np.float32)
    dist_mx[:] = np.inf
    for row in distance.values:
        if row[0] >= num_sensors or row[1] >= num_sensors:
            continue
        dist_mx[int(row[0]), int(row[1])] = row[2]
    distances = dist_mx[~np.isinf(dist_mx)].flatten()
    std = distances.std()
    adj_max = np.exp(-np.square(dist_mx/std))

    adj_max[adj_max < epsilon] = 0
    return adj_max


def load_metr_la():
    data = np.load(link_of_data['metr-la'])
    adj = np.load(link_of_adj['metr-la'])
    return adj, np.expand_dims(data[:, :, 0], axis=-1)


def load_pems_bay():
    data = h5py.File(link_of_data['pems-bay'], 'r')
    data = data['speed']['block0_values'][:]
    data = np.expand_dims(data, axis=-1)

    sensor_ids_filename = './dataset/PEMS-BAY/sensor_graph/graph_sensor_ids.txt'
    with open(sensor_ids_filename) as f:
        sensor_ids = f.read().strip().split(',')
    distance_df = pd.read_csv(
        link_of_adj['pems-bay'], dtype={'from': str, 'to': str})

    def get_adjancy_matrix(distance_df, sensor_ids, normalized_k=0.1):
        num_senors = len(sensor_ids)
        dist_mx = np.zeros((num_senors, num_senors), dtype=np.float32)
        dist_mx[:] = np.inf
        sensor_id_to_ind = {}
        for i, sensor_id in enumerate(sensor_ids):
            sensor_id_to_ind[sensor_id] = i

        for row in distance_df.values:
            if row[0] not in sensor_id_to_ind or row[1] not in sensor_id_to_ind:
                continue
            dist_mx[sensor_id_to_ind[row[0]],
                    sensor_id_to_ind[row[1]]] = row[2]

        distances = dist_mx[~np.isinf(dist_mx)].flatten()
        std = distances.std()
        adj_mx = np.exp(-np.square(dist_mx/std))

        adj_mx[adj_mx < normalized_k] = 0
        return sensor_ids, sensor_id_to_ind, adj_mx

    _, _, adj_mx = get_adjancy_matrix(distance_df, sensor_ids)
    return adj_mx, data


def Pems03_Mapper_Table(file_path):
    fileMapper = open(file_path)
    mapper_dict = {}
    for index, line in enumerate(fileMapper.readlines()):
        if line[-1] == '\n':
            line = line[:-1]
        mapper_dict[int(line)] = index
    return mapper_dict


def read_n_pems03_csv(file_path, num_sensors, epsilon=0.1):
    mapper_path = "." + file_path.split('.')[1] + '.txt'

    mapper_table = Pems03_Mapper_Table(mapper_path)
    try:
        distance = pd.read_csv(file_path)
    except FileNotFoundError:
        print(f'ERROR: input file was not fount in {path}')

    dist_mx = np.zeros((num_sensors, num_sensors), dtype=np.float32)
    dist_mx[:] = np.inf
    for row in distance.values:
        dist_mx[mapper_table[int(row[0])], mapper_table[int(row[1])]] = row[2]
    distances = dist_mx[~np.isinf(dist_mx)].flatten()
    std = distances.std()
    adj_max = np.exp(-np.square(dist_mx/std))

    adj_max[adj_max < epsilon] = 0
    return adj_max


def load_pems_04():
    data = np.load(link_of_data['pems04'])['data']
    adj = read_distance_csv(link_of_adj['pems04'], num_sensors=data.shape[1])
    return adj, np.expand_dims(data[288:, :, 0], axis=-1)


def load_pems_08():
    data = np.load(link_of_data['pems08'])['data']
    adj = read_distance_csv(link_of_adj['pems08'], num_sensors=data.shape[1])
    return adj, np.expand_dims(data[:, :, 0], axis=-1)


def load_n_pems_03():
    data = np.load(link_of_data['npems03'])['data']
    adj = read_n_pems03_csv(link_of_adj['npems03'], num_sensors=data.shape[1])
    return adj, np.expand_dims(data[:, :, 0], axis=-1)


def load_n_pems_04():
    data = np.load(link_of_data['npems04'])['data']
    adj = read_distance_csv(link_of_adj['npems04'], num_sensors=data.shape[1])
    return adj, np.expand_dims(data[:, :, 0], axis=-1)


def load_n_pems_07():
    data = np.load(link_of_data['npems07'])['data']
    adj = read_distance_csv(link_of_adj['npems07'], num_sensors=data.shape[1])
    return adj, np.expand_dims(data[:, :, 0], axis=-1)


def load_n_pems_08():
    data = np.load(link_of_data['npems08'])['data']
    adj = read_distance_csv(link_of_adj['npems08'], num_sensors=data.shape[1])
    return adj, np.expand_dims(data[:, :, 0], axis=-1)


def load_pems_d7():  # 不能算是稀疏矩阵了
    data = pd.read_csv(link_of_data['pemsD7'],
                       dtype=np.float32, header=None).to_numpy()
    data = np.expand_dims(data, axis=-1)

    dist_mx = pd.read_csv(link_of_adj['pemsD7'], header=None).to_numpy()
    distances = dist_mx[~np.isinf(dist_mx)].flatten()
    std = distances.std()
    adj = np.exp(-np.square(dist_mx/std))

    epsilon = 0.1
    adj[adj < epsilon] = 0

    return adj, data


function_for_dataset = {
    'METR-LA': load_metr_la,
    'PEMS-BAY': load_pems_bay,
    'PEMS04': load_pems_04,
    'PEMS08': load_pems_08,
    'PEMS07': load_pems_d7,
    'NPEMS03': load_n_pems_03,
    'NPEMS04': load_n_pems_04,
    'NPEMS07': load_n_pems_07,
    'NPEMS08': load_n_pems_08
}


def normlize_data(data):
    data = data.transpose((1, 2, 0))

    mean = np.mean(data, axis=(0, 2))
    data -= mean.reshape((1, -1, 1))
    std = np.std(data, axis=(0, 2))
    data /= std.reshape((1, -1, 1))
    mean = mean[0]
    std = std[0]
    data = data.transpose((2, 0, 1))

    return data, mean, std


def get_normalized_adj(A):
    """
    Returns a tensor, the degree normalized adjacency matrix.
    """
    alpha = 0.8
    D = np.array(np.sum(A, axis=1)).reshape((-1,))
    D[D <= 10e-5] = 10e-5
    diag = np.reciprocal(np.sqrt(D))
    A_wave = np.multiply(np.multiply(
        diag.reshape((-1, 1)), A), diag.reshape((1, -1)))
    A_reg = alpha / 2 * (np.eye(A.shape[0]) + A_wave)
    return A_reg.astype(np.float32)


def data_factory(name: str):
    adj, data = function_for_dataset[name.upper()]()
    normlized_data, mean, std = normlize_data(data)
    normlized_adj = get_normalized_adj(adj)
    return normlized_data, mean, std, normlized_adj


if __name__ == '__main__':
    for name, _ in function_for_dataset.items():
        data, mean, std, adj = data_factory(name)
        print(
            f"name: {name}.\n"
            f"number of nodes: {adj.shape[0]}\n",
            f"number of timestamps: {data.shape[0]}\n",
            f"number of features: {data.shape[-1]}\n",
            f"mean: {mean}",
            f"std: {std}")
