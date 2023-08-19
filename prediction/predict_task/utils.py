import os, argparse


def mkdir(path):
    path = path.split('/')
    for iter in range(1, len(path) + 1):
        temp_dir = '/'.join(path[:iter])
        if not os.path.exists(temp_dir):
            os.mkdir(temp_dir)


def initialization():
    Network = 'BJ-331'
    n_w = 6.1  # 6.1
    rho = 0.6  # parameter_ρ
    time_geo_gamma = 0.21  # parameter_γ
    time_geo_alpha = 0.86  # parameter_α
    beta1 = 3.67  # 3.67
    beta2 = 10.0  # 10
    pop_num = 50000
    mode = 'complete'  # test-50day,orgion-100day,pro-107day,complete-207day
    init = 20
    spread_func = 'ICM'  # ICM,ICM_on_graph
    home = 'no_home'  # home,no_home
    beta = 0.08
    gamma = 0.04
    R0 = 10
    rr = 1 / 7
    ir = R0 * rr
    if mode == 'simplified':
        simu_slot = 24 * 57
    elif mode == 'complete':
        simu_slot = 24 * 107
    shp = 'ST_R_XA_WGS'
    region_num = 331

    trajectories_save_dir = 'data/{0}/' \
                            'n_w={1}-beta1={2}-beta2={3}-rho={4}-gamma={5}-alpha={6}' \
                            '/trajectories' \
        .format(Network, n_w, beta1, beta2, rho, time_geo_gamma, time_geo_alpha)
    graph_save_dir = 'data/{0}/' \
                     'n_w={1}-beta1={2}-beta2={3}-rho={4}-gamma={5}-alpha={6}' \
                     '/graphs' \
        .format(Network, n_w, beta1, beta2, rho, time_geo_gamma, time_geo_alpha)
    model_save_dir = 'data/{0}/' \
                     'n_w={1}-beta1={2}-beta2={3}-rho={4}-gamma={5}-alpha={6}' \
                     '/save' \
        .format(Network, n_w, beta1, beta2, rho, time_geo_gamma, time_geo_alpha)
    output_save_dir = 'data/{0}/' \
                      'n_w={1}-beta1={2}-beta2={3}-rho={4}-gamma={5}-alpha={6}' \
                      '/prediction_results' \
        .format(Network, n_w, beta1, beta2, rho, time_geo_gamma, time_geo_alpha)
    parser = argparse.ArgumentParser()
    parser.add_argument('-mode', dest='mode', default=mode, required=False, type=str, help='mode_of_code')
    parser.add_argument('-beta1', dest='beta1', default=beta1, required=False, type=float, help='beta1_of_TimeGeo')
    parser.add_argument('-beta2', dest='beta2', default=beta2, required=False, type=float, help='beta2_of_TimeGeo')
    parser.add_argument('-rho', dest='rho', default=rho, required=False, type=float, help='rho_of_TimeGeo')
    parser.add_argument('-time_geo_gamma', dest='time_geo_gamma', default=time_geo_gamma, required=False, type=float,
                        help='gamma_of_TimeGeo')
    parser.add_argument('-time_geo_alpha', dest='time_geo_alpha', default=time_geo_alpha, required=False, type=float,
                        help='alpha_of_TimeGeo')
    parser.add_argument('-trajectories_save_dir', dest='trajectories_save_dir', default=trajectories_save_dir,
                        required=False, type=str, help='save_dir_of_trajectories')
    parser.add_argument('-graph_save_dir', dest='graph_save_dir', default=graph_save_dir,
                        required=False, type=str, help='save_dir_of_graph')
    parser.add_argument('-model_save_dir', dest='model_save_dir', default=model_save_dir,
                        required=False, type=str, help='save_dir_of_model')
    parser.add_argument('-output_save_dir', dest='output_save_dir', default=output_save_dir,
                        required=False, type=str, help='save_dir_of_output')
    parser.add_argument('-Network', dest='Network', default=Network, required=False, type=str, help='Network')
    parser.add_argument('-shp', dest='shp', default=shp, required=False, type=str, help='shape_file')
    parser.add_argument('-n_w', dest='n_w', default=n_w, required=False, type=float, help='avg_destination')
    parser.add_argument('-pop_num', dest='pop_num', default=pop_num, required=False, type=int, help='population_num')
    parser.add_argument('-region_num', dest='region_num', default=region_num, required=False, type=int,
                        help='region_num')
    parser.add_argument('-simu_slot', dest='simu_slot', default=simu_slot, required=False, type=int,
                        help='simulate_hours')
    parser.add_argument('-init', dest='init', default=init, required=False, type=int, help='initial_ill_num')
    parser.add_argument('-spread_func', dest='spread_func', default=spread_func, required=False, type=str,
                        help='function_of_spread')
    parser.add_argument('-home', dest='home', default=home, required=False, type=str,
                        help='infection_in_home_count_or_not')
    parser.add_argument('-beta', dest='beta', default=beta, required=False, type=float, help='recovery_rate')
    parser.add_argument('-gamma', dest='gamma', default=gamma, required=False, type=float, help='infection_rate')
    parser.add_argument('-R0', dest='R0', default=R0, required=False, type=float, help='R0_of_ICM_on_graph')
    parser.add_argument('-recovery_rate', dest='rr', default=rr, required=False, type=float,
                        help='recovery_rate_of_ICM_on_graph')
    parser.add_argument('-infect_rate', dest='ir', default=ir, required=False, type=float,
                        help='infect_rate_of_ICM_on_graph')

    # args = parser.parse_args()
    args = parser.parse_known_args()[0]
    return args
