import copy
import timeit

from networkx import nx
import numpy as np
from model import *
import csv


# total vertices
n = 25

# random scenarios
n_scenarios = 20

# seeds
seed = 12345
np.random.seed(seed)

# distance: D km x D km
D = 2000

# budget: from, to, step
B_f = 200
B_t = 5100
B_s = 200

max_drone_speed = 20
max_wind_speed = 15
max_payload = 7

winds = [0, 5, 10, 15]

prefixes = compute_prefixes()
distances = np.zeros((n, n))
directions = np.zeros((n, n))


def get_relative_wind(u, v, w_d):
    drone_direction = directions[u][v]
    if drone_direction < 0:
        drone_direction = drone_direction + 360
    r_wd = abs(drone_direction - w_d)
    
    if r_wd <= 45 or r_wd >= 315:
        r_wd = 0
    elif r_wd <= 90 or r_wd >= 270:
        r_wd = 45
    elif r_wd <= 135 or r_wd >= 225:
        r_wd = 135
    else:
        r_wd = 180

    if w_d == -1:
        r_wd = 0
    if w_d == -2:
        r_wd = 180

    return r_wd


def create_graph(c):
    p = c * np.log(n) / n
    G = nx.Graph()
    G.add_nodes_from([-1, -2])
    while not nx.is_connected(G):
        G = nx.erdos_renyi_graph(n, p, directed=False)  # seed=seed,

    pos = nx.random_layout(G)

    for u, vdict in G.adjacency():
        for v in vdict:
            # distances
            distances[u][v] = np.sqrt((pos[u][0] * D - pos[v][0] * D) ** 2 + (pos[u][1] * D - pos[v][1] * D) ** 2)

            # directions
            directions[u][v] = int(np.rad2deg(float(sp.atan2(pos[v][1] - pos[u][1], pos[v][0] - pos[u][0]))))

    return G, pos


def compute_edge_weight(u, v, payload_weight, drone_speed, global_wind_speed, global_wind_direction):
    distance = distances[u][v]
    relative_wind_direction = get_relative_wind(u, v, global_wind_direction)
    weight = prefixes[(payload_weight, drone_speed, global_wind_speed, relative_wind_direction)] * distance

    return weight


def update_graph(G, params):
    # pos = params['pos']
    drone_speed = params['v']
    payload_weight = params['p']
    global_wind_speed = params['ws']
    global_wind_direction = params['wd']  # -1: force head, -2: force tail

    for u, vdict in G.adjacency():
        for v in vdict:
            G[u][v]['weight'] = compute_edge_weight(u, v, payload_weight, drone_speed, global_wind_speed, global_wind_direction)


def alg_pre_proc(G, params):
    source = 0

    # white: 0, gray: 1, black: 2
    colors = np.ones(n, dtype=int)
    colors[source] = -1

    lens = np.zeros(n)

    # max consumption
    params['ws'] = max_wind_speed
    params['v'] = max_drone_speed

    # worst scenario
    params['wd'] = -2  # force head
    update_graph(G, params)
    for i in range(1, n):
        lens[i] = nx.shortest_path_length(G, source=source, target=i, weight='weight')

    params['p'] = 0
    update_graph(G, params)
    for i in range(1, n):
        lens[i] = lens[i] + nx.shortest_path_length(G, source=i, target=source, weight='weight')

    for i in range(1, n):
        if lens[i] <= params['B']:
            colors[i] = 0

    # best scenario
    params['wd'] = -1  # force tail
    params['p'] = max_payload
    update_graph(G, params)
    for i in range(1, n):
        lens[i] = nx.shortest_path_length(G, source=source, target=i, weight='weight')

    params['p'] = 0
    update_graph(G, params)
    for i in range(1, n):
        lens[i] = lens[i] + nx.shortest_path_length(G, source=i, target=source, weight='weight')

    for i in range(1, n):
        if lens[i] > params['B']:
            colors[i] = 2

    return colors


def run_test_colors():
    # budget, avg1, avg2, avg3
    with open('out/colors/colors_c%.1f_n%d.csv' % (c, n), mode='w', newline='') as file:
        writer = csv.writer(file, delimiter=',')

        # drone budget [kJ]
        for budget in range(B_f, B_t, B_s):
            print(budget)

            # white, gray, black
            exp_colors = np.zeros((n_scenarios, 3))

            for i in range(0, n_scenarios):
                G, pos = create_graph(c)

                params = {
                    'pos': pos,
                    'v': max_drone_speed,
                    'ws': max_wind_speed,
                    'p': max_payload,
                    'B': budget,
                }

                colors = alg_pre_proc(G, params)
                n_white = len(colors[colors == 0])
                n_gray = len(colors[colors == 1])
                n_black = len(colors[colors == 2])

                exp_colors[i][0] = 100.*n_white/(n-1)
                exp_colors[i][1] = 100.*n_gray/(n-1)
                exp_colors[i][2] = 100.*n_black/(n-1)

                print("[W=%.1f, G=%.1f, B=%.1f] perc" % (exp_colors[i][0], exp_colors[i][1], exp_colors[i][2]))

            avgs = np.mean(exp_colors, axis=0)
            stds = np.std(exp_colors, axis=0)
            print(avgs)
            print(stds)

            writer.writerow([
                budget,
                round(avgs[0], 1), round(avgs[1], 1), round(avgs[2], 1),
            ])


def run_test_alg_off_sp(drone_speed, c):
    with open('out/off_sp/alg_off_sp_c%.1f_n%d_v%d.csv' % (c, n, drone_speed), mode='w', newline='') as file:
        writer = csv.writer(file, delimiter=',')

        writer.writerow(["budget, whites, grays, blacks, abort, h_abort, success, delivered, fail"])

        # drone budget [kJ]
        for budget in range(B_f, B_t, B_s):
            print(budget)

            # abort, h_abort, success, delivered, fail
            exp_statuses = np.zeros((n_scenarios, 5))
            n_whites = np.zeros(n_scenarios)
            n_grays = np.zeros(n_scenarios)
            n_blacks = np.zeros(n_scenarios)

            i = 0
            while i < n_scenarios:
                G, pos = create_graph(c)

                params = {
                    'pos': pos,
                    'v': drone_speed,
                    'ws': max_wind_speed,
                    'wd': 0,
                    'p': max_payload,
                    'B': budget,
                }

                colors = alg_pre_proc(G, params)

                global_wind_speed = np.random.choice(winds)
                global_wind_direction = np.random.randint(360)

                n_white = len(colors[colors == 0])
                n_gray = len(colors[colors == 1])
                n_black = len(colors[colors == 2])

                if n_gray == 0:
                    continue

                s_abort = 0
                s_h_abort = 0
                s_success = 0
                s_delivered = 0
                s_fail = 0

                params = {
                    'pos': pos,
                    'v': drone_speed,
                    'ws': global_wind_speed,
                    'wd': global_wind_direction,
                    'p': max_payload,
                }

                G1 = copy.deepcopy(G)
                update_graph(G1, params)

                params['p'] = 0
                G2 = copy.deepcopy(G)
                update_graph(G2, params)

                for k in range(1, n):
                    # only gray vertices
                    # if colors[i] == -1:
                    #     print("%d: source" % i)
                    # if colors[i] == 0:
                    #     print("%d: white" % i)
                    # if colors[i] == 2:
                    #     print("%d: black" % i)
                    if colors[k] == 1:
                        destination = k

                        path_1 = nx.shortest_path(G1, source=0, target=destination, weight='weight')
                        path_len_1 = nx.shortest_path_length(G1, source=0, target=destination, weight='weight')

                        path2 = nx.shortest_path(G2, source=destination, target=0, weight='weight')
                        path_len_2 = nx.shortest_path_length(G2, source=destination, target=0, weight='weight')

                        out = {
                            'path': [path_1, path2],
                            'len': [path_len_1, path_len_2],
                        }

                        path = out['path']
                        path_len = out['len']
                        expected_len = np.sum(path_len)
                        # print("Expected energy=%d" % np.sum(path_len))

                        actual_len_go = actual_len_back = 0
                        for j in range(0, len(path[0])-1):
                            global_wind_speed = np.random.choice(winds)
                            global_wind_direction = np.random.randint(360)
                            edge_w = compute_edge_weight(path[0][j], path[0][j+1], max_payload, drone_speed, global_wind_speed, global_wind_direction)
                            actual_len_go = actual_len_go + edge_w
                        for j in range(0, len(path[1])-1):
                            global_wind_speed = np.random.choice(winds)
                            global_wind_direction = np.random.randint(360)
                            edge_w = compute_edge_weight(path[1][j], path[1][j + 1], 0, drone_speed, global_wind_speed, global_wind_direction)
                            actual_len_back = actual_len_back + edge_w

                        actual_len = actual_len_go + actual_len_back

                        # print("Actual energy=%d" % actual_len)
                        if expected_len > budget:
                            s_abort = s_abort + 1
                            if path_len_1 < budget:
                                s_h_abort = s_h_abort + 1
                            # print("abort")
                        else:
                            if actual_len <= budget:
                                s_success = s_success+1
                                # print("success")
                            else:
                                if actual_len_go <= budget:
                                    s_delivered = s_delivered + 1
                                    # print("delivered")
                                else:
                                    s_fail = s_fail + 1
                                    # print("fail")

                        # print("%d: gray (exp=%d, act=%d)" % (i, expected_len, actual_len))

                print("Gray=%d -> abort=%d, success=%d, delivered=%d, fail=%d" % (n_gray, s_abort, s_success, s_delivered, s_fail))

                exp_statuses[i][0] = 100. * s_abort / n_gray
                exp_statuses[i][1] = 100. * s_h_abort / n_gray
                exp_statuses[i][2] = 100. * s_success / n_gray
                exp_statuses[i][3] = 100. * s_delivered / n_gray
                exp_statuses[i][4] = 100. * s_fail / n_gray

                n_whites[i] = 100.*n_white/(n-1)
                n_grays[i] = 100.*n_gray/(n-1)
                n_blacks[i] = 100.*n_black/(n-1)

                i = i+1

            avgs = np.mean(exp_statuses, axis=0)
            # stds = np.std(exp_statuses, axis=0)
            print(np.mean(n_grays, axis=0))
            print(avgs)
            # print(stds)

            n_whites_avg = np.mean(n_whites, axis=0)
            n_grays_avg = np.mean(n_grays, axis=0)
            n_blacks_avg = np.mean(n_blacks, axis=0)
            # print(n_whites_avg)
            # print(n_grays_avg)
            # print(n_blacks_avg)

            writer.writerow([
                budget,
                round(n_whites_avg, 1), round(n_grays_avg, 1), round(n_blacks_avg, 1),
                round(avgs[0], 1), round(avgs[1], 1), round(avgs[2], 1), round(avgs[3], 1), round(avgs[4], 1),
            ])


def run_test_alg_on_sp(drone_speed, c):
    with open('out/on_sp/alg_on_sp_c%.1f_n%d_v%d.csv' % (c, n, drone_speed), mode='w', newline='') as file:
        writer = csv.writer(file, delimiter=',')

        writer.writerow(["budget, whites, grays, blacks, abort, success, delivered, fail"])

        # drone budget [kJ]
        for budget in range(B_f, B_t, B_s):
            print(budget)

            # abort, success, delivered, fail
            exp_statuses = np.zeros((n_scenarios, 4))
            n_whites = np.zeros(n_scenarios)
            n_grays = np.zeros(n_scenarios)
            n_blacks = np.zeros(n_scenarios)

            i = 0
            while i < n_scenarios:
                G, pos = create_graph(c)

                params = {
                    'pos': pos,
                    'v': drone_speed,
                    'ws': max_wind_speed,
                    'wd': 0,
                    'p': max_payload,
                    'B': budget,
                }

                colors = alg_pre_proc(G, params)

                n_white = len(colors[colors == 0])
                n_gray = len(colors[colors == 1])
                n_black = len(colors[colors == 2])

                if n_gray == 0:
                    continue

                # s_abort = 0
                s_success = 0
                s_delivered = 0
                s_fail = 0

                G_bak = copy.deepcopy(G)

                for k in range(1, n):
                    if colors[k] == 1:
                        dest = k

                        params = {
                            'pos': pos,
                            'v': drone_speed,
                            'ws': np.random.choice(winds),
                            'wd': np.random.randint(360),
                            'p': max_payload,
                        }
                        update_graph(G, params)

                        actual_len_go = actual_len_back = 0

                        src = 0

                        while True:
                            path = nx.shortest_path(G, source=src, target=dest, weight='weight')
                            if len(path) > 1:
                                u = path[0]
                                v = path[1]
                                weight = G[u][v]['weight']
                                actual_len_go = actual_len_go + weight

                                G.remove_node(u)

                                params = {
                                    'pos': pos,
                                    'v': drone_speed,
                                    'ws': np.random.choice(winds),
                                    'wd': np.random.randint(360),
                                    'p': max_payload,
                                }
                                update_graph(G, params)
                                src = v
                            else:
                                break

                        G = copy.deepcopy(G_bak)

                        params = {
                            'pos': pos,
                            'v': drone_speed,
                            'ws': np.random.choice(winds),
                            'wd': np.random.randint(360),
                            'p': 0,
                        }
                        update_graph(G, params)

                        src = dest
                        dest = 0

                        while True:
                            path = nx.shortest_path(G, source=src, target=dest, weight='weight')
                            if len(path) > 1:
                                u = path[0]
                                v = path[1]
                                weight = G[u][v]['weight']
                                actual_len_back = actual_len_back + weight

                                G.remove_node(u)

                                params = {
                                    'pos': pos,
                                    'v': drone_speed,
                                    'ws': np.random.choice(winds),
                                    'wd': np.random.randint(360),
                                    'p': max_payload,
                                }
                                update_graph(G, params)
                                src = v
                            else:
                                break

                        G = copy.deepcopy(G_bak)

                        actual_len = actual_len_go + actual_len_back

                        # print("Actual energy=%d" % actual_len)
                        if actual_len <= budget:
                            s_success = s_success + 1
                            # print("success")
                        else:
                            if actual_len_go <= budget:
                                s_delivered = s_delivered + 1
                                # print("delivered")
                            else:
                                s_fail = s_fail + 1
                                # print("fail")

                        # print("%d: gray (exp=%d, act=%d)" % (i, expected_len, actual_len))
                print("Gray=%d -> success=%d, delivered=%d, fail=%d" % (n_gray, s_success, s_delivered, s_fail))

                exp_statuses[i][1] = 100. * s_success / n_gray
                exp_statuses[i][2] = 100. * s_delivered / n_gray
                exp_statuses[i][3] = 100. * s_fail / n_gray

                n_whites[i] = 100. * n_white / (n - 1)
                n_grays[i] = 100. * n_gray / (n - 1)
                n_blacks[i] = 100. * n_black / (n - 1)

                i = i + 1

            avgs = np.mean(exp_statuses, axis=0)
            stds = np.std(exp_statuses, axis=0)
            print(np.mean(n_grays, axis=0))
            print(avgs)
            print(stds)

            n_whites_avg = np.mean(n_whites, axis=0)
            n_grays_avg = np.mean(n_grays, axis=0)
            n_blacks_avg = np.mean(n_blacks, axis=0)

            writer.writerow([
                budget,
                round(n_whites_avg, 1), round(n_grays_avg, 1), round(n_blacks_avg, 1),
                0, round(avgs[1], 1), round(avgs[2], 1), round(avgs[3], 1),
            ])


def run_test_alg_on_g(drone_speed, c):
    with open('out/on_g/alg_on_g_c%.1f_n%d_v%d.csv' % (c, n, drone_speed), mode='w', newline='') as file:
        writer = csv.writer(file, delimiter=',')

        writer.writerow(["budget, whites, grays, blacks, abort, success, delivered, fail, fail_noway"])

        # drone budget [kJ]
        for budget in range(B_f, B_t, B_s):
            print(budget)

            # [], success, delivered, fail, fail_noway
            exp_statuses = np.zeros((n_scenarios, 5))
            n_whites = np.zeros(n_scenarios)
            n_grays = np.zeros(n_scenarios)
            n_blacks = np.zeros(n_scenarios)

            i = 0
            while i < n_scenarios:
                G, pos = create_graph(c)

                params = {
                    'pos': pos,
                    'v': drone_speed,
                    'ws': max_wind_speed,
                    'wd': 0,
                    'p': max_payload,
                    'B': budget,
                }

                colors = alg_pre_proc(G, params)

                n_white = len(colors[colors == 0])
                n_gray = len(colors[colors == 1])
                n_black = len(colors[colors == 2])

                if n_gray == 0:
                    continue

                # s_abort = 0
                s_success = 0
                s_delivered = 0
                s_fail = 0
                s_fail_noway = 0

                G_bak = copy.deepcopy(G)

                for k in range(1, n):
                    if colors[k] == 1:
                        dest = k

                        params = {
                            'pos': pos,
                            'v': drone_speed,
                            'ws': np.random.choice(winds),
                            'wd': np.random.randint(360),
                            'p': max_payload,
                        }
                        update_graph(G, params)

                        actual_len_go = actual_len_back = 0

                        src = 0

                        no_way = False

                        while True:
                            v = -1
                            min_cost = D**2
                            for v_i in nx.neighbors(G, src):
                                weight = G[src][v_i]['weight']
                                if weight < min_cost:
                                    min_cost = weight
                                    v = v_i

                            if v == -1:
                                no_way = True
                                break

                            if v != dest:
                                weight = G[src][v]['weight']
                                actual_len_go = actual_len_go + weight

                                G.remove_node(src)

                                params = {
                                    'pos': pos,
                                    'v': drone_speed,
                                    'ws': np.random.choice(winds),
                                    'wd': np.random.randint(360),
                                    'p': max_payload,
                                }
                                update_graph(G, params)
                                src = v
                            else:
                                break

                        G = copy.deepcopy(G_bak)

                        params = {
                            'pos': pos,
                            'v': drone_speed,
                            'ws': np.random.choice(winds),
                            'wd': np.random.randint(360),
                            'p': 0,
                        }
                        update_graph(G, params)

                        src = dest
                        dest = 0

                        if no_way:
                            s_fail_noway = s_fail_noway + 1
                        else:
                            while True:
                                v = -1
                                min_cost = D ** 2
                                for v_i in nx.neighbors(G, src):
                                    weight = G[src][v_i]['weight']
                                    if weight < min_cost:
                                        min_cost = weight
                                        v = v_i

                                if v == -1:
                                    no_way = True
                                    break

                                if v != dest:
                                    weight = G[src][v]['weight']
                                    actual_len_back = actual_len_back + weight

                                    G.remove_node(src)

                                    params = {
                                        'pos': pos,
                                        'v': drone_speed,
                                        'ws': np.random.choice(winds),
                                        'wd': np.random.randint(360),
                                        'p': max_payload,
                                    }
                                    update_graph(G, params)
                                    src = v
                                else:
                                    break

                            G = copy.deepcopy(G_bak)

                            actual_len = actual_len_go + actual_len_back

                            # print("Actual energy=%d" % actual_len)
                            if no_way:
                                s_fail_noway = s_fail_noway + 1
                            else:
                                if actual_len <= budget:
                                    s_success = s_success + 1
                                    # print("success")
                                else:
                                    if actual_len_go <= budget:
                                        s_delivered = s_delivered + 1
                                        # print("delivered")
                                    else:
                                        s_fail = s_fail + 1
                                        # print("fail")

                            # print("%d: gray (exp=%d, act=%d)" % (i, expected_len, actual_len))
                print("Gray=%d -> success=%d, delivered=%d, fail=%d, fail_noway=%d" % (n_gray, s_success, s_delivered, s_fail, s_fail_noway))

                exp_statuses[i][1] = 100. * s_success / n_gray
                exp_statuses[i][2] = 100. * s_delivered / n_gray
                exp_statuses[i][3] = 100. * s_fail / n_gray
                exp_statuses[i][4] = 100. * s_fail_noway / n_gray

                n_whites[i] = 100. * n_white / (n - 1)
                n_grays[i] = 100. * n_gray / (n - 1)
                n_blacks[i] = 100. * n_black / (n - 1)

                i = i + 1

            avgs = np.mean(exp_statuses, axis=0)
            stds = np.std(exp_statuses, axis=0)
            print(np.mean(n_grays, axis=0))
            print(avgs)
            # print(stds)

            n_whites_avg = np.mean(n_whites, axis=0)
            n_grays_avg = np.mean(n_grays, axis=0)
            n_blacks_avg = np.mean(n_blacks, axis=0)

            writer.writerow([
                budget,
                round(n_whites_avg, 1), round(n_grays_avg, 1), round(n_blacks_avg, 1),
                0, round(avgs[1], 1), round(avgs[2], 1), round(avgs[3], 1), round(avgs[4], 1),
            ])


if __name__ == '__main__':
    drone_speed = 20
    for c in [0.5, 1, 1.5, 2]:
        run_test_colors()
        run_test_alg_off_sp(drone_speed, c)
        run_test_alg_on_sp(drone_speed, c)
        run_test_alg_on_g(drone_speed, c)
