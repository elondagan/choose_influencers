import numpy as np
import pandas as pd
import networkx as nx


# ------------------------------------------------------------------------------ predict graph state in each time-------

# predict matrices ____________________________________________

def get_IDs(csv_path):
    data = pd.read_csv(csv_path)
    data = dict(data)
    IDs = set((list(sorted(set(data['userID']))) + list(sorted((set(data['friendID']))))))
    IDs = list(IDs)
    IDs.sort()
    return IDs


def to_mat(csv_path):
    """
    convert a csv file to matrix
    :param csv_path: csv file path
    :return: neighborhood matrix
    """
    data = pd.read_csv(csv_path)
    data = dict(data)
    IDs = get_IDs(csv_path)
    mat_size = len(IDs)
    mat0 = np.zeros((mat_size, mat_size))
    for k in range(0, len(data['userID'])):
        i, j = IDs.index(data['userID'][k]), IDs.index(data['friendID'][k])
        mat0[i, j] = 1
        mat0[j, i] = 1
    for i in range(mat_size):
        mat0[i, i] = 1
    return mat0


def predict(mat0):
    """
    predict the next neighborhood matrix
    :param mat0: current matrix
    :return: next matrix
    """
    mat_size = len(mat0[0])
    p_mat = np.zeros((mat_size, mat_size))
    tot_friends = np.zeros(mat_size)

    for i in range(mat_size):
        tot_friends[i] = sum(mat0[i, :])

    sim_mat = np.zeros((mat_size, mat_size))
    for i in range(mat_size):
        for j in range(i, mat_size):
            sim_mat[i, j] = mat0[i, :] @ mat0[:, j]
            sim_mat[j, i] = sim_mat[i, j]

    for i in range(mat_size):
        for j in range(i, mat_size):
            k0 = sim_mat[i, j]  # k0 =: num of shared friends
            k1 = tot_friends[i]  # k1 =: number of friends i has
            k2 = tot_friends[j]  # k2 =: number of friends j has
            if mat0[i, j] == 0:
                p_mat[i, j] = min(1, (((1 / (k1 + 0.01)) + (1 / (k2 + 0.01))) * k0 * 3 + 0.05))
                p_mat[j, i] = min(1, (((1 / (k1 + 0.01)) + (1 / (k2 + 0.01))) * k0 * 3 + 0.05))
            else:
                p_mat[i, j] = 1

    # create the predicted matrix
    predicted_mat = np.zeros((mat_size, mat_size))
    for i in range(mat_size):
        for j in range(i, mat_size):
            r = np.random.rand()
            if r <= p_mat[i, j]:
                predicted_mat[i, j] = 1
                predicted_mat[j, i] = 1
    return predicted_mat


# evaluate prediction____________________________________________


def evaluate(prev_mat, cur_mat, predicted_mat):
    print(f"TP: {get_TP(prev_mat, cur_mat, predicted_mat)}")
    print(f"TN: {get_TN(prev_mat, cur_mat, predicted_mat)}")
    print(f"FP: {get_FP(prev_mat, cur_mat, predicted_mat)}")
    print(f"FN: {get_FN(prev_mat, cur_mat, predicted_mat)}")
    print(f"precision: {get_precision(prev_mat, cur_mat, predicted_mat)}")
    print(f"recall: {get_recall(prev_mat, cur_mat, predicted_mat)}")


def get_TP(mat0, mat1, predicted_mat1):
    true = 0
    all = 0
    for i in range(len(mat0)):
        for j in range(len(mat0)):
            if mat0[i, j] == 0 and mat1[i, j] == 1:
                if predicted_mat1[i, j] == 1:
                    true += 1
                all += 1
    return (true / all) * 100


def get_FN(mat0, mat1, predicted_mat1):
    wrong = 0
    all = 0
    for i in range(len(mat0)):
        for j in range(len(mat0)):
            if mat0[i, j] == 0 and mat1[i, j] == 1:
                if predicted_mat1[i, j] == 0:
                    wrong += 1
                all += 1
    return (wrong / all) * 100


def get_FP(mat0, mat1, predicted_mat1):
    wrong = 0
    all = 0
    for i in range(len(mat0)):
        for j in range(len(mat0)):
            if mat0[i, j] == 0 and mat1[i, j] == 0:
                if predicted_mat1[i, j] == 1:
                    wrong += 1
                all += 1
    return (wrong / all) * 100


def get_TN(mat0, mat1, predicted_mat1):
    true = 0
    all = 0
    for i in range(len(mat0)):
        for j in range(len(mat0)):
            if mat0[i, j] == 0 and mat1[i, j] == 0:
                if predicted_mat1[i, j] == 0:
                    true += 1
                all += 1
    return (true / all) * 100


def get_precision(mat0, mat1, predicted_mat1):
    return get_TP(mat0, mat1, predicted_mat1) / (
            get_FP(mat0, mat1, predicted_mat1) + get_TP(mat0, mat1, predicted_mat1))


def get_recall(mat0, mat1, predicted_mat1):
    return get_TP(mat0, mat1, predicted_mat1) / (
            get_FN(mat0, mat1, predicted_mat1) + get_TP(mat0, mat1, predicted_mat1))


# convert matrices to graphs ____________________________________________

def get_predicated_states(csv_path1, artist):
    """
    predict graph for each time period based
    :param csv_path0: friends state in time -1
    :param csv_path1: friends state in time 0
    :return: 7 graphs for each time t =0...6
    """
    all_ids = get_IDs(csv_path1)
    graphs = []

    mat0 = to_mat(csv_path1)
    graphs.append(set_graph(mat0, artist, all_ids))

    mat1 = predict(mat0)
    graphs.append(set_graph(mat1, artist, all_ids))

    mat2 = predict(mat1)
    graphs.append(set_graph(mat2, artist, all_ids))

    mat3 = predict(mat2)
    graphs.append(set_graph(mat3, artist, all_ids))

    mat4 = predict(mat3)
    graphs.append(set_graph(mat4, artist, all_ids))

    mat5 = predict(mat4)
    graphs.append(set_graph(mat5, artist, all_ids))

    mat6 = predict(mat5)
    graphs.append(set_graph(mat6, artist, all_ids))

    return graphs


def set_graph(mat, artist, ids):
    spotifly = np.array(pd.read_csv('spotifly.csv'))

    graph = nx.Graph()
    graph.add_nodes_from(ids, plays=0, purchased=0)

    # add edges to graph if two ids are friends
    mat_size = len(mat[0])
    for i in range(mat_size):
        for j in range(i, mat_size):
            if i == j:
                continue
            if mat[i, j] == 1:
                graph.add_edge(ids[i], ids[j])

    # update plays for current artist
    spotifly_size = len(spotifly[:, 0])
    for i in range(spotifly_size):
        if spotifly[i, 1] == artist:
            graph.nodes[spotifly[i, 0]]['plays'] = spotifly[i, 2]

    return graph


# ------------------------------------------------------------------------------ reduce influencers possibilities ------

def find_fans(G, playlist, artist):
    fans = playlist.groupby([" artistID"])
    croud = fans.get_group(artist)
    for row in croud.iterrows():
        user = row[1][0]
        plays = row[1][2]
        G.nodes[user]["num_listened"] = plays


def initialize():
    spotifly_df = pd.DataFrame(pd.read_csv("spotifly.csv"))
    spotifly = dict(spotifly_df)

    data_1 = pd.read_csv('instaglam0.csv')
    data_1 = dict(data_1)
    IDs1 = set((list(sorted(set(data_1['userID']))) + list(sorted((set(data_1['friendID']))))))
    IDs1 = list(IDs1)
    IDs1.sort()

    g_1 = nx.Graph()
    for user in spotifly_df.iterrows():
        g_1.add_node(user[1][0], purchased=0, num_listened=0, rank=0, a=1, h=1)

    for i in range(len(data_1['userID'])):
        g_1.add_edge(data_1['userID'][i], data_1['friendID'][i])

    return g_1, IDs1, spotifly_df


def find_bt(G, node):
    neighbors = G.neighbors(node)
    counter = 0
    for v in neighbors:
        if G.nodes[v]["purchased"] == 1:
            counter += 1
    return counter


def calculate_p(G, node):
    nt = G.degree[node]
    bt = find_bt(G, node)
    if G.nodes[node]["num_listened"] == 0:
        return bt / nt

    else:
        h = G.nodes[node]["num_listened"] / 1000
        res = min(1, h * bt / nt)
        return res


def give_rank(G, artist):
    for node in G.nodes():
        rank = 0
        friends = list(G.neighbors(node))
        # print(friends)
        if len(friends) > 0:
            for friend in friends:
                if G.nodes[friend]["num_listened"] > 0:
                    rank += ((G.degree[friend] * G.nodes[friend]["num_listened"] / 1000) + 1)
                else:
                    rank += G.degree[friend] + 1
        G.nodes[node]["rank"] = rank
    nodes = G.nodes()
    nodes = sorted(nodes, key=lambda x: (G.nodes[x]['rank'], G.degree[x]), reverse=True)

    influencers = [nodes[:15]] + [artist]
    return influencers


def find_potential_influencers():
    artists = [144882, 194647, 511147, 532992]
    result = []
    for artist in artists:
        g, IDs1, spotifly_df = initialize()

        find_fans(g, spotifly_df, artist)

        result.append(give_rank(g, artist))

    final_result = []
    for i in range(len(result)):
        final_result.append(result[0][0])
    return final_result


# ------------------------------------------------------------------------------ find best possible influencers---------

def update_graphs(graphs, id):
    """
    update all graphs that id have the product
    :param graphs: 7 graphs
    :param id: id to update
    """
    for t in range(7):
        graphs[t].nodes[id]['purchased'] = 1


def clear_graph(graphs, all_ids):
    for t in range(7):
        for id in all_ids:
            graphs[t].nodes[id]['purchased'] = 0


def cal_prob(Nt, Bt, h):
    # calculate person purchases probability
    if h != 0:
        return (h * Bt) / (1000 * Nt)
    else:
        return Bt / Nt


def get_friends_status(graph, friends_ids):
    # return the number of friends who purchased the product
    counter = 0
    for friend in friends_ids:
        if graph.nodes[friend]['purchased'] == 1:
            counter += 1
    return counter


def find_influencers_for_artist(graphs, possible_ids):
    spreaders_per_id = []

    # run a simulation:
    for index0, initial_influencer in enumerate(possible_ids):
        clear_graph(graphs, get_IDs('instaglam0.csv'))
        spreaders = set()
        spreaders.add(initial_influencer)
        update_graphs(graphs, initial_influencer)

        for t in range(1, 7):
            temp = set()
            already_tried = set()
            for s in spreaders:
                s_friends = graphs[t].neighbors(s)
                for friend in s_friends:

                    if graphs[t].nodes[friend]['purchased'] == 1 or friend in already_tried:
                        continue
                    else:
                        already_tried.add(friend)
                        Nt = len(list(graphs[t].neighbors(friend)))
                        Bt = get_friends_status(graphs[t - 1], graphs[t].neighbors(friend))
                        h = graphs[t].nodes[friend]['plays']
                        if np.random.rand() < cal_prob(Nt, Bt, h):
                            update_graphs(graphs, friend)
                            temp.add(friend)
            spreaders = spreaders.union(temp)

        spreaders_per_id.append(spreaders)

    return spreaders_per_id


def choose_5(res_list, ids):
    size = len(res_list)
    max_spread = 0
    decision = []

    for a in range(size):
        for b in range(size):
            if b == a:
                continue
            for c in range(size):
                if c == b or c == a:
                    continue
                for d in range(size):
                    if d == c or d == b or d == a:
                        continue
                    for e in range(size):
                        if e == a or e == d or e == c or e == b or e == a:
                            continue
                        cur_spread = len(
                            res_list[a].union(res_list[b].union(res_list[c].union(res_list[d].union(res_list[e])))))
                        if cur_spread > max_spread:
                            max_spread = cur_spread
                            decision = [ids[a], ids[b], ids[c], ids[d], ids[e]]
    return decision, max_spread


if __name__ == '__main__':
    csv_path_m1, csv_path0 = 'instaglam_1.csv', 'instaglam0.csv'
    artists = [144882, 194647, 511147, 532992]
    # artists = [70, 150, 989, 16326]

    all_ids = get_IDs(csv_path_m1)

    filtered_ids = find_potential_influencers()
    influencers = []

    for i, artist in enumerate(artists):
        graphs = get_predicated_states(csv_path0, artist)
        print(f"finding best influencers for artist {artist}...")
        options = find_influencers_for_artist(graphs, filtered_ids[i])
        res = choose_5(options, filtered_ids[i])
        print(f"best influencers are: {res[0]} with spreading of {res[1]/len(all_ids)}%")

