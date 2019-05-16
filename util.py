import numpy as np
import pandas as pd
from collections import OrderedDict
import matplotlib.pyplot as plt
from collections import OrderedDict
from collections import Counter
import numpy as np
import pandas as pd
np.set_printoptions(precision=4)
from collections import defaultdict
from scipy.sparse.csgraph import minimum_spanning_tree


# Turn the JSON file to a dictionary {lbl, parent}
def walk(node, res={}):
    if 'children' in dict.keys(node):
        kids_list = node['children']
        for curr in kids_list:
            res.update({curr['name']: node['name']})
            walk(curr)
    else:
        return
    return res


# Map a label mid to its display name
def load_display_names(classes_filename):
    classes = pd.read_csv(classes_filename, names=['mid', 'name'])
    display_names = dict(zip(classes.mid, classes.name))
    return display_names


# Map { image id --> url }
def image_to_url(images_path):
    urls = pd.read_csv(images_path)
    id_url = dict(zip(urls.ImageID, urls.Thumbnail300KURL))
    return id_url


# Parse a DF into a dict {image -> associated labels}
def image_to_labels(annotations):
    img_to_labels, col_name = dict(), 'ImageID'
    images = annotations[col_name].unique().tolist()
    for i in images:
        img_to_labels[i] = annotations[annotations[col_name] == i][
            'LabelName'].values.tolist()
    return img_to_labels


# Load train, test, validation image - url files into df.
def load_urls_to_df(path_train, path_val, path_test):
    df_train = pd.read_csv(path_train)
    df_val = pd.read_csv(path_val)
    df_test = pd.read_csv(path_test)
    urls = pd.concat([df_train, df_val, df_test])
    urls.set_index('ImageID', inplace=True)
    return urls


def plot_px_vs_entropy(singles, num_images):
    font_size = 'x-large'
    p, h, xy = OrderedDict(), OrderedDict(), OrderedDict()
    for k, v in singles.items():
        px = float(v)/float(num_images)
        p[k] = px
        h[k] = -px*np.log2(px)-(1-px)*np.log2(1-px)
        xy[k] = (p[k], h[k])
    x_val = [x[0] for x in xy.values()]
    y_val = [x[1] for x in xy.values()]
    plt.scatter(x_val, y_val)
    plt.xlabel('p', fontsize=font_size)
    plt.ylabel('H(p)', fontsize=font_size)
    plt.show()


def calc_stats(img2annots, vocab_size):
    vocab_count = Counter()
    for annots_list in img2annots.values():
        for annot in annots_list:
            vocab_count[annot] += 1

    # annots2vocab = {}
    # for i, (annot, count) in enumerate(sorted(vocab_count.items(), key=operator.itemgetter(1)), 1):
    #     if count >= vocab_th:
    #         annots2vocab[annot] = (annot, i)
    #     else:
    #         annots2vocab[annot] = ("unk", 0)

    singelton_counts = dict(vocab_count.most_common(vocab_size))
    id2annot = dict(zip(range(len(singelton_counts)), singelton_counts.keys()))

    pairs_counts = defaultdict(lambda: 0)
    for annots_list in img2annots.values():
        for i in range(len(annots_list) - 1):
            for j in range(i+1, len(annots_list)):
                annot_name_1 = annots_list[i]
                annot_name_2 = annots_list[j]
                pairs_counts[frozenset([annot_name_1, annot_name_2])] += 1

    return singelton_counts, pairs_counts, id2annot


def build_graph(singelton_counts, pairs_counts, id2annot, total_size):
    np.seterr(all='raise')
    graph = np.zeros((len(id2annot), len(id2annot)))
    for i in range(len(id2annot) - 1):
        for j in range(i + 1, len(id2annot)):
            pair_set = frozenset([id2annot[i], id2annot[j]])
            i_1 = singelton_counts[id2annot[i]] / total_size
            i_0 = 1 - i_1
            j_1 = singelton_counts[id2annot[j]] / total_size
            j_0 = 1 - j_1
            slack = 1e-10  # to avoid division by zero
            mutual_0_0 = (total_size - singelton_counts[id2annot[i]] - singelton_counts[id2annot[j]] +
                          pairs_counts[pair_set]) / total_size
            if mutual_0_0 > 0:
                mutual_0_0 = mutual_0_0 * np.log(mutual_0_0 / (i_0 * j_0 + slack))
            mutual_0_1 = (singelton_counts[id2annot[j]] - pairs_counts[pair_set]) / total_size
            if mutual_0_1 > 0:
                mutual_0_1 = mutual_0_1 * np.log(mutual_0_1 / (i_0 * j_1 + slack))
            mutual_1_0 = (singelton_counts[id2annot[i]] - pairs_counts[pair_set]) / total_size
            if mutual_1_0 > 0:
                mutual_1_0 = mutual_1_0 * np.log(mutual_1_0 / (i_1 * j_0 + slack))
            mutual_1_1 = pairs_counts[pair_set] / total_size
            if mutual_1_1 > 0:
                mutual_1_1 = mutual_1_1 * np.log(mutual_1_1 / (i_1 * j_1 + slack))
            # using negative weights to get maximum spanning tree
            graph[i, j] = - (mutual_0_0 + mutual_0_1 + mutual_1_0 + mutual_1_1)

    return graph


def build_chow_liu_tree(graph, id2annot):
    tree = minimum_spanning_tree(graph)
    chow_liu_tree = dict()
    for i, row in enumerate(tree):
        childern = [id2annot[x] for x in np.nonzero(np.squeeze(row.toarray()))[0]]
        chow_liu_tree[id2annot[i]] = childern

    return chow_liu_tree



# def calc_counts(img2annots, annots2vocab):
#     singelton_counts = defaultdict(lambda: 0)
#     pairs_counts = defaultdict(lambda: {})
#     for annots_list in img2annots.values():
#         for annot in annots_list:
#             singelton_counts[annots2vocab[annot][0]] += 1
#
#         for i in range(len(annots_list) - 1):
#             for j in range(i+1, len(annots_list)):
#                 annot_name_1 = annots2vocab[annots_list[i]][0]
#                 annot_name_2 = annots2vocab[annots_list[j]][0]
#                 pairs_counts[frozenset([annot_name_1, annot_name_2])] += 1
#
#     return singelton_counts, pairs_counts

