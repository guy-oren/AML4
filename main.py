import numpy as np
import pandas as pd
import util
import vis
import os
import pickle
import argparse

np.set_printoptions(precision=4)
pd.set_option('precision', 2)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--vocab_size", type=int, help="Maximum number of annotations to process, default is 400",
                        default=400)

    args = parser.parse_args()

    vocab_size = args.vocab_size
    oid_data = 'data/annotations-machine.csv'
    classes_fn = 'data/class-descriptions.csv'

    # Mapping between class lablel and class name
    classes_display_name = util.load_display_names(classes_fn)

    if not os.path.exists('data/img2annot.pickle'):
        annots = pd.read_csv(oid_data)
        img2annot = util.image_to_labels(annots)
        with open('data/img2annot.pickle', 'wb') as ofp:
            pickle.dump(img2annot, ofp)
    else:
        with open('data/img2annot.pickle', 'rb') as ifp:
            img2annot = pickle.load(ifp)

    singletion_counts, pairs_counts, id2annot = util.calc_stats(img2annot, vocab_size)

    samples_size = len(img2annot)
    graph = util.build_graph(singletion_counts, pairs_counts, id2annot, samples_size)

    # Dictionary with mapping between each Node and its childern nodes.
    # use for each node the class lablel
    chow_liu_tree = util.build_chow_liu_tree(graph, id2annot)

    vis.plot_network(chow_liu_tree, classes_display_name)


if __name__ == '__main__':
    main()
