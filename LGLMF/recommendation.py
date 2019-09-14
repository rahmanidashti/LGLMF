import time
import numpy as np
import scipy.sparse as sparse

from collections import defaultdict

from lib.LogisticMatrixFactorization import *
from lib.LocalGeographical import LocalGeographical
from lib.metrics import precisionk, recallk, ndcgk, mapk


def read_training_data():
    train_data = open(train_file, 'r').readlines()
    training_tuples = set()
    for eachline in train_data:
        uid, lid, _ = eachline.strip().split()
        uid, lid, = int(uid), int(lid)
        training_tuples.add((uid, lid))
    return training_tuples


def read_ground_truth():
    ground_truth = defaultdict(set)
    truth_data = open(test_file, 'r').readlines()
    for eachline in truth_data:
        uid, lid, _ = eachline.strip().split()
        uid, lid = int(uid), int(lid)
        ground_truth[uid].add(lid)
    return ground_truth


def main():
    training_tuples = read_training_data()
    ground_truth = read_ground_truth()

    start_time = time.time()

    LMF.train_model()
    LG.max_checkins(train_file)
    LG.read_poi_coos(poi_file)
    LG.rec_score()
    print("End of Train")

    elapsed_time = time.time() - start_time
    print("Done. Elapsed time:", elapsed_time, "s")

    execution_time = open("./result/execution_time" + ".txt", 'w')
    execution_time.write(str(elapsed_time))

    rec_list = open("./result/reclist_top_" + str(top_k) + ".txt", 'w')
    result_5 = open("./result/result_top_" + str(5) + ".txt", 'w')
    result_10 = open("./result/result_top_" + str(10) + ".txt", 'w')
    result_15 = open("./result/result_top_" + str(15) + ".txt", 'w')
    result_20 = open("./result/result_top_" + str(20) + ".txt", 'w')

    all_uids = list(range(user_num))
    all_lids = list(range(poi_num))
    np.random.shuffle(all_uids)

    # list for different ks
    precision_5, recall_5, nDCG_5, MAP_5 = [], [], [], []
    precision_10, recall_10, nDCG_10, MAP_10 = [], [], [], []
    precision_15, recall_15, nDCG_15, MAP_15 = [], [], [], []
    precision_20, recall_20, nDCG_20, MAP_20 = [], [], [], []

    for cnt, uid in enumerate(all_uids):
        if uid in ground_truth:
            overall_scores = [LG.predict(uid=uid, lid=lid) * LMF.predict_logistic(uid, lid)
                              if (uid, lid) not in training_tuples else -1
                              for lid in all_lids]
            overall_scores = np.array(overall_scores)

            predicted = list(reversed(overall_scores.argsort()))[:top_k]
            actual = ground_truth[uid]

            # calculate the average of different k
            precision_5.append(precisionk(actual, predicted[:5]))
            recall_5.append(recallk(actual, predicted[:5]))
            nDCG_5.append(ndcgk(actual, predicted[:5]))
            MAP_5.append(mapk(actual, predicted[:5], 5))

            precision_10.append(precisionk(actual, predicted[:10]))
            recall_10.append(recallk(actual, predicted[:10]))
            nDCG_10.append(ndcgk(actual, predicted[:10]))
            MAP_10.append(mapk(actual, predicted[:10], 10))

            precision_15.append(precisionk(actual, predicted[:15]))
            recall_15.append(recallk(actual, predicted[:15]))
            nDCG_15.append(ndcgk(actual, predicted[:15]))
            MAP_15.append(mapk(actual, predicted[:15], 15))

            precision_20.append(precisionk(actual, predicted[:20]))
            recall_20.append(recallk(actual, predicted[:20]))
            nDCG_20.append(ndcgk(actual, predicted[:20]))
            MAP_20.append(mapk(actual, predicted[:20], 20))

            print(cnt, uid, "pre@10:", np.mean(precision_10), "rec@10:", np.mean(recall_10))

            rec_list.write('\t'.join([
                str(cnt),
                str(uid),
                ','.join([str(lid) for lid in predicted])
            ]) + '\n')

            # write the different ks
            result_5.write('\t'.join([str(cnt), str(uid), str(np.mean(precision_5)), str(np.mean(recall_5)),
                                      str(np.mean(nDCG_5)), str(np.mean(MAP_5))]) + '\n')
            result_10.write('\t'.join([str(cnt), str(uid), str(np.mean(precision_10)), str(np.mean(recall_10)),
                                       str(np.mean(nDCG_10)), str(np.mean(MAP_10))]) + '\n')
            result_15.write('\t'.join([str(cnt), str(uid), str(np.mean(precision_15)), str(np.mean(recall_15)),
                                       str(np.mean(nDCG_15)), str(np.mean(MAP_15))]) + '\n')
            result_20.write('\t'.join([str(cnt), str(uid), str(np.mean(precision_20)), str(np.mean(recall_20)),
                                       str(np.mean(nDCG_20)), str(np.mean(MAP_20))]) + '\n')

    print("<< Task Finished >>")


if __name__ == '__main__':
    data_dir = "../datasets/gowalla/"

    size_file = data_dir + "Gowalla_data_size.txt"
    check_in_file = data_dir + "Gowalla_checkins.txt"
    train_file = data_dir + "Gowalla_train.txt"
    tune_file = data_dir + "Gowalla_tune.txt"
    test_file = data_dir + "Gowalla_test.txt"
    poi_file = data_dir + "Gowalla_poi_coos.txt"

    user_num, poi_num = open(size_file, 'r').readlines()[0].strip('\n').split()
    user_num, poi_num = int(user_num), int(poi_num)

    top_k = 100

    result = load_matrix(train_file, num_users=user_num, num_items=poi_num)
    LMF = LogisticMF(counts=result, num_factors=10)
    LG = LocalGeographical(user_num, poi_num)

    main()
