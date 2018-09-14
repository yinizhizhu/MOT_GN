import motmetrics as mm
import pandas as pd
import numpy as np
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('gt', type=str)
parser.add_argument('res', type=str)
args = parser.parse_args()

# print(args)
# print (args.gt)
# print (args.res)


def motMetrics():
    df_gt = mm.io.loadtxt(args.gt)
    df_test = mm.io.loadtxt(args.res)
    return mm.utils.compare_to_groundtruth(df_gt, df_test, 'iou', distth=0.5)

def compute_motchallenge():
    df_gt = mm.io.loadtxt('/home/lee/Desktop/Evaluation/gt.txt')
    df_test = mm.io.loadtxt('/home/lee/Desktop/Evaluation/result.txt')
    return mm.utils.compare_to_groundtruth(df_gt, df_test, 'iou', distth=0.5)


def compute_motchallenge1():
    df_gt = mm.io.loadtxt('/home/lee/Desktop/Evaluation/gt.txt')
    df_test = mm.io.loadtxt('/home/lee/Desktop/Evaluation/result1.txt')
    return mm.utils.compare_to_groundtruth(df_gt, df_test, 'iou', distth=0.5)

accs = [motMetrics()]

# For testing
# [a.events.to_pickle(n) for (a,n) in zip(accs, dnames)]

mh = mm.metrics.create()
summary = mh.compute_many(accs, metrics=mm.metrics.motchallenge_metrics, names=['Testing'], generate_overall=True)

print()
print(mm.io.render_summary(summary, namemap=mm.io.motchallenge_metric_names, formatters=mh.formatters))

# expected = pd.DataFrame([
#     [0.557659, 0.729730, 0.451253, 0.582173, 0.941441, 8.0, 1, 6, 1, 13, 150, 7, 7, 0.526462, 0.277201],
#     [0.644619, 0.819760, 0.531142, 0.608997, 0.939920, 10.0, 5, 4, 1, 45, 452, 7, 6, 0.564014, 0.345904],
#     [0.624296, 0.799176, 0.512211, 0.602640, 0.940268, 18.0, 6, 10, 2, 58, 602, 14, 13, 0.555116, 0.330177],
# ])
#
# np.testing.assert_allclose(summary, expected, atol=1e-3)
