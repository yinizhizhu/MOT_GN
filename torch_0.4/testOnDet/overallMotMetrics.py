import time, os
import motmetrics as mm


def motMetrics(gt_dir, res_dir):
    df_gt = mm.io.loadtxt(gt_dir)
    df_test = mm.io.loadtxt(res_dir)
    return mm.utils.compare_to_groundtruth(df_gt, df_test, 'iou', distth=0.5)


def overallMotMetric_inner():
    basic_dir = 'Results/MOT16/IoU/'
    seqs = ['02', '04', '05', '09', '10', '11', '13']
    lengths = ['100', '200']
    settings = ['balanced', 'balanced_FP', 'balancedNearby', 'balancedNearby_FP']
    types = ['training', 'validation']

    out = open(basic_dir + 'overallMotMetrics_inner.txt', 'w')
    for length in lengths:
        for setting in settings:
            for type in types:
                print (length, setting, type)
                start = time.time()
                accs = []
                names = []
                for seq in seqs:
                    cur_dir = basic_dir+'%s/%s/motmetrics_inner_%s_dets/'%(seq, length, setting)
                    gt_dir = cur_dir+'gt_%s.txt'%type
                    res_dir = cur_dir + 'res_%s.txt'%type
                    if not os.path.exists(gt_dir):
                        print ('    There is no', gt_dir)
                    if not os.path.exists(res_dir):
                        print ("    There is no", res_dir)
                    # print (gt_dir, res_dir)
                    accs.append(motMetrics(gt_dir, res_dir))
                    names.append(seq)

                mh = mm.metrics.create()
                summary = mh.compute_many(accs, metrics=mm.metrics.motchallenge_metrics, names=names, generate_overall=True)
                print(mm.io.render_summary(summary, namemap=mm.io.motchallenge_metric_names, formatters=mh.formatters))
                print ('Time consuming:', time.time()-start)
                print()
    # out.write("['%s','%s']\n"%(args.gt, args.res))
    out.close()


# overallMotMetric_inner()


def overallMotMetric_cross():
    basic_dir = 'Results/MOT16/IoU/'
    seqs = ['02', '04', '05', '09', '10', '11', '13']
    lengths = ['600', '1050', '837', '525', '654', '900', '750']  # the length of the sequence

    out = open(basic_dir + 'overallMotMetrics_cross.txt', 'w')

    start = time.time()
    accs = []
    names = []
    for i in range(7):
        seq = seqs[i]
        length = lengths[i]
        cur_dir = basic_dir+'%s/%s/motmetrics_cross_dets/'%(seq, length)
        gt_dir = cur_dir+'gt_training.txt'
        res_dir = cur_dir + 'res_training.txt'
        if not os.path.exists(gt_dir):
            print ('    There is no', gt_dir)
        if not os.path.exists(res_dir):
            print ("    There is no", res_dir)
        print (gt_dir, res_dir)
        accs.append(motMetrics(gt_dir, res_dir))
        names.append(seq)

    mh = mm.metrics.create()
    summary = mh.compute_many(accs, metrics=mm.metrics.motchallenge_metrics, names=names, generate_overall=True)
    print(mm.io.render_summary(summary, namemap=mm.io.motchallenge_metric_names, formatters=mh.formatters))
    print ('Time consuming:', time.time()-start)
    print()
    # out.write("['%s','%s']\n"%(args.gt, args.res))
    out.close()


overallMotMetric_cross()