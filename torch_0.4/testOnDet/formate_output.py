import shutil


def deleteDir(del_dir):
    shutil.rmtree(del_dir)


def format():
    import os
    in_basic = 'Results/MOT16/'
    seqs = ['02', '04', '05', '09', '10', '11', '13']
    types = [['MotMetrics_IoU_cross_dets', 'MotMetrics_IoU_cross_gts'],
             ['MotMetrics_IoU_inner_dets', 'MotMetrics_IoU_inner_gts'],
             ['MotMetrics_IoU_inner_balanced_dets', 'MotMetrics_IoU_inner_balanced_gts'],
             ['MotMetrics_IoU_inner_balancedNearby_dets', 'MotMetrics_IoU_inner_balancedNearby_gts']]

    out_basic = in_basic+'MotMetrics/'
    if not os.path.exists(out_basic):
        os.mkdir(out_basic)
    else:
        deleteDir(out_basic)
        os.mkdir(out_basic)

    for type_2 in types:
        type, type2 = type_2
        in_dir = in_basic + type + '/'
        in_dir2 = in_basic + type2 + '/'

        print type
        tmp = type.split('_')[:-1]
        print tmp
        tmp = '_'.join(tmp)
        print tmp
        out_dir = out_basic + tmp + '/'
        print in_dir, out_dir
        if not os.path.exists(out_dir):
            os.mkdir(out_dir)
        else:
            deleteDir(out_dir)
            os.mkdir(out_dir)
            print out_dir

        for seq in seqs:
            in_txt = in_dir + seq+'.txt'
            in_txt2 = in_dir2 + seq + '.txt'
            out_txt = out_dir +seq+'.txt'
            print ' ', in_txt, in_txt2, out_txt

            f = open(in_txt, 'r')
            f2 = open(in_txt2, 'r')
            out = open(out_txt, 'w')
            for line in f.readlines():
                tag_f2 = 0
                line = line.strip().split(' ')
                line2 = f2.readline().strip().split(' ')
                print '0.0', line
                print '0.0', line2
                if line[0] == 'IDF1':
                    print >> out, '\t',
                elif line[0] == 'OVERALL':
                    continue
                elif line[0] == 'Testing':
                    tag_f2 = 1
                    line[0] = type.split('_')[-1]
                    line2[0] = type2.split('_')[-1]
                elif line[0] == 'The':
                    tag_f2 = 1
                    line = line[2:]
                    line2 = line2[2:]
                elif '*' in line[0]:
                    line = [line[1]]

                # output the motmetrics of the detections
                for word in line[:-1]:
                    if len(word):
                        print >> out, '%s\t'%word,
                if len(line[-1]):
                    print >> out, line[-1]

                # output the motmetrics of the gts
                if tag_f2:
                    for word in line2[:-1]:
                        if len(word):
                            print >> out, '%s\t' % word,
                    if len(line2[-1]):
                        print >> out, line2[-1]
            out.close()
            f.close()
format()