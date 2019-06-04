import shutil, os

year = 17
name = 'motmetrics'
# types = ['DPM-B0.5', 'FRCNN-B0.9', 'SDP-B0.6']
types = ['DPM0', 'FRCNN', 'SDP']
seqs = [2, 4, 5, 9, 10, 11, 13]  # the set of sequences
lengths = [600, 1050, 837, 525, 654, 900, 750]  # the length of the sequence

name2 = 'GN_VOT_NoRecover'
test_seqs = [1, 3, 6, 7, 8, 12, 14]

# copy the results for testing sets
for type in types:
    for i in xrange(len(seqs)):
        src_dir = 'Results/MOT%d/IoU/%02d/%d/%s_%s_4_cf_0.3_iou_0.5_vc_0.98__NoRecover_fgap_0_dseq_new/res_training.txt'\
                  %(year,
                   seqs[i],
                   lengths[i],
                   name,
                   type)

        des_d = '../Validation/%s/'%(name2)
        if not os.path.exists(des_d):
            os.mkdir(des_d)
        des_dir = des_d + 'MOT%d-%02d-%s.txt'%(year, seqs[i], type)

        print src_dir
        print des_dir
        shutil.copyfile(src_dir, des_dir)
