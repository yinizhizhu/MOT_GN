edge_initial = 1


def get_id_num(res_dir):
    data = open(res_dir, 'r')
    id, num = 0, 0
    for line in data.readlines():
        line = line.strip().split(',')
        tmp = int(line[1])
        id = max(tmp, id)
        num += 1
    data.close()
    return id, num

if edge_initial:
    init_dir = 'Random'
else:
    init_dir = 'IoU'

year = 17
seqs = [2, 4, 5, 9, 10, 11, 13]  # the set of sequences
lengths = [600, 1050, 837, 525, 654, 900, 750]  # the length of the sequence

names = ['MOT_M_ANew_bb', 'MOT_M_ANew_bb_uupdate_tau_0.9_0.7', 'MOT_M_ANew_bb_uupdate_tau_0.5_0.5', 'MOT_M_ANew_bb_uupdate_tau_0.5_0.5_crowded', 'App2_bb', 'Motion1_bb']
test_seqs = [1, 3, 6, 7, 8, 12, 14]

types = ['DPM', 'FRCNN', 'SDP']

out = open('id_num.txt', 'w')
for type in types:
    print >> out, type
    for i in xrange(len(seqs)):
        print >> out, '     ', seqs[i], lengths[i]
        for name in names:
            des_d = '%s/'%(name)
            des_dir = des_d + 'MOT%d-%02d-%s.txt'%(year, test_seqs[i], type)
            print des_dir
            id, num = get_id_num(des_dir)
            print >> out, '         %s: %d\t%d'%(name, id, num)
        print '****'*12
        print >> out, '***'*12
out.close()