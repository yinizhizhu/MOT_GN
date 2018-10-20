import shutil

seqs = [2, 4, 5, 9, 10, 11, 13]
lengths = [600, 1050, 837, 525, 654, 900, 750]  # the length of the sequence
basis_dir = '../MOT/MOT17/train/'

# for i in xrange(len(seqs)):
#     seq = seqs[i]
#     src = basis_dir + 'MOT17-%02d-FRCNN/det/det.txt'%seq
#     des = basis_dir + 'MOT17-%02d-FRCNN/det/det_src.txt'%seq
#     print src, des
#     shutil.copyfile(src, des)


for i in xrange(len(seqs)):
    seq = seqs[i]
    length = lengths[i]

    src_dir = basis_dir + 'MOT17-%02d-FRCNN/det/det_src.txt'%seq
    src = open(src_dir, 'r')
    container = [[] for i in xrange(length+1)]
    for line in src.readlines():
        line = line.strip()
        attrs = line.split(',')
        index = int(attrs[0])
        container[index].append(line)
    src.close()

    des_dir = basis_dir + 'MOT17-%02d-FRCNN/det/det.txt'%seq
    des = open(des_dir, 'w')
    for i in xrange(1, length+1):
        for j in xrange(len(container[i])):
            # print container[i][j]
            print >> des, container[i][j]
    des.close()