import os, shutil

shutil.rmtree('Case')

if not os.path.exists('Case/'):
    os.mkdir('Case/')

if not os.path.exists('Case/App2'):
    os.mkdir('Case/App2')

if not os.path.exists('Case/Motion1'):
    os.mkdir('Case/Motion1')

if not os.path.exists('Case/M_A_1'):
    os.mkdir('Case/M_A_1')

if not os.path.exists('Case/M_A_2'):
    os.mkdir('Case/M_A_2')


def copy_img(seq, frames, o_dir):
    for f in frames:
        basic_dir = '../MOT/MOT16/test/MOT16-%02d/img1/%06d.jpg'%(seq, f)
        out_dir = o_dir + '%06d.jpg'%f
        shutil.copyfile(basic_dir, out_dir)


def get_two(res_dir, out_dir, fs):
    out = open(out_dir, 'w')
    data = open(res_dir, 'r')
    for line in data.readlines():
        line = line.strip()
        attrs = line.split(',')
        index = int(attrs[0])
        if index in fs:
            # print line
            print >> out, line
    data.close()
    out.close()


names = ['App2_bb', 'Motion1_bb', 'MOT_M_ANew_bb_uupdate_tau_0.5_0.5_crowded']

# Bad case for App2
test_seq = 3
detection = 'SDP'
frames = [916, 917]
o_dir = 'Case/App2/'
copy_img(test_seq, frames, o_dir)
for i in xrange(len(names)):
    name = names[i]
    out_dir = o_dir + '%d.txt'%(i+1)
    res_dir = '%s/MOT17-%02d-%s.txt'%(name, test_seq, detection)
    get_two(res_dir, out_dir, frames)
print test_seq, detection

# Bad case for Motion1
test_seq = 3
detection = 'SDP'
frames = [677, 679]
o_dir = 'Case/Motion1/'
copy_img(test_seq, frames, o_dir)
for i in xrange(len(names)):
    name = names[i]
    out_dir = o_dir + '%d.txt'%(i+1)
    res_dir = '%s/MOT17-%02d-%s.txt'%(name, test_seq, detection)
    get_two(res_dir, out_dir, frames)
print test_seq, detection


names = ['', '', 'MOT_M_ANew_bb_uupdate_tau_0.5_0.5_crowded', 'DMAN']

# Bad case1 for M_A
test_seq = 3
detection = 'SDP'
frames = [897, 905]
o_dir = 'Case/M_A_1/'
copy_img(test_seq, frames, o_dir)
for i in xrange(2, len(names)):
    name = names[i]
    out_dir = o_dir + '%d.txt'%(i+1)
    res_dir = '%s/MOT17-%02d-%s.txt'%(name, test_seq, detection)
    get_two(res_dir, out_dir, frames)
print test_seq, detection

# Bad case2 for M_A
test_seq = 8
detection = 'SDP'
frames = [163, 164]
o_dir = 'Case/M_A_2/'
copy_img(test_seq, frames, o_dir)
for i in xrange(2, len(names)):
    name = names[i]
    out_dir = o_dir + '%d.txt'%(i+1)
    res_dir = '%s/MOT17-%02d-%s.txt'%(name, test_seq, detection)
    get_two(res_dir, out_dir, frames)
print test_seq, detection
