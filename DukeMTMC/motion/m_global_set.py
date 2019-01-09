u_initial = 1  # 1 - random, 0 - 0

edge_initial = 1  # 3 - 0.5, 2 - appearance similarity, 1 - random, 0 - IoU
#  Average epoch: 34.9520958084 for Random. The final time consuming:4245.31065106
#  Average epoch: 33.577245509 for IoU. The final time consuming:3791.01934195

criterion_s = 0  # 1 - MSELoss, 0 - CrossEntropyLoss
u_s, e_s = 1, 1  # u_s - Uphi, e_s - Ephi
train_test = 100  # n for training, n for testing
u_evaluation = 0  # 1 - initiate randomly, 0 - initiate with the u learned
test_gt_det = 0  # 1 - detections of gt, 0 - detections of det



u_update = 1  # 1 - update when testing, 0 - without updating
if u_update:
    u_dir = '_uupdate'
else:
    u_dir = ''

decay = 1.9
decay_dir = '_decay'

f_gap = 5
if f_gap == 5:
    recover_dir = '_Recover'
else:
    recover_dir = '_NoRecover'



tau_threshold = 1.0  # The threshold of matching cost
tau_dis = 2.0   # The times of the current bbx's scale
gap = 25    # max frame number for side connection

show_recovering = 0  # 1 - 11, 0 - 10

overlap = 0.85

SEQLEN = 177841

# edge_init|                       random                       |
#  u_init  |        random           |            learned
#          |   pre-lr  |     lr      |     pre-lr    |    lr    |
#     33   |     |       |         |    |
#     99   |     |       |     0.1717    |   0.3157 |

# heads = [0, 5543, 3607, 27244, 31182, 1, 22402, 18968, 46766]
# trainval = [49700, 227540]
#
# head = trainval[0] - heads[5] + 1
# tail = head + SEQLEN - 1
# print head, tail


# OSError: [Errno 4] Interrupted system call
