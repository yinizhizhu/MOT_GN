u_initial = 1  # 1 - random, 0 - 0

edge_initial = 0  # 3 - 0.5, 2 - appearance similarity, 1 - random, 0 - IoU
#  Average epoch: 34.9520958084 for Random. The final time consuming:4245.31065106
#  Average epoch: 33.577245509 for IoU. The final time consuming:3791.01934195

criterion_s = 0  # 1 - MSELoss, 0 - CrossEntropyLoss

u_s, e_s = 1, 1  # u_s - Uphi, e_s - Ephi

train_test = 100  # n for training, n for testing
u_evaluation = 0  # 1 - initiate randomly, 0 - initiate with the u learned

test_gt_det = 0  # 1 - detections of gt, 0 - detections of det

tau_conf_score = 0.0  # The threshold of confidence score

tau_threshold = 1.0  # The threshold of matching cost

tau_dis = 2.0   # The times of the current bbx's scale

tau_vr = 0.5    # The visibility ratio shouldn't be too small
tau_frame = 25  # The difference of the anchor and positive should be near

gap = 25    # max frame number for side connection
f_gap = 5   # max frame number for recovering

show_recovering = 0  # 1 - 11, 0 - 10

app_fine_tune = 1   # 1 - fine-tuned the appearance model, 0 - pre-trained appearance model
fine_tune_dir = '../MOT/Fine-tune_GPU_5_3_60_aug/appearance_19.pth'

overlap = 0.85  # the IoU

rho = 0.5   # f = rho*f + (1-rho)*f

# edge_init|                       random                       |
#  u_init  |        random           |            learned
#          |   pre-lr  |     lr      |     pre-lr    |    lr    |
#     33   |     |       |         |    |
#     99   |     |       |     0.1717    |   0.3157 |