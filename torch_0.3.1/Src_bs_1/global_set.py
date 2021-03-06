u_initial = 1  # 1 - random, 0 - 0

edge_initial = 1  # 3 - 0.5, 2 - appearance similarity, 1 - random, 0 - IoU
#  Average epoch: 34.9520958084 for Random. The final time consuming:4245.31065106
#  Average epoch: 33.577245509 for IoU. The final time consuming:3791.01934195

criterion_s = 0  # 1 - MSELoss, 0 - CrossEntropyLoss

u_s, e_s = 1, 1  # u_s - Uphi, e_s - Ephi

train_test = 30  # n for training, n for testing
u_evaluation = 0  # 1 - initiate randomly, 0 - initiate with the u learned
# edge_init|                       random                       |
#  u_init  |        random           |            learned
#          |   pre-lr  |     lr      |     pre-lr    |    lr    |
#     33   |     |       |         |    |
#     99   |     |       |     0.1717    |   0.3157 |
