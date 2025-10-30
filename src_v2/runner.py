import os

# for i in range(1,5):
#     os.system(
#         f"python train.py --train_csv ../data/20230426_170611_new_rural_cat/train_new_rural_time{i}_with_cat.csv "
#         f"--val_csv ../data/20230426_170611_new_rural_cat/validation_new_rural_time{i}_with_cat.csv"
#         )
    
for time in ["4"]:
    os.system(
        f"python train.py --train_csv ../data/20230421_190706_prev_rural_cat/train_prev_rural_time{time}_with_cat.csv "
        f"--val_csv ../data/20230421_190706_prev_rural_cat/validation_prev_rural_time{time}_with_cat.csv"
        )
