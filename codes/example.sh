#for EPFO queries
CUDA_VISIBLE_DEVICES=0 python run_gmm_Model_final.py --data_name "NELL-q2b" -b 1024 -k 2 -lr 0.0007 --warm_up_steps 1000 --dropout_rate 0.3 -wc 0 -ls 0.2 --data_path "../data/"

CUDA_VISIBLE_DEVICES=0 python run_gmm_Model_final.py --data_name "FB15k-237-q2b" -b 2048 -k 2 -lr 0.001 --warm_up_steps 1000 --dropout_rate 0.2 -wc 0 -ls 0.2 --data_path "../data/"

CUDA_VISIBLE_DEVICES=0 python run_gmm_Model_final.py --data_name "FB15k-q2b" -b 2048 -k 2 -lr 0.001 --warm_up_steps 1000 --dropout_rate 0.1 -wc 0 -ls 0.2 --data_path "../data/"

#for queries with negation
CUDA_VISIBLE_DEVICES=0 python run_gmm_Model_final_inv.py --data_name "NELL-betae" -b 3000 -k 2 -lr 0.0007 --warm_up_steps 1000 --dropout_rate 0.3 -wc 0 -ls 0.7 --data_path "../data/"

CUDA_VISIBLE_DEVICES=0 python run_gmm_Model_final.py --data_name "FB15k-237-betae" -b 8192 -k 2 -lr 0.001 --warm_up_steps 1000 --dropout_rate 0.1 -wc 0 -ls 0.5 --data_path "../data/"

CUDA_VISIBLE_DEVICES=0 python run_gmm_Model_final.py --data_name "FB15k-betae" -b 8192 -k 2 -lr 0.001 --warm_up_steps 1000 --dropout_rate 0.1 -wc 0 -ls 0.5 --data_path "../data/"
