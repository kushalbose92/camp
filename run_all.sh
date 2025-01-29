
# python -u heterophilous_main.py --dataset 'Roman-empire' --train_lr 0.01 --seed 0 --num_layers 2 --mlp_layers 1 --hidden_dim 512 --train_iter 1000 --test_iter 1 --use_saved_model False --dropout 0.20 --train_w_decay 0.0 --num_splits 10 --batch_size None  --centrality 'degree' --device 'cuda:0' | tee output/roman_empire_2.txt

# python -u heterophilous_main.py --dataset 'Roman-empire' --train_lr 0.01 --seed 0 --num_layers 4 --mlp_layers 1 --hidden_dim 512 --train_iter 1000 --test_iter 1 --use_saved_model False --dropout 0.20 --train_w_decay 0.0 --num_splits 10 --batch_size None  --centrality 'degree' --device 'cuda:0' | tee output/roman_empire_4_none.txt
# python -u heterophilous_main.py --dataset 'Roman-empire' --train_lr 0.01 --seed 0 --num_layers 4 --mlp_layers 1 --hidden_dim 512 --train_iter 1000 --test_iter 1 --use_saved_model False --dropout 0.20 --train_w_decay 0.0 --num_splits 10 --batch_size 8192  --centrality 'degree' --device 'cuda:0' | tee output/roman_empire_4.txt

# python -u heterophilous_main.py --dataset 'Roman-empire' --train_lr 0.01 --seed 0 --num_layers 8 --mlp_layers 1 --hidden_dim 512 --train_iter 1000 --test_iter 1 --use_saved_model False --dropout 0.20 --train_w_decay 0.0 --num_splits 10 --batch_size None  --centrality 'degree' --device 'cuda:0' | tee output/roman_empire_8_none.txt
# python -u heterophilous_main.py --dataset 'Roman-empire' --train_lr 0.01 --seed 0 --num_layers 8 --mlp_layers 1 --hidden_dim 512 --train_iter 1000 --test_iter 1 --use_saved_model False --dropout 0.20 --train_w_decay 0.0 --num_splits 10 --batch_size 4096  --centrality 'degree' --device 'cuda:0' | tee output/roman_empire_8.txt

# python -u heterophilous_main.py --dataset 'Roman-empire' --train_lr 0.01 --seed 0 --num_layers 16 --mlp_layers 1 --hidden_dim 512 --train_iter 1000 --test_iter 1 --use_saved_model False --dropout 0.20 --train_w_decay 0.0 --num_splits 10 --batch_size None  --centrality 'degree' --device 'cuda:0' | tee output/roman_empire_16_none.txt
# python -u heterophilous_main.py --dataset 'Roman-empire' --train_lr 0.01 --seed 0 --num_layers 16 --mlp_layers 1 --hidden_dim 512 --train_iter 1000 --test_iter 1 --use_saved_model False --dropout 0.20 --train_w_decay 0.0 --num_splits 10 --batch_size 4096  --centrality 'degree' --device 'cuda:0' | tee output/roman_empire_16.txt

# python -u heterophilous_main.py --dataset 'Roman-empire' --train_lr 0.01 --seed 0 --num_layers 32 --mlp_layers 1 --hidden_dim 512 --train_iter 1000 --test_iter 1 --use_saved_model False --dropout 0.20 --train_w_decay 0.0 --num_splits 10 --batch_size None  --centrality 'degree' --device 'cuda:0' | tee output/roman_empire_32_none.txt
# python -u heterophilous_main.py --dataset 'Roman-empire' --train_lr 0.01 --seed 0 --num_layers 32 --mlp_layers 1 --hidden_dim 512 --train_iter 1000 --test_iter 1 --use_saved_model False --dropout 0.20 --train_w_decay 0.0 --num_splits 10 --batch_size 64  --centrality 'degree' --device 'cuda:0' | tee output/roman_empire_32.txt

# python -u heterophilous_main.py --dataset 'Roman-empire' --train_lr 0.01 --seed 0 --num_layers 64 --mlp_layers 1 --hidden_dim 512 --train_iter 1000 --test_iter 1 --use_saved_model False --dropout 0.20 --train_w_decay 0.0 --num_splits 10 --batch_size None  --centrality 'degree' --device 'cuda:0' | tee output/roman_empire_64_none.txt
# python -u heterophilous_main.py --dataset 'Roman-empire' --train_lr 0.01 --seed 0 --num_layers 64 --mlp_layers 1 --hidden_dim 512 --train_iter 1000 --test_iter 1 --use_saved_model False --dropout 0.20 --train_w_decay 0.0 --num_splits 10 --batch_size 32  --centrality 'degree' --device 'cuda:0' | tee output/roman_empire_64.txt



# python -u heterophilous_main.py --dataset 'Amazon-ratings' --train_lr 0.01 --seed 0 --num_layers 2 --mlp_layers 1 --hidden_dim 512 --train_iter 1000 --test_iter 1 --use_saved_model False --dropout 0.20 --train_w_decay 0.0 --num_splits 10 --batch_size None  --centrality 'degree' --device 'cuda:0' | tee output/amazon_ratings_2.txt

# python -u heterophilous_main.py --dataset 'Amazon-ratings' --train_lr 0.01 --seed 0 --num_layers 4 --mlp_layers 1 --hidden_dim 512 --train_iter 1000 --test_iter 1 --use_saved_model False --dropout 0.20 --train_w_decay 0.0 --num_splits 10 --batch_size None  --centrality 'degree' --device 'cuda:0' | tee output/amazon_ratings_4_none.txt
# python -u heterophilous_main.py --dataset 'Amazon-ratings' --train_lr 0.01 --seed 0 --num_layers 4 --mlp_layers 1 --hidden_dim 512 --train_iter 1000 --test_iter 1 --use_saved_model False --dropout 0.20 --train_w_decay 0.0 --num_splits 10 --batch_size 4096  --centrality 'degree' --device 'cuda:0' | tee output/amazon_ratings_4.txt

# python -u heterophilous_main.py --dataset 'Amazon-ratings' --train_lr 0.01 --seed 0 --num_layers 8 --mlp_layers 1 --hidden_dim 512 --train_iter 1000 --test_iter 1 --use_saved_model False --dropout 0.20 --train_w_decay 0.0 --num_splits 10 --batch_size None  --centrality 'degree' --device 'cuda:0' | tee output/amazon_ratings_8_none.txt
# python -u heterophilous_main.py --dataset 'Amazon-ratings' --train_lr 0.01 --seed 0 --num_layers 8 --mlp_layers 1 --hidden_dim 512 --train_iter 1000 --test_iter 1 --use_saved_model False --dropout 0.20 --train_w_decay 0.0 --num_splits 10 --batch_size 4096  --centrality 'degree' --device 'cuda:0' | tee output/amazon_ratings_8.txt

# python -u heterophilous_main.py --dataset 'Amazon-ratings' --train_lr 0.01 --seed 0 --num_layers 16 --mlp_layers 1 --hidden_dim 512 --train_iter 1000 --test_iter 1 --use_saved_model False --dropout 0.20 --train_w_decay 0.0 --num_splits 10 --batch_size None  --centrality 'degree' --device 'cuda:0' | tee output/amazon_ratings_16_none.txt
# python -u heterophilous_main.py --dataset 'Amazon-ratings' --train_lr 0.01 --seed 0 --num_layers 16 --mlp_layers 1 --hidden_dim 512 --train_iter 1000 --test_iter 1 --use_saved_model False --dropout 0.20 --train_w_decay 0.0 --num_splits 10 --batch_size 32  --centrality 'degree' --device 'cuda:0' | tee output/amazon_ratings_16.txt

# python -u heterophilous_main.py --dataset 'Amazon-ratings' --train_lr 0.01 --seed 0 --num_layers 32 --mlp_layers 1 --hidden_dim 512 --train_iter 1000 --test_iter 1 --use_saved_model False --dropout 0.20 --train_w_decay 0.0 --num_splits 10 --batch_size None  --centrality 'degree' --device 'cuda:0' | tee output/amazon_ratings_32_none.txt
# python -u heterophilous_main.py --dataset 'Amazon-ratings' --train_lr 0.01 --seed 0 --num_layers 32 --mlp_layers 1 --hidden_dim 512 --train_iter 1000 --test_iter 1 --use_saved_model False --dropout 0.20 --train_w_decay 0.0 --num_splits 10 --batch_size 32  --centrality 'degree' --device 'cuda:0' | tee output/amazon_ratings_32.txt

# python -u heterophilous_main.py --dataset 'Amazon-ratings' --train_lr 0.01 --seed 0 --num_layers 64 --mlp_layers 1 --hidden_dim 512 --train_iter 1000 --test_iter 1 --use_saved_model False --dropout 0.20 --train_w_decay 0.0 --num_splits 10 --batch_size None  --centrality 'degree' --device 'cuda:0' | tee output/amazon_ratings_64_none.txt
# python -u heterophilous_main.py --dataset 'Amazon-ratings' --train_lr 0.01 --seed 0 --num_layers 64 --mlp_layers 1 --hidden_dim 512 --train_iter 1000 --test_iter 1 --use_saved_model False --dropout 0.20 --train_w_decay 0.0 --num_splits 10 --batch_size 32  --centrality 'degree' --device 'cuda:0' | tee output/amazon_ratings_64.txt




# python -u heterophilous_main.py --dataset 'Minesweeper' --train_lr 0.01 --seed 0 --num_layers 2 --mlp_layers 1 --hidden_dim 512 --train_iter 1000 --test_iter 1 --use_saved_model False --dropout 0.20 --train_w_decay 0.0 --num_splits 10 --batch_size None --centrality 'degree' --device 'cuda:0' | tee output/minesweeper_2.txt

# python -u heterophilous_main.py --dataset 'Minesweeper' --train_lr 0.01 --seed 0 --num_layers 4 --mlp_layers 1 --hidden_dim 512 --train_iter 1000 --test_iter 1 --use_saved_model False --dropout 0.20 --train_w_decay 0.0 --num_splits 10 --batch_size None --centrality 'degree' --device 'cuda:0' | tee output/minesweeper_4_none.txt
# python -u heterophilous_main.py --dataset 'Minesweeper' --train_lr 0.01 --seed 0 --num_layers 4 --mlp_layers 1 --hidden_dim 512 --train_iter 1000 --test_iter 1 --use_saved_model False --dropout 0.20 --train_w_decay 0.0 --num_splits 10 --batch_size 8192 --centrality 'degree' --device 'cuda:0' | tee output/minesweeper_4.txt

# python -u heterophilous_main.py --dataset 'Minesweeper' --train_lr 0.01 --seed 0 --num_layers 8 --mlp_layers 1 --hidden_dim 512 --train_iter 1000 --test_iter 1 --use_saved_model False --dropout 0.20 --train_w_decay 0.0 --num_splits 10 --batch_size None --centrality 'degree' --device 'cuda:0' | tee output/minesweeper_8_none.txt
# python -u heterophilous_main.py --dataset 'Minesweeper' --train_lr 0.01 --seed 0 --num_layers 8 --mlp_layers 1 --hidden_dim 512 --train_iter 1000 --test_iter 1 --use_saved_model False --dropout 0.20 --train_w_decay 0.0 --num_splits 10 --batch_size 4096 --centrality 'degree' --device 'cuda:0' | tee output/minesweeper_8.txt

# python -u heterophilous_main.py --dataset 'Minesweeper' --train_lr 0.01 --seed 0 --num_layers 16 --mlp_layers 1 --hidden_dim 512 --train_iter 1000 --test_iter 1 --use_saved_model False --dropout 0.20 --train_w_decay 0.0 --num_splits 10 --batch_size None --centrality 'degree' --device 'cuda:0' | tee output/minesweeper_16_none.txt
# python -u heterophilous_main.py --dataset 'Minesweeper' --train_lr 0.01 --seed 0 --num_layers 16 --mlp_layers 1 --hidden_dim 512 --train_iter 1000 --test_iter 1 --use_saved_model False --dropout 0.20 --train_w_decay 0.0 --num_splits 10 --batch_size 1024 --centrality 'degree' --device 'cuda:0' | tee output/minesweeper_16.txt

# python -u heterophilous_main.py --dataset 'Minesweeper' --train_lr 0.01 --seed 0 --num_layers 32 --mlp_layers 1 --hidden_dim 512 --train_iter 1000 --test_iter 1 --use_saved_model False --dropout 0.20 --train_w_decay 0.0 --num_splits 10 --batch_size None --centrality 'degree' --device 'cuda:0' | tee output/minesweeper_32_none.txt
# python -u heterophilous_main.py --dataset 'Minesweeper' --train_lr 0.01 --seed 0 --num_layers 32 --mlp_layers 1 --hidden_dim 512 --train_iter 1000 --test_iter 1 --use_saved_model False --dropout 0.20 --train_w_decay 0.0 --num_splits 10 --batch_size 512 --centrality 'degree' --device 'cuda:0' | tee output/minesweeper_32.txt

# python -u heterophilous_main.py --dataset 'Minesweeper' --train_lr 0.01 --seed 0 --num_layers 64 --mlp_layers 1 --hidden_dim 512 --train_iter 1000 --test_iter 1 --use_saved_model False --dropout 0.20 --train_w_decay 0.0 --num_splits 10 --batch_size None --centrality 'degree' --device 'cuda:0' | tee output/minesweeper_64_none.txt
# python -u heterophilous_main.py --dataset 'Minesweeper' --train_lr 0.01 --seed 0 --num_layers 64 --mlp_layers 1 --hidden_dim 512 --train_iter 1000 --test_iter 1 --use_saved_model False --dropout 0.20 --train_w_decay 0.0 --num_splits 10 --batch_size 256 --centrality 'degree' --device 'cuda:0' | tee output/minesweeper_64.txt




# python -u heterophilous_main.py --dataset 'Tolokers' --train_lr 0.01 --seed 0 --num_layers 2 --mlp_layers 1 --hidden_dim 512 --train_iter 1000 --test_iter 1 --use_saved_model False --dropout 0.20 --train_w_decay 0.0 --num_splits 10 --batch_size None --centrality 'degree' --device 'cuda:0' | tee output/tolokers_2.txt

# python -u heterophilous_main.py --dataset 'Tolokers' --train_lr 0.01 --seed 0 --num_layers 4 --mlp_layers 1 --hidden_dim 512 --train_iter 1000 --test_iter 1 --use_saved_model False --dropout 0.20 --train_w_decay 0.0 --num_splits 10 --batch_size None  --centrality 'degree' --device 'cuda:0' | tee output/tolokers_4_none.txt
# python -u heterophilous_main.py --dataset 'Tolokers' --train_lr 0.01 --seed 0 --num_layers 4 --mlp_layers 1 --hidden_dim 512 --train_iter 1000 --test_iter 1 --use_saved_model False --dropout 0.20 --train_w_decay 0.0 --num_splits 10 --batch_size 4096  --centrality 'degree' --device 'cuda:0' | tee output/tolokers_4.txt

# python -u heterophilous_main.py --dataset 'Tolokers' --train_lr 0.01 --seed 0 --num_layers 8 --mlp_layers 1 --hidden_dim 512 --train_iter 1000 --test_iter 1 --use_saved_model False --dropout 0.20 --train_w_decay 0.0 --num_splits 10 --batch_size None --centrality 'degree' --device 'cuda:0' | tee output/tolokers_8_none.txt
# python -u heterophilous_main.py --dataset 'Tolokers' --train_lr 0.01 --seed 0 --num_layers 8 --mlp_layers 1 --hidden_dim 512 --train_iter 1000 --test_iter 1 --use_saved_model False --dropout 0.20 --train_w_decay 0.0 --num_splits 10 --batch_size 1024 --centrality 'degree' --device 'cuda:0' | tee output/tolokers_8.txt

# python -u heterophilous_main.py --dataset 'Tolokers' --train_lr 0.01 --seed 0 --num_layers 16 --mlp_layers 1 --hidden_dim 512 --train_iter 1000 --test_iter 1 --use_saved_model False --dropout 0.20 --train_w_decay 0.0 --num_splits 10 --batch_size None  --centrality 'degree' --device 'cuda:0' | tee output/tolokers_16_none.txt
# python -u heterophilous_main.py --dataset 'Tolokers' --train_lr 0.01 --seed 0 --num_layers 16 --mlp_layers 1 --hidden_dim 512 --train_iter 1000 --test_iter 1 --use_saved_model False --dropout 0.20 --train_w_decay 0.0 --num_splits 10 --batch_size 64  --centrality 'degree' --device 'cuda:0' | tee output/tolokers_16.txt

# python -u heterophilous_main.py --dataset 'Tolokers' --train_lr 0.01 --seed 0 --num_layers 32 --mlp_layers 1 --hidden_dim 512 --train_iter 1000 --test_iter 1 --use_saved_model False --dropout 0.20 --train_w_decay 0.0 --num_splits 10 --batch_size None  --centrality 'degree' --device 'cuda:0' | tee output/tolokers_32_none.txt
# python -u heterophilous_main.py --dataset 'Tolokers' --train_lr 0.01 --seed 0 --num_layers 32 --mlp_layers 1 --hidden_dim 512 --train_iter 1000 --test_iter 1 --use_saved_model False --dropout 0.20 --train_w_decay 0.0 --num_splits 10 --batch_size 32  --centrality 'degree' --device 'cuda:0' | tee output/tolokers_32.txt

# python -u heterophilous_main.py --dataset 'Tolokers' --train_lr 0.01 --seed 0 --num_layers 64 --mlp_layers 1 --hidden_dim 512 --train_iter 1000 --test_iter 1 --use_saved_model False --dropout 0.20 --train_w_decay 0.0 --num_splits 10 --batch_size None  --centrality 'degree' --device 'cuda:0' | tee output/tolokers_64_none.txt
# python -u heterophilous_main.py --dataset 'Tolokers' --train_lr 0.01 --seed 0 --num_layers 64 --mlp_layers 1 --hidden_dim 512 --train_iter 1000 --test_iter 1 --use_saved_model False --dropout 0.20 --train_w_decay 0.0 --num_splits 10 --batch_size 32  --centrality 'degree' --device 'cuda:0' | tee output/tolokers_64.txt




# python -u heterophilous_main.py --dataset 'Questions' --train_lr 0.01 --seed 0 --num_layers 2 --mlp_layers 1 --hidden_dim 512 --train_iter 1000 --test_iter 1 --use_saved_model False --dropout 0.20 --train_w_decay 0.0 --num_splits 10 --batch_size None --centrality 'degree' --device 'cuda:0' | tee output/questions_2.txt

# python -u heterophilous_main.py --dataset 'Questions' --train_lr 0.01 --seed 0 --num_layers 4 --mlp_layers 1 --hidden_dim 512 --train_iter 1000 --test_iter 1 --use_saved_model False --dropout 0.20 --train_w_decay 0.0 --num_splits 10 --batch_size None --centrality 'degree' --device 'cuda:0' | tee output/questions_4_none.txt
# python -u heterophilous_main.py --dataset 'Questions' --train_lr 0.01 --seed 0 --num_layers 4 --mlp_layers 1 --hidden_dim 512 --train_iter 1000 --test_iter 1 --use_saved_model False --dropout 0.20 --train_w_decay 0.0 --num_splits 10 --batch_size 4096 --centrality 'degree' --device 'cuda:0' | tee output/questions_4.txt

# python -u heterophilous_main.py --dataset 'Questions' --train_lr 0.01 --seed 0 --num_layers 8 --mlp_layers 1 --hidden_dim 512 --train_iter 1000 --test_iter 1 --use_saved_model False --dropout 0.20 --train_w_decay 0.0 --num_splits 10 --batch_size None --centrality 'degree' --device 'cuda:0' | tee output/questions_8_none.txt
# python -u heterophilous_main.py --dataset 'Questions' --train_lr 0.01 --seed 0 --num_layers 8 --mlp_layers 1 --hidden_dim 512 --train_iter 1000 --test_iter 1 --use_saved_model False --dropout 0.20 --train_w_decay 0.0 --num_splits 10 --batch_size 1024 --centrality 'degree' --device 'cuda:0' | tee output/questions_8.txt

# python -u heterophilous_main.py --dataset 'Questions' --train_lr 0.01 --seed 0 --num_layers 16 --mlp_layers 1 --hidden_dim 512 --train_iter 1000 --test_iter 1 --use_saved_model False --dropout 0.20 --train_w_decay 0.0 --num_splits 10 --batch_size None --centrality 'degree' --device 'cuda:0' | tee output/questions_16_none.txt
# python -u heterophilous_main.py --dataset 'Questions' --train_lr 0.01 --seed 0 --num_layers 16 --mlp_layers 1 --hidden_dim 512 --train_iter 1000 --test_iter 1 --use_saved_model False --dropout 0.20 --train_w_decay 0.0 --num_splits 10 --batch_size 128 --centrality 'degree' --device 'cuda:0' | tee output/questions_16.txt

# python -u heterophilous_main.py --dataset 'Questions' --train_lr 0.01 --seed 0 --num_layers 32 --mlp_layers 1 --hidden_dim 512 --train_iter 1000 --test_iter 1 --use_saved_model False --dropout 0.20 --train_w_decay 0.0 --num_splits 10 --batch_size None --centrality 'degree' --device 'cuda:0' | tee output/questions_32_none.txt
# python -u heterophilous_main.py --dataset 'Questions' --train_lr 0.01 --seed 0 --num_layers 32 --mlp_layers 1 --hidden_dim 512 --train_iterthe 1000 --test_iter 1 --use_saved_model False --dropout 0.20 --train_w_decay 0.0 --num_splits 10 --batch_size 32 --centrality 'degree' --device 'cuda:0' | tee output/questions_32.txt

# python -u heterophilous_main.py --dataset 'Questions' --train_lr 0.01 --seed 0 --num_layers 64 --mlp_layers 1 --hidden_dim 512 --train_iter 1000 --test_iter 1 --use_saved_model False --dropout 0.20 --train_w_decay 0.0 --num_splits 10 --batch_size None --centrality 'degree' --device 'cuda:0' | tee output/questions_64_none.txt
# python -u heterophilous_main.py --dataset 'Questions' --train_lr 0.01 --seed 0 --num_layers 64 --mlp_layers 1 --hidden_dim 512 --train_iter 1000 --test_iter 1 --use_saved_model False --dropout 0.20 --train_w_decay 0.0 --num_splits 10 --batch_size 8 --centrality 'degree' --device 'cuda:0' | tee output/questions_64.txt




# ------------------------


# python -u tu_datasets_main.py --dataset 'MUTAG' --train_lr 0.001 --seed 0 --num_layers 4 --mlp_layers 1 --hidden_dim 64 --train_iter 50 --test_iter 1 --use_saved_model False --dropout 0.0 --train_w_decay 0.0 --num_splits 25 --batch_size 8 --batch_frac 0.0 --model gin --centrality 'degree' --device 'cuda:0' | tee output/mutag_gin.txt

# python -u tu_datasets_main.py --dataset 'MUTAG' --train_lr 0.001 --seed 0 --num_layers 10 --mlp_layers 1 --hidden_dim 64 --train_iter 50 --test_iter 1 --use_saved_model False --dropout 0.0 --train_w_decay 0.0 --num_splits 25 --batch_size 8 --batch_frac 1.0 --model gin --centrality 'degree' --device 'cuda:0' | tee output/mutag_gin_1.0.txt

# python -u tu_datasets_main.py --dataset 'MUTAG' --train_lr 0.001 --seed 0 --num_layers 10 --mlp_layers 1 --hidden_dim 64 --train_iter 50 --test_iter 1 --use_saved_model False --dropout 0.0 --train_w_decay 0.0 --num_splits 25 --batch_size 8 --batch_frac 0.6 --model gin --centrality 'degree' --device 'cuda:0' | tee output/mutag_gin_0.6.txt

# python -u tu_datasets_main.py --dataset 'MUTAG' --train_lr 0.001 --seed 0 --num_layers 10 --mlp_layers 1 --hidden_dim 64 --train_iter 50 --test_iter 1 --use_saved_model False --dropout 0.0 --train_w_decay 0.0 --num_splits 25 --batch_size 8 --batch_frac 0.4 --model gin --centrality 'degree' --device 'cuda:0' | tee output/mutag_gin_0.4.txt

# python -u tu_datasets_main.py --dataset 'MUTAG' --train_lr 0.001 --seed 0 --num_layers 10 --mlp_layers 1 --hidden_dim 64 --train_iter 50 --test_iter 1 --use_saved_model False --dropout 0.0 --train_w_decay 0.0 --num_splits 25 --batch_size 8 --batch_frac 0.8 --model gin --centrality 'degree' --device 'cuda:0' | tee output/mutag_gin_0.8.txt



# python -u tu_datasets_main.py --dataset 'PROTEINS' --train_lr 0.001 --seed 0 --num_layers 4 --mlp_layers 1 --hidden_dim 64 --train_iter 50 --test_iter 1 --use_saved_model False --dropout 0.0 --train_w_decay 0.0001 --num_splits 25 --batch_size 64 --batch_frac 0.0 --model gin --centrality 'degree' --device 'cuda:0' | tee output/proteins_gin.txt

# python -u tu_datasets_main.py --dataset 'PROTEINS' --train_lr 0.001 --seed 0 --num_layers 10 --mlp_layers 1 --hidden_dim 64 --train_iter 50 --test_iter 1 --use_saved_model False --dropout 0.0 --train_w_decay 0.0001 --num_splits 25 --batch_size 64 --batch_frac 1.0 --model gin --centrality 'degree' --device 'cuda:0' | tee output/proteins_gin_1.0.txt

# python -u tu_datasets_main.py --dataset 'PROTEINS' --train_lr 0.001 --seed 0 --num_layers 10 --mlp_layers 1 --hidden_dim 64 --train_iter 50 --test_iter 1 --use_saved_model False --dropout 0.0 --train_w_decay 0.0001 --num_splits 25 --batch_size 64 --batch_frac 0.6 --model gin --centrality 'degree' --device 'cuda:0' | tee output/proteins_gin_0.6.txt

# python -u tu_datasets_main.py --dataset 'PROTEINS' --train_lr 0.001 --seed 0 --num_layers 10 --mlp_layers 1 --hidden_dim 64 --train_iter 50 --test_iter 1 --use_saved_model False --dropout 0.0 --train_w_decay 0.0001 --num_splits 25 --batch_size 64 --batch_frac 0.4 --model gin --centrality 'degree' --device 'cuda:0' | tee output/proteins_gin_0.4.txt

# python -u tu_datasets_main.py --dataset 'PROTEINS' --train_lr 0.001 --seed 0 --num_layers 10 --mlp_layers 1 --hidden_dim 64 --train_iter 50 --test_iter 1 --use_saved_model False --dropout 0.0 --train_w_decay 0.0001 --num_splits 25 --batch_size 64 --batch_frac 0.8 --model gin --centrality 'degree' --device 'cuda:0' | tee output/proteins_gin_0.8.txt




# python -u tu_datasets_main.py --dataset 'COLLAB' --train_lr 0.001 --seed 0 --num_layers 10 --mlp_layers 1 --hidden_dim 64 --train_iter 20 --test_iter 1 --use_saved_model False --dropout 0.0 --train_w_decay 0.0005 --num_splits 25 --batch_size 8 --batch_frac 0.6  --model gin --centrality 'degree' --device 'cuda:0' | tee output/collab_1.txt

# python -u tu_datasets_main.py --dataset 'COLLAB' --train_lr 0.001 --seed 0 --num_layers 10 --mlp_layers 1 --hidden_dim 64 --train_iter 20 --test_iter 1 --use_saved_model False --dropout 0.0 --train_w_decay 0.0005 --num_splits 25 --batch_size 8 --batch_frac 0.4  --model gin --centrality 'degree' --device 'cuda:0' | tee output/collab_2.txt

# python -u tu_datasets_main.py --dataset 'COLLAB' --train_lr 0.001 --seed 0 --num_layers 10 --mlp_layers 1 --hidden_dim 64 --train_iter 20 --test_iter 1 --use_saved_model False --dropout 0.0 --train_w_decay 0.0005 --num_splits 25 --batch_size 8 --batch_frac 0.6  --model gin --centrality 'degree' --device 'cuda:0' | tee output/collab_3.txt




# python -u tu_datasets_main.py --dataset 'IMDB-BINARY' --train_lr 0.001 --seed 0 --num_layers 4 --mlp_layers 1 --hidden_dim 64 --train_iter 50 --test_iter 1 --use_saved_model False --dropout 0.0 --train_w_decay 0.00001 --num_splits 25 --batch_size 64 --batch_frac 0.0 --model gcn --centrality 'degree' --device 'cuda:0' | tee output/imdb-binary_gcn.txt

# python -u tu_datasets_main.py --dataset 'IMDB-BINARY' --train_lr 0.001 --seed 0 --num_layers 4 --mlp_layers 1 --hidden_dim 64 --train_iter 50 --test_iter 1 --use_saved_model False --dropout 0.0 --train_w_decay 0.00001 --num_splits 25 --batch_size 64 --batch_frac 0.0 --model gin --centrality 'degree' --device 'cuda:0' | tee output/imdb-binary_gin.txt


# python -u tu_datasets_main.py --dataset 'IMDB-BINARY' --train_lr 0.001 --seed 0 --num_layers 10 --mlp_layers 1 --hidden_dim 64 --train_iter 50 --test_iter 1 --use_saved_model False --dropout 0.0 --train_w_decay 0.00001 --num_splits 25 --batch_size 64 --batch_frac 1.0 --model gcn --centrality 'degree' --device 'cuda:0' | tee output/imdb-binary_gcn_1.txt

# python -u tu_datasets_main.py --dataset 'IMDB-BINARY' --train_lr 0.001 --seed 0 --num_layers 10 --mlp_layers 1 --hidden_dim 64 --train_iter 50 --test_iter 1 --use_saved_model False --dropout 0.0 --train_w_decay 0.00001 --num_splits 25 --batch_size 64 --batch_frac 0.6 --model gcn --centrality 'degree' --device 'cuda:0' | tee output/imdb-binary_gcn_0.6.txt

# python -u tu_datasets_main.py --dataset 'IMDB-BINARY' --train_lr 0.001 --seed 0 --num_layers 10 --mlp_layers 1 --hidden_dim 64 --train_iter 50 --test_iter 1 --use_saved_model False --dropout 0.0 --train_w_decay 0.00001 --num_splits 25 --batch_size 64 --batch_frac 1.0 --model gin --centrality 'degree' --device 'cuda:0' | tee output/imdb-binary_gin_1.txt

# python -u tu_datasets_main.py --dataset 'IMDB-BINARY' --train_lr 0.001 --seed 0 --num_layers 10 --mlp_layers 1 --hidden_dim 64 --train_iter 50 --test_iter 1 --use_saved_model False --dropout 0.0 --train_w_decay 0.00001 --num_splits 25 --batch_size 64 --batch_frac 0.6 --model gin --centrality 'degree' --device 'cuda:0' | tee output/imdb-binary_gin_0.6.txt

# python -u tu_datasets_main.py --dataset 'IMDB-BINARY' --train_lr 0.001 --seed 0 --num_layers 10 --mlp_layers 1 --hidden_dim 64 --train_iter 50 --test_iter 1 --use_saved_model False --dropout 0.0 --train_w_decay 0.00001 --num_splits 25 --batch_size 64 --batch_frac 0.4 --model gin --centrality 'degree' --device 'cuda:0' | tee output/imdb-binary_gin_0.4.txt

# python -u tu_datasets_main.py --dataset 'IMDB-BINARY' --train_lr 0.001 --seed 0 --num_layers 10 --mlp_layers 1 --hidden_dim 64 --train_iter 50 --test_iter 1 --use_saved_model False --dropout 0.0 --train_w_decay 0.00001 --num_splits 25 --batch_size 64 --batch_frac 0.8 --model gin --centrality 'degree' --device 'cuda:0' | tee output/imdb-binary_gin_0.8.txt



# python -u tu_datasets_main.py --dataset 'ENZYMES' --train_lr 0.001 --seed 0 --num_layers 10 --mlp_layers 1 --hidden_dim 64 --train_iter 50 --test_iter 1 --use_saved_model False --dropout 0.0 --train_w_decay 0.0005 --num_splits 25 --batch_size 32 --batch_frac 1.0 --model gin --centrality 'degree' --device 'cuda:0' | tee output/enzymes_1.txt


# python -u tu_datasets_main.py --dataset 'ENZYMES' --train_lr 0.001 --seed 0 --num_layers 10 --mlp_layers 1 --hidden_dim 64 --train_iter 50 --test_iter 1 --use_saved_model False --dropout 0.0 --train_w_decay 0.0005 --num_splits 25 --batch_size 32 --batch_frac 0.6 --model gin --centrality 'degree' --device 'cuda:0' | tee output/enzymes_0".6".txt

# python -u tu_datasets_main.py --dataset 'ENZYMES' --train_lr 0.001 --seed 0 --num_layers 10 --mlp_layers 1 --hidden_dim 64 --train_iter 50 --test_iter 1 --use_saved_model False --dropout 0.0 --train_w_decay 0.0005 --num_splits 25 --batch_size 32 --batch_frac 0.4 --model gin --centrality 'degree' --device 'cuda:0' | tee output/enzymes_"0.4".txt

# python -u tu_datasets_main.py --dataset 'ENZYMES' --train_lr 0.001 --seed 0 --num_layers 10 --mlp_layers 1 --hidden_dim 64 --train_iter 50 --test_iter 1 --use_saved_model False --dropout 0.0 --train_w_decay 0.0005 --num_splits 25 --batch_size 32 --batch_frac 0.8 --model gin --centrality 'degree' --device 'cuda:0' | tee output/enzymes_"0.8".txt




# python -u tu_datasets_main.py --dataset 'REDDIT-BINARY' --train_lr 0.001 --seed 0 --num_layers 10 --mlp_layers 1 --hidden_dim 64 --train_iter 50 --test_iter 1 --use_saved_model False --dropout 0.0 --train_w_decay 0.00001 --num_splits 25 --batch_size 64 --batch_frac 1.0 --model gin --centrality 'degree' --device 'cuda:0' | tee output/reddit-binary_1.txt

# python -u tu_datasets_main.py --dataset 'REDDIT-BINARY' --train_lr 0.001 --seed 0 --num_layers 10 --mlp_layers 1 --hidden_dim 64 --train_iter 50 --test_iter 1 --use_saved_model False --dropout 0.0 --train_w_decay 0.00001 --num_splits 25 --batch_size 64 --batch_frac 0.6 --model gin --centrality 'degree' --device 'cuda:0' | tee output/reddit-binary_"0.6".txt

# python -u tu_datasets_main.py --dataset 'REDDIT-BINARY' --train_lr 0.001 --seed 0 --num_layers 10 --mlp_layers 1 --hidden_dim 64 --train_iter 50 --test_iter 1 --use_saved_model False --dropout 0.0 --train_w_decay 0.00001 --num_splits 25 --batch_size 64 --batch_frac 0.4 --model gin --centrality 'degree' --device 'cuda:0' | tee output/reddit-binary_"0.4".txt

# python -u tu_datasets_main.py --dataset 'REDDIT-BINARY' --train_lr 0.001 --seed 0 --num_layers 10 --mlp_layers 1 --hidden_dim 64 --train_iter 50 --test_iter 1 --use_saved_model False --dropout 0.0 --train_w_decay 0.00001 --num_splits 25 --batch_size 64 --batch_frac 0.8 --model gin --centrality 'degree' --device 'cuda:0' | tee output/reddit-binary_"0.8".txt



# python -u tu_datasets_main.py --dataset 'REDDIT-BINARY' --train_lr 0.001 --seed 0 --num_layers 4 --mlp_layers 1 --hidden_dim 64 --train_iter 50 --test_iter 1 --use_saved_model False --dropout 0.0 --train_w_decay 0.00001 --num_splits 25 --batch_size 64 --batch_frac 0.0 --model gcn --centrality 'degree' --device 'cuda:0' | tee output/reddit-binary_gcn.txt

# python -u tu_datasets_main.py --dataset 'REDDIT-BINARY' --train_lr 0.001 --seed 0 --num_layers 4 --mlp_layers 1 --hidden_dim 64 --train_iter 50 --test_iter 1 --use_saved_model False --dropout 0.0 --train_w_decay 0.00001 --num_splits 25 --batch_size 64 --batch_frac 0.0 --model gin --centrality 'degree' --device 'cuda:0' | tee output/reddit-binary_gin.txt

# python -u tu_datasets_main.py --dataset 'REDDIT-BINARY' --train_lr 0.001 --seed 0 --num_layers 10 --mlp_layers 1 --hidden_dim 64 --train_iter 50 --test_iter 1 --use_saved_model False --dropout 0.0 --train_w_decay 0.00001 --num_splits 25 --batch_size 64 --batch_frac 1.0 --model gcn --centrality 'degree' --device 'cuda:0' | tee output/reddit-binary_gcn_1.txt

# python -u tu_datasets_main.py --dataset 'REDDIT-BINARY' --train_lr 0.001 --seed 0 --num_layers 10 --mlp_layers 1 --hidden_dim 64 --train_iter 50 --test_iter 1 --use_saved_model False --dropout 0.0 --train_w_decay 0.00001 --num_splits 25 --batch_size 64 --batch_frac 0.6 --model gcn --centrality 'pagerank' --device 'cuda:0' | tee output/reddit-binary_gcn_"0.6".txt

# python -u tu_datasets_main.py --dataset 'REDDIT-BINARY' --train_lr 0.001 --seed 0 --num_layers 10 --mlp_layers 1 --hidden_dim 64 --train_iter 50 --test_iter 1 --use_saved_model False --dropout 0.0 --train_w_decay 0.00001 --num_splits 25 --batch_size 64 --batch_frac 1.0 --model gin --centrality 'pagerank' --device 'cuda:0' | tee output/reddit-binary_gin_1.txt

# python -u tu_datasets_main.py --dataset 'REDDIT-BINARY' --train_lr 0.001 --seed 0 --num_layers 10 --mlp_layers 1 --hidden_dim 64 --train_iter 50 --test_iter 1 --use_saved_model False --dropout 0.0 --train_w_decay 0.00001 --num_splits 25 --batch_size 64 --batch_frac 0.6 --model gin --centrality 'pagerank' --device 'cuda:0' | tee output/reddit-binary_gin_"0.6".txt

# python -u tu_datasets_main.py --dataset 'REDDIT-BINARY' --train_lr 0.001 --seed 0 --num_layers 10 --mlp_layers 1 --hidden_dim 64 --train_iter 50 --test_iter 1 --use_saved_model False --dropout 0.0 --train_w_decay 0.00001 --num_splits 25 --batch_size 64 --batch_frac 0.4 --model gcn --centrality 'degree' --device 'cuda:0' | tee output/reddit-binary_gcn_"0.4".txt

# python -u tu_datasets_main.py --dataset 'REDDIT-BINARY' --train_lr 0.001 --seed 0 --num_layers 10 --mlp_layers 1 --hidden_dim 64 --train_iter 50 --test_iter 1 --use_saved_model False --dropout 0.0 --train_w_decay 0.00001 --num_splits 25 --batch_size 64 --batch_frac 0.8 --model gcn --centrality 'degree' --device 'cuda:0' | tee output/reddit-binary_gcn_"0.8".txt


# python -u tu_datasets_main.py --dataset 'COLLAB' --train_lr 0.001 --seed 0 --num_layers 12 --mlp_layers 1 --hidden_dim 64 --train_iter 50 --test_iter 1 --use_saved_model False --dropout 0.50 --train_w_decay 1e-5 --num_splits 25 --batch_size 64 --batch_frac 1.0 --model gcn --centrality 'closeness' --device 'cuda:0' | tee output/COLLAB_GCN_12_layers.txt

# python -u tu_datasets_main.py --dataset 'COLLAB' --train_lr 0.001 --seed 0 --num_layers 14 --mlp_layers 1 --hidden_dim 64 --train_iter 50 --test_iter 1 --use_saved_model False --dropout 0.50 --train_w_decay 1e-5 --num_splits 25 --batch_size 64 --batch_frac 1.0 --model gcn --centrality 'closeness' --device 'cuda:0' | tee output/COLLAB_GCN_14_layers.txt

# python -u tu_datasets_main.py --dataset 'COLLAB' --train_lr 0.001 --seed 0 --num_layers 16 --mlp_layers 1 --hidden_dim 64 --train_iter 50 --test_iter 1 --use_saved_model False --dropout 0.50 --train_w_decay 1e-5 --num_splits 25 --batch_size 64 --batch_frac 1.0 --model gcn --centrality 'closeness' --device 'cuda:0' | tee output/COLLAB_GCN_16_layers.txt


# python -u tu_datasets_main.py --dataset 'COLLAB' --train_lr 0.001 --seed 0 --num_layers 12 --mlp_layers 1 --hidden_dim 64 --train_iter 50 --test_iter 1 --use_saved_model False --dropout 0.50 --train_w_decay 1e-5 --num_splits 25 --batch_size 64 --batch_frac 1.0 --model gin --centrality 'betweenness' --device 'cuda:0' | tee output/COLLAB_GIN_12_layers.txt

# python -u tu_datasets_main.py --dataset 'COLLAB' --train_lr 0.001 --seed 0 --num_layers 14 --mlp_layers 1 --hidden_dim 64 --train_iter 50 --test_iter 1 --use_saved_model False --dropout 0.50 --train_w_decay 1e-5 --num_splits 25 --batch_size 64 --batch_frac 1.0 --model gin --centrality 'betweenness' --device 'cuda:0' | tee output/COLLAB_GIN_14_layers.txt

# python -u tu_datasets_main.py --dataset 'COLLAB' --train_lr 0.001 --seed 0 --num_layers 16 --mlp_layers 1 --hidden_dim 64 --train_iter 50 --test_iter 1 --use_saved_model False --dropout 0.50 --train_w_decay 1e-5 --num_splits 25 --batch_size 64 --batch_frac 1.0 --model gin --centrality 'betweenness' --device 'cuda:0' | tee output/COLLAB_GIN_16_layers.txt



# python -u tu_datasets_main.py --dataset 'IMDB-BINARY' --train_lr 0.001 --seed 0 --num_layers 12 --mlp_layers 1 --hidden_dim 64 --train_iter 50 --test_iter 1 --use_saved_model False --dropout 0.50 --train_w_decay 1e-5 --num_splits 25 --batch_size 64 --batch_frac 1.0 --model gcn --centrality 'pagerank' --device 'cuda:0' | tee output/IMDB-BINARY_GCN_12_layers.txt

# python -u tu_datasets_main.py --dataset 'IMDB-BINARY' --train_lr 0.001 --seed 0 --num_layers 14 --mlp_layers 1 --hidden_dim 64 --train_iter 50 --test_iter 1 --use_saved_model False --dropout 0.50 --train_w_decay 1e-5 --num_splits 25 --batch_size 64 --batch_frac 1.0 --model gcn --centrality 'pagerank' --device 'cuda:0' | tee output/IMDB-BINARY_GCN_14_layers.txt

# python -u tu_datasets_main.py --dataset 'IMDB-BINARY' --train_lr 0.001 --seed 0 --num_layers 14 --mlp_layers 1 --hidden_dim 64 --train_iter 50 --test_iter 1 --use_saved_model False --dropout 0.50 --train_w_decay 1e-5 --num_splits 25 --batch_size 64 --batch_frac 1.0 --model gcn --centrality 'pagerank' --device 'cuda:0' | tee output/IMDB-BINARY_GCN_16_layers.txt

# python -u tu_datasets_main.py --dataset 'IMDB-BINARY' --train_lr 0.001 --seed 0 --num_layers 6 --mlp_layers 1 --hidden_dim 64 --train_iter 50 --test_iter 1 --use_saved_model False --dropout 0.50 --train_w_decay 1e-5 --num_splits 25 --batch_size 64 --batch_frac 1.0 --model gcn --centrality 'pagerank' --device 'cuda:0' | tee output/IMDB-BINARY_GCN_6_layers.txt

# python -u tu_datasets_main.py --dataset 'IMDB-BINARY' --train_lr 0.001 --seed 0 --num_layers 7 --mlp_layers 1 --hidden_dim 64 --train_iter 50 --test_iter 1 --use_saved_model False --dropout 0.50 --train_w_decay 1e-5 --num_splits 25 --batch_size 64 --batch_frac 1.0 --model gcn --centrality 'pagerank' --device 'cuda:0' | tee output/IMDB-BINARY_GCN_7_layers.txt

# python -u tu_datasets_main.py --dataset 'IMDB-BINARY' --train_lr 0.001 --seed 0 --num_layers 8 --mlp_layers 1 --hidden_dim 64 --train_iter 50 --test_iter 1 --use_saved_model False --dropout 0.50 --train_w_decay 1e-5 --num_splits 25 --batch_size 64 --batch_frac 1.0 --model gcn --centrality 'pagerank' --device 'cuda:0' | tee output/IMDB-BINARY_GCN_8_layers.txt



# python -u tu_datasets_main.py --dataset 'IMDB-BINARY' --train_lr 0.001 --seed 0 --num_layers 10 --mlp_layers 1 --hidden_dim 64 --train_iter 50 --test_iter 1 --use_saved_model False --dropout 0.50 --train_w_decay 1e-5 --num_splits 25 --batch_size 64 --batch_frac 1.0 --model gin --centrality 'pagerank' --device 'cuda:0' | tee output/IMDB-BINARY_GIN_10_layers.txt

# python -u tu_datasets_main.py --dataset 'IMDB-BINARY' --train_lr 0.001 --seed 0 --num_layers 12 --mlp_layers 1 --hidden_dim 64 --train_iter 50 --test_iter 1 --use_saved_model False --dropout 0.50 --train_w_decay 1e-5 --num_splits 25 --batch_size 64 --batch_frac 1.0 --model gin --centrality 'pagerank' --device 'cuda:0' | tee output/IMDB-BINARY_GIN_12_layers.txt

# python -u tu_datasets_main.py --dataset 'IMDB-BINARY' --train_lr 0.001 --seed 0 --num_layers 14 --mlp_layers 1 --hidden_dim 64 --train_iter 50 --test_iter 1 --use_saved_model False --dropout 0.50 --train_w_decay 1e-5 --num_splits 25 --batch_size 64 --batch_frac 1.0 --model gin --centrality 'pagerank' --device 'cuda:0' | tee output/IMDB-BINARY_GIN_14_layers.txt

# python -u tu_datasets_main.py --dataset 'IMDB-BINARY' --train_lr 0.001 --seed 0 --num_layers 16 --mlp_layers 1 --hidden_dim 64 --train_iter 50 --test_iter 1 --use_saved_model False --dropout 0.50 --train_w_decay 1e-5 --num_splits 25 --batch_size 64 --batch_frac 1.0 --model gin --centrality 'pagerank' --device 'cuda:0' | tee output/IMDB-BINARY_GIN_16_layers.txt

# python -u tu_datasets_main.py --dataset 'IMDB-BINARY' --train_lr 0.001 --seed 0 --num_layers 8 --mlp_layers 1 --hidden_dim 64 --train_iter 50 --test_iter 1 --use_saved_model False --dropout 0.50 --train_w_decay 1e-5 --num_splits 25 --batch_size 64 --batch_frac 1.0 --model gin --centrality 'pagerank' --device 'cuda:0' | tee output/IMDB-BINARY_GIN_8_layers.txt



# python -u tu_datasets_main.py --dataset 'PROTEINS' --train_lr 0.001 --seed 0 --num_layers 10 --mlp_layers 1 --hidden_dim 64 --train_iter 200 --test_iter 1 --use_saved_model False --dropout 0.50 --train_w_decay 1e-5 --num_splits 25 --batch_size 64 --batch_frac 1.0 --model gcn --centrality 'closeness' --device 'cuda:0' | tee output/PROTEINS_GCN_10_layers.txt

# python -u tu_datasets_main.py --dataset 'PROTEINS' --train_lr 0.001 --seed 0 --num_layers 12 --mlp_layers 1 --hidden_dim 64 --train_iter 200 --test_iter 1 --use_saved_model False --dropout 0.50 --train_w_decay 1e-5 --num_splits 25 --batch_size 64 --batch_frac 1.0 --model gcn --centrality 'closeness' --device 'cuda:0' | tee output/PROTEINS_GCN_12_layers.txt

# python -u tu_datasets_main.py --dataset 'PROTEINS' --train_lr 0.001 --seed 0 --num_layers 14 --mlp_layers 1 --hidden_dim 64 --train_iter 200 --test_iter 1 --use_saved_model False --dropout 0.50 --train_w_decay 1e-5 --num_splits 25 --batch_size 64 --batch_frac 1.0 --model gcn --centrality 'closeness' --device 'cuda:0' | tee output/PROTEINS_GCN_14_layers.txt

# python -u tu_datasets_main.py --dataset 'PROTEINS' --train_lr 0.001 --seed 0 --num_layers 16 --mlp_layers 1 --hidden_dim 64 --train_iter 200 --test_iter 1 --use_saved_model False --dropout 0.50 --train_w_decay 1e-5 --num_splits 25 --batch_size 64 --batch_frac 1.0 --model gcn --centrality 'degree' --device 'cuda:0' | tee output/PROTEINS_GCN_16_layers.txt


python -u tu_datasets_main.py --dataset 'PROTEINS' --train_lr 0.001 --seed 0 --num_layers 10 --mlp_layers 1 --hidden_dim 64 --train_iter 200 --test_iter 1 --use_saved_model False --dropout 0.50 --train_w_decay 1e-5 --num_splits 25 --batch_size 64 --batch_frac 1.0 --model gin --centrality 'closeness' --device 'cuda:0' | tee output/PROTEINS_GIN_10_layers.txt

python -u tu_datasets_main.py --dataset 'PROTEINS' --train_lr 0.001 --seed 0 --num_layers 12 --mlp_layers 1 --hidden_dim 64 --train_iter 200 --test_iter 1 --use_saved_model False --dropout 0.50 --train_w_decay 1e-5 --num_splits 25 --batch_size 64 --batch_frac 1.0 --model gin --centrality 'closeness' --device 'cuda:0' | tee output/PROTEINS_GIN_12_layers.txt

python -u tu_datasets_main.py --dataset 'PROTEINS' --train_lr 0.001 --seed 0 --num_layers 14 --mlp_layers 1 --hidden_dim 64 --train_iter 200 --test_iter 1 --use_saved_model False --dropout 0.50 --train_w_decay 1e-5 --num_splits 25 --batch_size 64 --batch_frac 1.0 --model gin --centrality 'closeness' --device 'cuda:0' | tee output/PROTEINS_GIN_14_layers.txt

python -u tu_datasets_main.py --dataset 'PROTEINS' --train_lr 0.001 --seed 0 --num_layers 16 --mlp_layers 1 --hidden_dim 64 --train_iter 200 --test_iter 1 --use_saved_model False --dropout 0.50 --train_w_decay 1e-5 --num_splits 25 --batch_size 64 --batch_frac 1.0 --model gin --centrality 'closeness' --device 'cuda:0' | tee output/PROTEINS_GIN_16_layers.txt



# python -u tu_datasets_main.py --dataset 'MUTAG' --train_lr 0.001 --seed 0 --num_layers 16 --mlp_layers 1 --hidden_dim 64 --train_iter 200 --test_iter 1 --use_saved_model False --dropout 0.50 --train_w_decay 1e-5 --num_splits 25 --batch_size 64 --batch_frac 1.0 --model gcn --centrality 'degree' --device 'cuda:0' | tee output/MUTAG_GCN_16-layers.txt

# python -u tu_datasets_main.py --dataset 'MUTAG' --train_lr 0.001 --seed 0 --num_layers 12 --mlp_layers 1 --hidden_dim 64 --train_iter 200 --test_iter 1 --use_saved_model False --dropout 0.50 --train_w_decay 1e-5 --num_splits 25 --batch_size 64 --batch_frac 1.0 --model gin --centrality 'degree' --device 'cuda:0' | tee output/MUTAG_GIN_12-layers.txt

# python -u tu_datasets_main.py --dataset 'MUTAG' --train_lr 0.001 --seed 0 --num_layers 14 --mlp_layers 1 --hidden_dim 64 --train_iter 200 --test_iter 1 --use_saved_model False --dropout 0.50 --train_w_decay 1e-5 --num_splits 25 --batch_size 64 --batch_frac 1.0 --model gin --centrality 'degree' --device 'cuda:0' | tee output/MUTAG_GIN_14-layers.txt

# python -u tu_datasets_main.py --dataset 'MUTAG' --train_lr 0.001 --seed 0 --num_layers 16 --mlp_layers 1 --hidden_dim 64 --train_iter 200 --test_iter 1 --use_saved_model False --dropout 0.50 --train_w_decay 1e-5 --num_splits 25 --batch_size 64 --batch_frac 1.0 --model gin --centrality 'degree' --device 'cuda:0' | tee output/MUTAG_GIN_16-layers.txt

