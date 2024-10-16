# Behavior Cloning (BC)
for seed in 0 1 2 3
do
    python train_dt.py --seed $seed --context_length 30 --epochs 5 --model_type 'naive' --num_steps 500000 --num_buffers 50 --game 'Breakout' --batch_size 128
done

for seed in 0 1 2 3
do
    python train_dt.py --seed $seed --context_length 30 --epochs 5 --model_type 'naive' --num_steps 500000 --num_buffers 50 --game 'Qbert' --batch_size 128
done

for seed in 0 1 2 3
do
    python train_dt.py --seed $seed --context_length 50 --epochs 5 --model_type 'naive' --num_steps 500000 --num_buffers 50 --game 'Pong' --batch_size 512
done

for seed in 0 1 2 3
do
    python train_dt.py --seed $seed --context_length 30 --epochs 5 --model_type 'naive' --num_steps 500000 --num_buffers 50 --game 'Seaquest' --batch_size 128
done