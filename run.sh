#python main_supcon.py --batch_size 200 --model resnet18 --dataset cifar100 --temp 0.5 --epochs 10000    #19768

#python main_supcon.py --batch_size 256 --model resnet18 --dataset cifar10 --temp 0.1 --epochs 500        #12525

#python main_supcon.py --batch_size 265 --num_classes 10 --model resnet18  --dataset cifar100 --epochs 400 --learning_rate 0.001 --lr_decay_epochs "200, 300" --lr_decay_rate 0.5 --temp 0.05   #12525

#python main_supcon_incremental.py --learning_rate 0.001 --alfa 0.0  --save_freq 1

#python main_supcon_incremental.py --learning_rate 0.001 --temp 0.01 --alfa 0.2  --save_freq 100 --batch_size 512  --memory_size 50 --num_init_classes 10 --num_classes 20 --epochs 500 --lr_decay_epochs "200, 350"

#python main_supcon.py --temp 0.01 --model resnet18 --num_classes 10 --dataset cifar100 --save_freq 100 --batch_size 512 --epochs 500 --lr_decay_epochs "200, 300" --lr_decay_rate 0.5

#python3 main_supcon.py --temp 0.05 --dataset cifar10 --model resnet18 --num_classes 4 --save_freq 20 --batch_size 128 --epochs 500 --lr_decay_epochs "100" --learning_rate 0.001 --print_freq 20

#python3 main_supcon.py --temp 0.05 --dataset mnist --model mlp --num_classes 2 --save_freq 10 --batch_size 256 --epochs 100 --lr_decay_epochs "500" --learning_rate 0.001 --print_freq 5

#python3 main_supcon_incremental.py --dataset mnist --learning_rate 0.001 --temp 0.05 --alfa 0.2  --save_freq 20 --batch_size 256  --fixed_memory 200 --num_init_classes 2 --num_classes 4 --epochs 200 --print_freq 5 --lr_decay_epochs "600" 

#python3 main_supcon_incremental.py --dataset mnist --learning_rate 0.001 --temp 0.05 --alfa 0.2  --save_freq 20 --batch_size 256  --fixed_memory 200 --num_init_classes 4 --num_classes 6 --epochs 200 --print_freq 5 --lr_decay_epochs "600" 

#python3 main_supcon_incremental.py --dataset mnist --learning_rate 0.001 --temp 0.05 --alfa 0.2  --save_freq 20 --batch_size 256  --fixed_memory 200 --num_init_classes 6 --num_classes 8 --epochs 200 --print_freq 5 --lr_decay_epochs "600" 

#python3 main_supcon_incremental.py --dataset mnist --learning_rate 0.001 --temp 0.05 --alfa 0.2  --save_freq 20 --batch_size 256  --fixed_memory 200 --num_init_classes 8 --num_classes 10 --epochs 200 --print_freq 5 --lr_decay_epochs "600" 



python3 main_supcon_incremental.py --dataset mnist --learning_rate 0.001 --temp 0.05 --alfa 0.2  --save_freq 20 --batch_size 256  --fixed_memory 500 --num_init_classes 2 --num_classes 4 --epochs 200 --print_freq 5 --lr_decay_epochs "600" 

python3 main_supcon_incremental.py --dataset mnist --learning_rate 0.001 --temp 0.05 --alfa 0.2  --save_freq 20 --batch_size 256  --fixed_memory 500 --num_init_classes 4 --num_classes 6 --epochs 200 --print_freq 5 --lr_decay_epochs "600" 

python3 main_supcon_incremental.py --dataset mnist --learning_rate 0.001 --temp 0.05 --alfa 0.2  --save_freq 20 --batch_size 256  --fixed_memory 500 --num_init_classes 6 --num_classes 8 --epochs 200 --print_freq 5 --lr_decay_epochs "600" 

python3 main_supcon_incremental.py --dataset mnist --learning_rate 0.001 --temp 0.05 --alfa 0.2  --save_freq 20 --batch_size 256  --fixed_memory 500 --num_init_classes 8 --num_classes 10 --epochs 200 --print_freq 5 --lr_decay_epochs "600" 
