训练：
python main.py --use_td3 --save_model  --exp_name td3 --seed 2
python main.py --use_epf --save_model  --exp_name epf --seed 2

python train_metadrive.py --use_epo --save_model  --exp_name 100w_epo_seed1500  --seed 1500 
python train_metadrive.py --use_qpsl --save_model --exp_name 100w_qpsl_seed1500 --seed 1500  --delta 0.02 

绘制训练曲线：
./plot.sh


针对metadrive的不同算法的曲线的参数：
rec: soft_plus delta=0.1
qpsl:delta=0.02
lag: init=1.0 lr=1e-4