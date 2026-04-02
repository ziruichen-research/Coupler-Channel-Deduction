python main.py --conf=./conf/train.yaml \
--gpu=0 \
--mod \
input_ant_size=3 \
input_car_size=3 \
num_cur_range_train=[0,1] \
max_samples=2000 \
batch_size=100 \
micro_batch=100 \
learning_rate=0.001 \
lr_scheduler.type=TransformerWarmupScheduler \
lr_scheduler.update_mode=step \
lr_scheduler_list.TransformerWarmupScheduler.d_model=512 \
lr_scheduler_list.TransformerWarmupScheduler.warmup_steps=4000 \
scheme=afcoupler \
exp_dir=./exp/DeepMIMO-tmp/input_3antx3car/data2000/AFCoupler

#scheme=conv_coupler \
#conv_coupler.d_model=128 \
#conv_coupler.depth_pred=8 \

#scheme=acoupler \
#acoupler.d_model=128 \
#acoupler.num_heads=1 \
#acoupler.heads_aggregate=False \
#acoupler.dropout=0 \

#scheme=afcoupler \
#afcoupler.d_model=128 \

#scheme=acdnet \
#acdnet.hidden_size=512 \
#batch_size=500 \
#micro_batch=500 \

#scheme=rcdnet \
#rcdnet.hidden_size=512 \
#batch_size=500 \
#micro_batch=500 \

#scheme=convlstm \
#convlstm.hidden_dim=16 \
#convlstm.depth_pred=2 \

# ConvLSTM 用
#learning_rate=0.001 \
#lr_scheduler.type=StepLR \
#lr_scheduler.update_mode=epoch \
#lr_scheduler_list.StepLR.step_size=10000 \
#lr_scheduler_list.StepLR.gamma=0.8 \

# 其他模型用
#lr_scheduler.type=TransformerWarmupScheduler \
#lr_scheduler.update_mode=step \
#lr_scheduler_list.TransformerWarmupScheduler.d_model=128 \
#lr_scheduler_list.TransformerWarmupScheduler.warmup_steps=4000 \
