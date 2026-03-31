python main.py --conf=./conf/train.yaml \
--gpu=3 \
--mod \
input_ant_size=2 \
input_car_size=2 \
num_cur_range_train=[0,1] \
max_samples=2000 \
batch_size=500 \
micro_batch=500 \
learning_rate=0.001 \
lr_scheduler.type=TransformerWarmupScheduler \
lr_scheduler.update_mode=step \
lr_scheduler_list.TransformerWarmupScheduler.d_model=1024 \
lr_scheduler_list.TransformerWarmupScheduler.warmup_steps=4000 \
scheme=rcdnet \
rcdnet.hidden_size=1024 \
exp_dir=./exp/my_train/input_2antx2car/data2000_Ncur0or1/RCDNet-d1024_batch500


#scheme=coupler \

#scheme=conv_coupler \
#conv_coupler.d_model=128 \
#conv_coupler.depth_pred=8 \

#scheme=acoupler_flat \
#acoupler.d_model=128 \
#acoupler.num_heads=1 \
#acoupler.heads_aggregate=False \
#acoupler.dropout=0 \

#scheme=acoupler_flat_v2 \
#acoupler_flat_v2.d_model=128 \
#acoupler_flat_v2.num_heads=1 \
#acoupler_flat_v2.dropout=0 \

#scheme=acoupler_standard \
#acoupler_standard.d_model=2048 \
#acoupler_standard.num_heads=8 \
#acoupler_standard.dim_feedforward=512 \
#acoupler_standard.dropout=0 \
#acoupler_standard.has_proj=True \

#scheme=acoupler \
#acoupler.d_model=128 \
#acoupler.num_heads=1 \
#acoupler.heads_aggregate=False \
#acoupler.dropout=0 \

#scheme=acouplerV2 \
#acouplerV2.num_heads=4 \

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

#scheme=afcdnet \

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
