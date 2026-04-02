python main.py --conf=./conf/train.yaml \
--gpu=3 \
--mod \
dataset_type=DICHASUS \
data_dirs=['./data/DICHASUS/h5_CD/0152-0157_DirectSample512to32'] \
dichasus.train_dataset_name='train_nearest32_seq36000.h5' \
dichasus.test_dataset_name='test_nearest32_seq36000.h5' \
const_h=1 \
input_ant_size=16 \
input_car_size=16 \
num_cur_range_train=[0,0] \
max_samples=36000 \
batch_size=100 \
micro_batch=100 \
num_epochs=2000 \
save_epoch_inter=100 \
test_epoch_inter=20 \
loss_type=mse \
learning_rate=0.001 \
lr_scheduler.type=StepLR \
lr_scheduler.update_mode=epoch \
lr_scheduler_list.StepLR.step_size=10000 \
lr_scheduler_list.StepLR.gamma=0.8 \
scheme=convlstm \
convlstm.hidden_dim=16 \
convlstm.depth_pred=2 \
exp_dir=./exp/DICHASUS/0152-0157_DirectSample512to32/input_16antx16car/data36k-nei32/ConvLSTM-depth2-ch16


#num_epochs=2000 \
#save_epoch_inter=100 \

#lr_scheduler.type=TransformerWarmupScheduler \
#lr_scheduler.update_mode=step \
#lr_scheduler_list.TransformerWarmupScheduler.d_model=512 \
#lr_scheduler_list.TransformerWarmupScheduler.warmup_steps=4000 \

#learning_rate=0.001 \
#lr_scheduler.type=StepLR \
#lr_scheduler.update_mode=epoch \
#lr_scheduler_list.StepLR.step_size=10000 \
#lr_scheduler_list.StepLR.gamma=0.8 \

#scheme=conv_coupler \
#conv_coupler.d_model=256 \
#conv_coupler.depth_pred=8 \
#conv_coupler.dropout=0.1 \

#scheme=acoupler \
#acoupler.num_heads=1 \
#acoupler.d_model=256 \
#acoupler.heads_aggregate=False \
#acoupler.dropout=0.1 \
#exp_dir=./exp/DICHASUS/0152-0157_DirectSample512to32/input_8antx8car/ACoupler-d256_data36k-nei32

#scheme=afcoupler \
#afcoupler.d_model=256 \
#afcoupler.dropout=0.1 \
#exp_dir=./exp/DICHASUS/0152-0157_DirectSample512to32/input_8antx8car/AFCoupler-d256_data36k-nei32

#scheme=acdnet \
#acdnet.hidden_size=512 \
#acdnet.dropout=0.1 \

#scheme=rcdnet \
#rcdnet.hidden_size=512 \
#rcdnet.dropout=0.1 \

#scheme=convlstm \
#convlstm.hidden_dim=16 \
#convlstm.depth_pred=2 \
