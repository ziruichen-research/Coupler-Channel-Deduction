python main.py --conf=./conf/test.yaml \
--gpu=2  \
--mod \
input_ant_size=16 \
input_car_size=16 \
dataset_type=DICHASUS \
data_dirs=['./data/DICHASUS/h5_CD/0152-0157_DirectSample512to32'] \
const_h=1 \
dichasus.train_dataset_name=train_nearest32_seq36000.h5 \
dichasus.test_dataset_name=test_nearest32_seq36000.h5 \
dichasus.trace_src_dataset=./data/DICHASUS/src_h5/0152-0157_DirectSample512to32/data_0158.h5 \
dichasus.trace_part_name=dichasus-0158 \
activate_tasks=['history_length','trace_smooth'] \
test_tasks.trace_smooth.past_len_list=[0,1,4,16] \
scheme=convlstm \
convlstm.hidden_dim=16 \
convlstm.depth_pred=2 \
exp_dir=./exp/DICHASUS/0152-0157_DirectSample512to32/input_16antx16car/data36k-nei32/ConvLSTM-depth2-ch16/test \
checkpoint_path=./exp/DICHASUS/0152-0157_DirectSample512to32/input_16antx16car/data36k-nei32/ConvLSTM-depth2-ch16/ckpts/model.pth \

#dichasus.test_dataset_name=test_nearest64_seq36000.h5 \

#dichasus.trace_src_dataset=./data/DICHASUS/h5_CD/0152-0157_DirectSample512to32/test.h5 \
#dichasus.trace_part_name=dichasus-0152 \

#dichasus.trace_src_dataset=./data/DICHASUS/src_h5/0152-0157_DirectSample512to32/data_0158.h5 \
#dichasus.trace_part_name=dichasus-0158 \

#scheme=conv_coupler \
#conv_coupler.d_model=256 \
#conv_coupler.depth_pred=8 \
#conv_coupler.dropout=0.1 \

#scheme=acoupler \
#acoupler.num_heads=1 \
#acoupler.d_model=256 \
#acoupler.heads_aggregate=False \
#acoupler.dropout=0.1 \

#scheme=afcoupler \
#afcoupler.d_model=256 \
#afcoupler.dropout=0.1 \

#scheme=acdnet \
#acdnet.hidden_size=512 \
#acdnet.dropout=0.1 \

#scheme=rcdnet \
#rcdnet.hidden_size=512 \
#rcdnet.dropout=0.1 \

#scheme=convlstm \
#convlstm.hidden_dim=16 \
#convlstm.depth_pred=2 \