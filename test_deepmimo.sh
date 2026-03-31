python main.py --conf=./conf/test.yaml \
--gpu=3 \
--mod \
input_ant_size=4 \
input_car_size=4 \
scheme=acoupler_flat_v2 \
acoupler_flat_v2.d_model=128 \
acoupler_flat_v2.num_heads=1 \
acoupler_flat_v2.dropout=0 \
exp_dir=./exp/my_train/input_4antx4car/data2000_Ncur0or1/ACouplerFlatV2-head1-d128_LR-d1024/test \
checkpoint_path=./exp/my_train/input_4antx4car/data2000_Ncur0or1/ACouplerFlatV2-head1-d128_LR-d1024/ckpts/model.pth \
test_tasks.trace_smooth.past_len_list=[0,1,4,8,16]

#activate_tasks=['trace_smooth'] \

#scheme=conv_coupler \
#conv_coupler.d_model=128 \
#conv_coupler.depth_pred=8 \

#scheme=acoupler \
#acoupler.d_model=128 \
#acoupler.num_heads=1 \
#acoupler.dropout=0 \
#acoupler.heads_aggregate=False \

#scheme=acoupler_flat_v2 \
#acoupler_flat_v2.d_model=128 \
#acoupler_flat_v2.num_heads=1 \
#acoupler_flat_v2.dropout=0 \

#scheme=afcoupler \
#afcoupler.d_model=128 \

#scheme=acdnet \
#acdnet.hidden_size=512 \

#scheme=rcdnet \
#rcdnet.hidden_size=512 \

#scheme=afcdnet \

#scheme=convlstm \
#convlstm.hidden_dim=16 \
#convlstm.depth_pred=2 \


