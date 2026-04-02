python main.py --conf=./conf/test.yaml \
--gpu=0 \
--mod \
input_ant_size=3 \
input_car_size=3 \
scheme=afcoupler \
exp_dir=./exp/DeepMIMO/input_3antx3car/data2000/AFCoupler/test \
checkpoint_path=./exp/DeepMIMO/input_3antx3car/data2000/AFCoupler/model.pth \

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


