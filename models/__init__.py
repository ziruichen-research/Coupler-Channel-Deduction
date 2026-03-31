from .convLSTM import ConvLSTMCoupler
from .acdnet import ACDNet
from .rcdnet import RCDNet
from .afcdnet import AFCDNet
from .coupler import Coupler
from .conv_coupler import Conv1dCoupler
from .acoupler import ACoupler
from .acoupler_v2 import ACouplerV2
from .acoupler_flat import ACouplerFlat
from .acoupler_flat_v2 import ACouplerFlat_v2
from .acoupler_standard import ACouplerStandard
from .afcoupler import AFCoupler
from .lmmse import get_lmmse_model


def get_model(conf):
    name = conf['scheme']
    p = conf  # 根配置
    mp = conf['model']  # 模型特定配置

    if name == 'coupler':
        return Coupler(
            input_ant_size=p['input_ant_size'], input_car_size=p['input_car_size'],
            ant_size=p['num_ant'], car_size=p['num_car'],
            d_model=mp['d_model'], depth_pred=mp['depth_pred']
        )
    elif name == 'conv_coupler':
        return Conv1dCoupler(
            input_ant_size=p['input_ant_size'], input_car_size=p['input_car_size'],
            ant_size=p['num_ant'], car_size=p['num_car'],
            d_model=mp['d_model'], depth_pred=mp['depth_pred'], dropout=mp['dropout']
        )
    elif name == 'afcoupler':
        return AFCoupler(
            input_ant_size=p['input_ant_size'], input_car_size=p['input_car_size'],
            ant_size=p['num_ant'], car_size=p['num_car'],
            d_model=mp['d_model'], depth_pred=mp['depth_pred'],
            is_learnable=mp['is_learnable'], dropout=mp['dropout']
        )
    elif name == 'acoupler':
        return ACoupler(
            input_ant_size=p['input_ant_size'], input_car_size=p['input_car_size'],
            ant_size=p['num_ant'], car_size=p['num_car'],
            d_model=mp['d_model'], num_heads=mp['num_heads'], heads_aggregate=mp['heads_aggregate'],
            depth_pred=mp['depth_pred'], dropout=mp['dropout']
        )
    elif name == 'acouplerV2':
        return ACouplerV2(
            input_ant_size=p['input_ant_size'], input_car_size=p['input_car_size'],
            ant_size=p['num_ant'], car_size=p['num_car'],
            d_model=mp['d_model'], num_heads=mp['num_heads'], depth_pred=mp['depth_pred']
        )
    elif name == 'acoupler_flat':
        return ACouplerFlat(
            input_ant_size=p['input_ant_size'], input_car_size=p['input_car_size'],
            ant_size=p['num_ant'], car_size=p['num_car'],
            d_model=mp['d_model'], num_heads=mp['num_heads'], heads_aggregate=mp['heads_aggregate'],
            depth_pred=mp['depth_pred'], dropout=mp['dropout']
        )
    elif name == 'acoupler_flat_v2':
        return ACouplerFlat_v2(
            input_ant_size=p['input_ant_size'], input_car_size=p['input_car_size'],
            ant_size=p['num_ant'], car_size=p['num_car'],
            d_model=mp['d_model'], num_heads=mp['num_heads'],
            depth_pred=mp['depth_pred'], dropout=mp['dropout']
        )
    elif name == 'acoupler_standard':
        return ACouplerStandard(
            input_ant_size=p['input_ant_size'], input_car_size=p['input_car_size'],
            ant_size=p['num_ant'], car_size=p['num_car'],
            d_model=mp['d_model'], num_heads=mp['num_heads'], dim_feedforward=mp['dim_feedforward'],
            depth_pred=mp['depth_pred'], dropout=mp['dropout'], has_proj=mp['has_proj']
        )
    elif name == 'acdnet':
        return ACDNet(
            input_ant_size=p['input_ant_size'], input_car_size=p['input_car_size'],
            ant_size=p['num_ant'], car_size=p['num_car'],
            depth=mp['depth'], depth_pred=mp['depth_pred'], hidden_size=mp['hidden_size'], dropout=mp['dropout']
        )
    elif name == 'rcdnet':
        return RCDNet(
            input_ant_size=p['input_ant_size'], input_car_size=p['input_car_size'],
            ant_size=p['num_ant'], car_size=p['num_car'],
            depth=mp['depth'], depth_pred=mp['depth_pred'], hidden_size=mp['hidden_size'], dropout=mp['dropout']
        )
    elif name == 'afcdnet':
        return AFCDNet(
            input_ant_size=p['input_ant_size'], input_car_size=p['input_car_size'],
            ant_size=p['num_ant'], car_size=p['num_car'],
            depth=mp['depth'], depth_pred=mp['depth_pred'], hidden_size=mp['hidden_size']
        )
    elif name == 'convlstm':
        return ConvLSTMCoupler(
            input_ant_size=p['input_ant_size'], input_car_size=p['input_car_size'],
            ant_size=p['num_ant'], car_size=p['num_car'],
            depth_pred=mp['depth_pred'], hidden_dim=mp['hidden_dim'],
        )
    elif name == 'lmmse':
        return get_lmmse_model(conf)
    else:
        raise ValueError(f"Unknown scheme: {name}")