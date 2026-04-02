from .conv_coupler import Conv1dCoupler
from .acoupler import ACoupler
from .afcoupler import AFCoupler
from .acdnet import ACDNet
from .rcdnet import RCDNet
from .convLSTM import ConvLSTMCoupler
from .lmmse import get_lmmse_model


def get_model(conf):
    """Factory for all supported model schemes."""
    name = conf['scheme']
    p = conf
    mp = conf['model']

    if name == 'conv_coupler':
        return Conv1dCoupler(
            input_ant_size=p['input_ant_size'], input_car_size=p['input_car_size'],
            ant_size=p['num_ant'], car_size=p['num_car'],
            d_model=mp['d_model'], depth_pred=mp['depth_pred'],
            dropout=mp['dropout'] if 'dropout' in mp else 0.
        )
    elif name == 'afcoupler':
        return AFCoupler(
            input_ant_size=p['input_ant_size'], input_car_size=p['input_car_size'],
            ant_size=p['num_ant'], car_size=p['num_car'],
            d_model=mp['d_model'], depth_pred=mp['depth_pred'],
            is_learnable=mp['is_learnable'],
            dropout=mp['dropout'] if 'dropout' in mp else 0.
        )
    elif name == 'acoupler':
        return ACoupler(
            input_ant_size=p['input_ant_size'], input_car_size=p['input_car_size'],
            ant_size=p['num_ant'], car_size=p['num_car'],
            d_model=mp['d_model'], num_heads=mp['num_heads'], heads_aggregate=mp['heads_aggregate'],
            depth_pred=mp['depth_pred'],
            dropout=mp['dropout'] if 'dropout' in mp else 0.
        )
    elif name == 'acdnet':
        return ACDNet(
            input_ant_size=p['input_ant_size'], input_car_size=p['input_car_size'],
            ant_size=p['num_ant'], car_size=p['num_car'],
            depth=mp['depth'], depth_pred=mp['depth_pred'], hidden_size=mp['hidden_size'],
            dropout=mp['dropout'] if 'dropout' in mp else 0.
        )
    elif name == 'rcdnet':
        return RCDNet(
            input_ant_size=p['input_ant_size'], input_car_size=p['input_car_size'],
            ant_size=p['num_ant'], car_size=p['num_car'],
            depth=mp['depth'], depth_pred=mp['depth_pred'], hidden_size=mp['hidden_size'],
            dropout=mp['dropout'] if 'dropout' in mp else 0.
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
