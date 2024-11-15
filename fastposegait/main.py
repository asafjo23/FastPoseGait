import os
import argparse
from socket import timeout
import datetime
import torch
import torch.nn as nn
from modeling import models
from utils import config_loader, get_ddp_module, init_seeds, params_count, get_msg_mgr

# parser = argparse.ArgumentParser(description='Main program for fastposegait.')
# parser.add_argument('--local_rank', type=int, default=0,
#                     help="passed by torch.distributed.launch module")
# parser.add_argument('--cfgs', type=str,
#                     default='config/default.yaml', help="path of config file")
# parser.add_argument('--phase', default='train',
#                     choices=['train', 'test'], help="choose train or test phase")
# parser.add_argument('--log_to_file', action='store_true',
#                     help="log to file, default path is: output/<dataset>/<model>/<save_name>/<logs>/<Datetime>.txt")
# parser.add_argument('--iter', default=0, help="iter to restore")
# opt = parser.parse_args()


def initialization(cfgs, training):
    msg_mgr = get_msg_mgr()
    engine_cfg = cfgs['trainer_cfg'] if training else cfgs['evaluator_cfg']
    output_path = os.path.join('output/', cfgs['data_cfg']['dataset_name'],
                               cfgs['model_cfg']['model'], engine_cfg['save_name'])
    if training:
        msg_mgr.init_manager(output_path, False, engine_cfg['log_iter'],
                             engine_cfg['restore_hint'] if isinstance(engine_cfg['restore_hint'], (int)) else 0)
    else:
        msg_mgr.init_logger(output_path, False)

    msg_mgr.log_info(engine_cfg)
    random_data = os.urandom(4)
    seed = int.from_bytes(random_data, byteorder="big")

    init_seeds(seed, cuda_deterministic=False)


def run_model(cfgs, training):
    msg_mgr = get_msg_mgr()
    model_cfg = cfgs['model_cfg']
    msg_mgr.log_info(model_cfg)
    Model = getattr(models, model_cfg['model'])
    model = Model(cfgs, training)
    
    # if cfgs['trainer_cfg']['fix_BN']:
    #     model.fix_BN()
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    
    # Convert SyncBatchNorm to BatchNorm
    model = nn.SyncBatchNorm.convert_sync_batchnorm(model)
    
    msg_mgr.log_info(params_count(model))
    msg_mgr.log_info("Model Initialization Finished!")

    if training:
        Model.run_train(model)
    else:
        Model.run_test(model)


if __name__ == '__main__':
    import yaml
    def config_loader(file_path):
        with open(file_path, 'r', encoding='utf-8') as file:
            cfgs = yaml.safe_load(file)
        return cfgs

    cfgs = config_loader("/home/ajoseph/FastPoseGait/configs/test/test_CCPG.yaml")
    training = True
    initialization(cfgs, training)
    run_model(cfgs, training)
