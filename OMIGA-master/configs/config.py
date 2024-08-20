import argparse

def boolean(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--env_name', type=str, default='mujoco')
    parser.add_argument('--scenario', type=str, default='Ant-v2', help="Which mujoco task to run on")
    parser.add_argument('--agent_conf', type=str, default='2x4')
    parser.add_argument('--agent_obsk', type=int, default=1)
    parser.add_argument('--mix',default=True,type=boolean)
    parser.add_argument('--data_type', type=str, default='expert')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--algo', default='ant-medium-4e-4', type=str)
    parser.add_argument('--device', default='cuda:0', type=str)
    parser.add_argument('--wandb', default=False, type=boolean)
    parser.add_argument('--data_dir', default='/mnt/data/chenhaosheng/OMIGA-master/data/', type=str)
    parser.add_argument('--clr',default=1e-5, type=float)
    parser.add_argument('--total_iterations', default=int(1e6), type=int)
    parser.add_argument('--log_iterations', default=int(10000), type=int)
    parser.add_argument('--save_iterations', default=int(200000), type=int)
    parser.add_argument('--gamma', default=0.99, type=float)
    parser.add_argument('--tau', default=0.005, type=float)
    parser.add_argument('--hidden_sizes', default=256)
    parser.add_argument('--mix_hidden_sizes', default=64)
    parser.add_argument('--vae_hidden_sizes', default=64)  
    parser.add_argument('--adap_total_alpha_tau', default=0.9, type=float) 
    parser.add_argument('--batch_size', default=128, type=int)
    parser.add_argument('--lr', default=4e-4, type=float)
    parser.add_argument('--kl_lr', default=4e-5, type=float)
    parser.add_argument('--grad_norm_clip', default=1.0, type=float)
    parser.add_argument('--quality', default=0.5, type=float)
    parser.add_argument('--h_tot', default=0.05, type=float)
    parser.add_argument('--quantile', default=0.7, type=float)

    return parser
