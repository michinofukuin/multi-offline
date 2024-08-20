import os
import torch
import numpy as np

from envs.ma_mujoco.multiagent_mujoco.mujoco_multi import MujocoMulti
from envs.env_wrappers import ShareDummyVecEnv
from utils.logger import setup_logger_kwargs, Logger
from utils.util import evaluate
from datasets.offline_dataset import ReplayBuffer
from algos.OMIGA import OMIGA
from algos.mine import mine
from algos.new import new
from algos.new_2 import new_2
from algos.new_3 import new_3
from utils.vae_class import VAE
import wandb
from tqdm import tqdm

def make_train_env(config):
    def get_env_fn(rank):
        def init_env():
            if config['env_name'] == "mujoco":
                env_args = {"scenario": config['scenario'],
                            "agent_conf": config['agent_conf'],
                            "agent_obsk": config['agent_obsk'],
                            "episode_limit": 1000}
                env = MujocoMulti(env_args=env_args)
            else:
                print("Can not support the " + config['env_name'] + "environment.")
                raise NotImplementedError
            env.seed(config['seed'])
            return env

        return init_env
    return ShareDummyVecEnv([get_env_fn(0)])


def make_eval_env(config):
    def get_env_fn(rank):
        def init_env():
            if config['env_name'] == "mujoco":
                env_args = {"scenario": config['scenario'],
                            "agent_conf": config['agent_conf'],
                            "agent_obsk": config['agent_obsk'],
                            "episode_limit": 1000}
                env = MujocoMulti(env_args=env_args)
            else:
                print("Can not support the " + config['env_name'] + "environment.")
                raise NotImplementedError
            env.seed(config['seed'])
            return env

        return init_env
    return ShareDummyVecEnv([get_env_fn(0)])

def make_eval_env_2(config,seed=0):
    def get_env_fn(rank):
        def init_env():
            if config['env_name'] == "mujoco":
                env_args = {"scenario": config['scenario'],
                            "agent_conf": config['agent_conf'],
                            "agent_obsk": config['agent_obsk'],
                            "episode_limit": 1000}
                env = MujocoMulti(env_args=env_args)
            else:
                print("Can not support the " + config['env_name'] + "environment.")
                raise NotImplementedError
            env.seed(seed)
            return env

        return init_env
    return ShareDummyVecEnv([get_env_fn(0)])

def run(config):
    assert config['env_name'] == 'mujoco', "Invalid environment"
    env_name = config['scenario'] + '-' + config['agent_conf'] + '-' + config['data_type']
    exp_name = config['algo']
    name = config['algo'] + '-' + config['scenario'] + '-' + config['agent_conf'] + '-' + config['data_type'] + '-' + 'test_s' + str(config['seed'])
    logger_dic = config['algo']
    if config['wandb'] == True:
        wandb.init(project=exp_name, name=name, group=env_name)

    # Seeding
    np.random.seed(config['seed'])
    torch.manual_seed(config['seed']) 

    env = make_train_env(config)
    eval_env = make_eval_env(config)

    state_dim = env.observation_space[0].shape[0]
    action_dim = env.action_space[0].shape[0]
    n_agents = len(env.observation_space)
    print('state_dim:', state_dim, 'action_dim:', action_dim, 'num_agents:', n_agents)

    logger_kwargs = setup_logger_kwargs(logger_dic, config['seed'])
    logger = Logger(**logger_kwargs)
    logger.save_config(config)
    
    # Datasets
    offline_dataset = ReplayBuffer(state_dim, action_dim, n_agents, env_name, config['data_dir'], device=config['device'])
    offline_dataset.load()

    result_logs = {}

    def _eval_and_log(train_result, config):
        train_result = {k: v.detach().cpu().numpy() for k, v in train_result.items()}
        print('\n==========Policy testing==========')
        # evaluation via real-env rollout
        tot_r = []
        num_seed = [0, 1, 2, 3, 4]
        for s in num_seed:
            ev = make_eval_env_2(config, seed=s)
            ep_r = evaluate(agent, ev, config['env_name'])
            tot_r.append(ep_r)
        ep_r_mean = np.mean(tot_r)
        ep_r_std = np.std(tot_r)
        train_result.update({'ep_r_mean': ep_r_mean})
        train_result.update({'ep_r_std': ep_r_std})
        result_log = {'log': train_result, 'step': iteration}
        result_logs[str(iteration)] = result_log

        for k, v in sorted(train_result.items()):
            print(f'- {k:23s}:{v:15.10f}')
        print(f'iteration={iteration}')
        print('\n==========Policy training==========', flush=True)

        return train_result
  
    # Agent
    # agent = OMIGA(state_dim, action_dim, n_agents, eval_env, config)
    # agent = mine(state_dim, action_dim, n_agents, eval_env, config)
    agent = new_2(state_dim, action_dim, n_agents, eval_env, config)
    max_behavior = int(2e5)
    
    # print('\n=============Train behaviour=============')
    # for iteration in tqdm(range(0, max_behavior), ncols=70, desc='train behaviour', initial=1, total=max_behavior, ascii=True, disable=os.environ.get("DISABLE_TQDM", False)):
    #     o, s, a, r, mask, s_next, o_next, a_next = offline_dataset.sample(int(256))
    #     res = agent.train_behaviour(o, a, r, o_next, mask)
    #     if iteration % 10000 == 0:
    #         print(res)

    # directory = "/mnt/data/chenhaosheng/OMIGA-master/behaviour/HalfCheetah-expert"
    # directory = "/mnt/data/chenhaosheng/OMIGA-master/behaviour/HalfCheetah-medium"
    # directory = "/mnt/data/chenhaosheng/OMIGA-master/behaviour/HalfCheetah-medium-replay"
    # directory = "/mnt/data/chenhaosheng/OMIGA-master/behaviour/HalfCheetah-medium-expert"
    # directory = "/mnt/data/chenhaosheng/OMIGA-master/behaviour/Ant-expert"
    directory = "/mnt/data/chenhaosheng/OMIGA-master/behaviour/Ant-medium"
    agent.load_behaviour(directory)
    # if not os.path.exists(directory):
    #     os.makedirs(directory)
    # agent.save_behaviour(directory)
    
    # Train
    print('\n==========Start training==========')

    for iteration in tqdm(range(0, config['total_iterations']), ncols=70, desc=config['algo'], initial=1, total=config['total_iterations'], ascii=True, disable=os.environ.get("DISABLE_TQDM", False)):
        o, s, a, r, mask, s_next, o_next, a_next = offline_dataset.sample(config['batch_size'])
        train_result = agent.train_step(o, s, a, r, mask, s_next, o_next, a_next)
        # if iteration % config['save_iterations'] == 0:
        #     directory = "/mnt/data/chenhaosheng/OMIGA-master/ant_expert_514/{}".format(iteration)
        #     if not os.path.exists(directory):
        #         os.makedirs(directory)
        #     agent.save_models(directory)
        if iteration % config['log_iterations'] == 0:
            train_result = _eval_and_log(train_result, config)
            if config['wandb'] == True:
                wandb.log(train_result)

    # Save results
    logger.save_result_logs(result_logs)

    env.close()
    eval_env.close()

if __name__ == "__main__":
    from configs.config import get_parser
    args = get_parser().parse_args() 
    run(vars(args))