import torch
import numpy as np
from algos.mine import mine
from envs.ma_mujoco.multiagent_mujoco.mujoco_multi import MujocoMulti
from envs.env_wrappers import ShareDummyVecEnv
from utils.vae_class import VAE
from configs.config import get_parser
args = get_parser().parse_args() 
env_name='Ant-v2'
def evaluate(agent, env, environment, seed, num_evaluation=10, max_steps=None):
    episode_rewards = []
    if max_steps is None and environment == "mujoco":
        max_steps = 1000
    assert max_steps != None

    for eval_iter in range(num_evaluation):
        obs, s, _ = env.reset()
        episode_reward = 0
        for t in range(max_steps):

            actions = agent.step((np.array(obs)).astype(np.float32))
            action = actions.numpy()
            
            next_obs, next_s, reward, done, info, _ = env.step(action)
            episode_reward += reward[0,0,0]

            if done[0,0]:
                break
            obs = next_obs
        episode_rewards.append(episode_reward)
        
    return np.mean(episode_rewards)

def make_eval_env(seed,env_name):
    def get_env_fn(rank):
        def init_env():
            env_args = {"scenario": env_name,
                        "agent_conf": '2x4',
                        "agent_obsk": 1,
                        "episode_limit": 1000}
            env = MujocoMulti(env_args=env_args)
            env.seed(seed)
            return env
        return init_env
    return ShareDummyVecEnv([get_env_fn(0)])

if __name__ == "__main__":
    env = make_eval_env(0, env_name)
    state_dim = env.observation_space[0].shape[0]
    action_dim = env.action_space[0].shape[0]
    n_agents = len(env.observation_space)
    max_action = env.action_space[0].high[0]
    vae = VAE(state_dim + n_agents, action_dim, action_dim*2, max_action, hidden_dim=750).to(device='cuda:3')
    vae.load_state_dict(torch.load('/mnt/data/chenhaosheng/CFCQL/continuous/results/2/Ant-v2/expert_2024-04-17_09-45-51-392729/model.pt'))
    vae.eval()
    num_seeds = [0,1,2,3,4]
    rewards = []
    for seed in (num_seeds):
        np.random.seed(seed)
        torch.manual_seed(seed) 
        env = make_eval_env(seed, env_name)
        agent = mine(state_dim, action_dim, n_agents, env, vars(args))
        agent.vae = vae
        agent.load_models('/mnt/data/chenhaosheng/OMIGA-master/ant_expert_3/60000')  # Load your trained model
        avg_reward = evaluate(agent, env, 'mujoco', seed)
        rewards.append(avg_reward)
        print(f"Seed {seed}: Average Reward = {avg_reward}")

    mean_reward = np.mean(rewards)
    st = np.std(rewards)
    print(f"Average Reward : {mean_reward} ± {st}")
