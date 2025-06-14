import gymnasium
import numpy as np

import pufferlib
from pufferlib.ocean.poly_time import binding


class PolyTime(pufferlib.PufferEnv):
    def __init__(self, num_envs=1, render_mode=None, log_interval=10, nen_halt_penalty=1.0,
            correctness_reward=10.0, incorrectness_penalty=10.0, max_agents=50, max_items=50,
            tape_size=124, max_utility=50, buf=None, seed=0):

        if not isinstance(max_agents, int) or max_agents <= 0:
            raise ValueError("max_agents must be an integer greater than 0.")
        self.max_agents = max_agents

        if not isinstance(max_items, int) or max_items <= 0:
            raise ValueError("max_items must be an int > 0")
        self.max_items = max_items

        if not isinstance(tape_size , int) or tape_size  <= 0:
            raise ValueError("tape_size  must be an int > 0")
        self.tape_size  = tape_size 

        if not isinstance(max_utility, int) or max_utility < 10:
            raise ValueError("max_utility must be an int >= 10")
        self.max_utility = max_utility

        
        self.num_obs = 2 + self.max_agents*self.max_items + self.tape_size + self.max_items + 1

        self.num_actions = self.tape_size + self.max_items + 1

        
        self.num_agents = num_envs
        self.render_mode = render_mode
        
        # self.nen_halt_penalty = nen_halt_penalty
        # self.correctness_reward = correctness_reward
        # self.incorrectness_penalty = incorrectness_penalty
        
        self.log_interval = log_interval

        self.single_observation_space = gymnasium.spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(self.num_obs,),
            dtype=np.float32
        )
        
        self.single_action_space = gymnasium.spaces.Box(
            low=-1,
            high=1,
            shape=(self.num_actions,),
            dtype=np.float32
        )

        super().__init__(buf)

        self.c_envs = binding.vec_init(
            self.observations,
            self.actions,
            self.rewards,
            self.terminals,
            self.truncations,
            num_envs,
            seed,
            nen_halt_penalty=nen_halt_penalty,
            correctness_reward=correctness_reward,
            incorrectness_penalty=incorrectness_penalty,
            max_agents=self.max_agents,
            max_items=self.max_items,
            tape_size=self.tape_size,
            max_utility=self.max_utility
        )
        
    def reset(self, seed=None):
        self.tick = 0      
        if seed is None:
            binding.vec_reset(self.c_envs, 0)
        else:
            binding.vec_reset(self.c_envs, seed)
        return self.observations, []
        
    def step(self, actions):
        self.actions[:] = np.clip(actions.flatten(), -1.0, 1.0)
        binding.vec_step(self.c_envs)
        self.tick += 1
        info = []
        if self.tick % self.log_interval == 0:
            log = binding.vec_log(self.c_envs)
            if log:
                info.append(log)

        return (self.observations, self.rewards, self.terminals, self.truncations, info)
            
    def render(self):
        binding.vec_render(self.c_envs, 0)
        
    def close(self):
        binding.vec_close(self.c_envs)
            
            
def test_performance(num_envs=3, timeout=10, atn_cache=1024):
    env = PolyTime(num_envs=num_envs)
    env.reset(seed=0)
    tick = 0

    actions = np.random.random_sample((atn_cache, num_envs, env.num_actions))

    import time
    start = time.time()
    while time.time() - start < timeout:
        atn = actions[tick % atn_cache]
        env.step(atn)
        tick += 1

    sps = num_envs * tick / (time.time() - start)
    print(f'SPS: {sps:,}')
    
    env.close()
        

if __name__ == "__main__": 
    test_performance(num_envs=3, timeout=10, atn_cache=1024)
    test_performance(num_envs=6, timeout=10, atn_cache=1024)
    test_performance(num_envs=9, timeout=10, atn_cache=1024)
    test_performance(num_envs=18, timeout=10, atn_cache=1024)
    test_performance(num_envs=24, timeout=10, atn_cache=1024)
    test_performance(num_envs=30, timeout=10, atn_cache=1024)
    
    test_performance(num_envs=4, timeout=10, atn_cache=1024)
    test_performance(num_envs=8, timeout=10, atn_cache=1024)
    test_performance(num_envs=12, timeout=10, atn_cache=1024)
    test_performance(num_envs=20, timeout=10, atn_cache=1024)
    test_performance(num_envs=28, timeout=10, atn_cache=1024)
    test_performance(num_envs=36, timeout=10, atn_cache=1024)

    test_performance(num_envs=5, timeout=10, atn_cache=1024)
    test_performance(num_envs=10, timeout=10, atn_cache=1024)
    test_performance(num_envs=25, timeout=10, atn_cache=1024)
    test_performance(num_envs=35, timeout=10, atn_cache=1024)
    test_performance(num_envs=40, timeout=10, atn_cache=1024)
    test_performance(num_envs=45, timeout=10, atn_cache=1024)
    pass