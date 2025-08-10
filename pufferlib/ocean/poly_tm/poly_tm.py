import gymnasium
import numpy as np

import pufferlib
from pufferlib.ocean.poly_tm import binding


class PolyTM(pufferlib.PufferEnv):
    def __init__(
        self,
        num_envs=1024,
        render_mode="auto",
        log_interval=100,
        log_episodes=10,
        tape_size=100,
        problem_size=4,
        max_steps=10,
        buf=None,
        seed=0
    ):
                
        # convert everything to int (okay not everything, but u get the point)
        tape_size = int(tape_size)
        problem_size = int(problem_size)
        max_steps = int(max_steps)
        
        if tape_size <= 10:
            raise ValueError("tape_size must be atleast 10")
        if problem_size <= 0:
            raise ValueError("problem_size must be positive")
        if max_steps <= 0:
            raise ValueError("max_steps must be positive")
        
        self.observation_size = tape_size
        self.problem_size = problem_size

        self.max_steps = max_steps

        self.num_ops = 3

        self.num_actions = 3 + 1
        
        self.log_idx = 0
        
        self.num_agents = num_envs
        self.render_mode = render_mode
        
        self.log_interval = log_interval
        self.log_episodes = log_episodes
        self.logs = 0
        

        self.single_observation_space = gymnasium.spaces.Box(
            low=np.iinfo(np.int32).min,
            high=np.iinfo(np.int32).max,
            shape=(self.observation_size,),
            dtype=np.int32
        )
        
        self.single_action_space = gymnasium.spaces.MultiDiscrete(
            [tape_size]*3 + [self.num_ops],
            dtype=np.int32
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
            tape_size=tape_size,
            problem_size=problem_size,
            max_steps=max_steps
        )
        
    def reset(self, seed=None):
        self.tick = 0
        binding.vec_reset(self.c_envs, seed)
        return self.observations, []
        
    def step(self, actions):
        self.actions[:] = actions
        
        self.tick += 1
        binding.vec_step(self.c_envs)
        
        self.logs += sum(self.terminals) 

        info = []
        if self.logs >= self.log_episodes:
            log = binding.vec_log(self.c_envs)
            if log:
                log['log_idx'] = self.log_idx
                info.append(log)
                self.log_idx += 1
                self.logs = 0

        return (self.observations, self.rewards, self.terminals, self.truncations, info)
            
    def render(self):
        binding.vec_render(self.c_envs, 0)
        
    def close(self):
        binding.vec_close(self.c_envs)
            
            
def test_performance(num_envs=3, timeout=10, atn_cache=1024):
    env = PolyTM(num_envs=num_envs)
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
    test_performance(num_envs=36, timeout=10, atn_cache=1024)