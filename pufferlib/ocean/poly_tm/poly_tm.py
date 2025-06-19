import gymnasium
import numpy as np

import pufferlib
from pufferlib.ocean.poly_tm import binding


class PolyTM(pufferlib.PufferEnv):
    def __init__(self, num_envs=1, render_mode="auto", log_interval=1, nen_halt_penalty=1.0,
            correctness_reward=10.0, incorrectness_penalty=10.0, tape_size=512,
            observation_window=16, work_alphabet=256,
            state_alphabet=256, move_state=10, move_work=10,
            max_a=4, max_i=50, max_u=50, buf=None, seed=0):

        self.tape_size = tape_size
        self.observation_window = observation_window
        
        self.work_alphabet = work_alphabet
        self.state_alphabet = state_alphabet
        self.move_work = move_work
        self.move_state = move_state

        
        self.max_a = max_a
        self.max_i = max_i
        self.max_u = max_u
        
        self.num_obs = 2 * self.observation_window + 1 + 2*self.observation_window+ 1 + 1

        self.num_actions = 5

        
        self.num_agents = num_envs
        self.render_mode = render_mode
        
        self.log_interval = log_interval

        self.single_observation_space = gymnasium.spaces.Box(
            low=0,
            high=max(self.work_alphabet, self.state_alphabet) - 1,
            shape=(self.num_obs,),
            dtype=np.int32
        )
        
        self.single_action_space = gymnasium.spaces.MultiDiscrete(
            [self.work_alphabet, 2 * self.move_work + 1, self.state_alphabet, 2 * self.move_state + 1, self.state_alphabet],
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
            nen_halt_penalty=nen_halt_penalty,
            correctness_reward=correctness_reward,
            incorrectness_penalty=incorrectness_penalty,
            tape_size=self.tape_size,
            observation_window=self.observation_window,
            work_alphabet=self.work_alphabet,
            state_alphabet=self.state_alphabet,
            move_work=self.move_work,
            move_state=self.move_state,
            max_a=self.max_a,
            max_i=self.max_i,
            max_u=self.max_u,
        )
        
    def reset(self, seed=None):
        self.tick = 0      
        if seed is None:
            binding.vec_reset(self.c_envs, 0)
        else:
            binding.vec_reset(self.c_envs, seed)
        return self.observations, []
        
    def step(self, actions):
        self.actions[:] = actions
        
        self.tick += 1
        binding.vec_step(self.c_envs)
        
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