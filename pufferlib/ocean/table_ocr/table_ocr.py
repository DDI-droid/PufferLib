'''Table OCR Environment
 * use this as an environment for Improving Table Cell/Column/Row bbox detection
 * -- approaches
 * -- -- Manual reward signal (very difficult as it seems intractable)
 * -- -- Use a VLM to give rewars (implementation heavy)
 *
 * This environment does the first approach
 '''

import gymnasium
import gymnasium.spaces.multi_discrete
import numpy as np

import pufferlib
from pufferlib.ocean.table_ocr import binding

class TableOCR(pufferlib.PufferEnv):
    def __init__(
        self,
        num_envs=1,
        render_mode='auto',
        log_interval=10,
        log_episodes=10,
        n_row_boxes=46,
        d_position=5,
        img_height=7015,
        img_width=4962,
        min_steps=4,
        max_steps=50,
        stable_r_coeff=0.07,
        epsilon_del=0.1,
        buf=None,
        seed=0
    ):

        n_row_boxes = int(n_row_boxes)
        img_height = int(img_height)
        img_width = int(img_width)
        min_steps = int(min_steps)
        max_steps = int(max_steps)
        log_interval = int(log_interval)
        log_episodes = int(log_episodes)
        num_envs = int(num_envs)

        self.n_row_boxes = n_row_boxes

        self.n_observations = 2 * self.n_row_boxes

        self.n_boxes_action = 2 * self.n_row_boxes
        self.n_halt_action = 1

        self.single_observation_space = gymnasium.spaces.Box(low=-1, high=img_height,
            shape=(self.n_observations,), dtype=np.float32)
        
        self.single_action_space = gymnasium.spaces.MultiDiscrete([3] * (self.n_boxes_action) + [2] * (self.n_halt_action), dtype=np.int32)

        self.log_interval = log_interval
        self.log_episodes = log_episodes
        self.logs = 0
        self.log_idx = 0       

        self.render_mode = render_mode
        self.num_agents = num_envs


        super().__init__(buf)
        
        self.c_envs = binding.vec_init(
            self.observations, 
            self.actions,
            self.rewards,
            self.terminals,
            self.truncations,
            num_envs,
            seed,
            d_position=d_position,
            img_height=img_height,
            img_width=img_width,
            min_steps=min_steps,
            max_steps=max_steps,
            stable_r_coeff=stable_r_coeff,
            epsilon_del=epsilon_del
        )

    def reset(self, seed=None):
        self.tick = 0      
        binding.vec_reset(self.c_envs, seed)
        return self.observations, []

    def step(self, actions):
        self.actions[:] = actions
        binding.vec_step(self.c_envs)

        self.tick += 1
        
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


def test_performance(timeout=20, atn_cache=1024, num_envs=400):
    tick = 0
    import time
    start = time.time()
    while time.time() - start < timeout:
        atns = actions[tick % atn_cache]
        env.step(atns)
        tick += 1

    print(f'SPS: %f', num_envs*tick / (time.time() - start))

if __name__ == '__main__':
    from cProfile import run
    num_envs = 400
    env = TableOCR(num_envs=num_envs, log_interval=10000000)
    env.reset(seed=0)
    actions = np.random.randint(0, env.single_action_space.nvec, (1024, num_envs, env.n_actions), dtype=np.int32)
    test_performance(20, 1024, num_envs)
    exit(0)

    run('test_performance(20)', 'stats.profile')
    import pstats
    from pstats import SortKey
    p = pstats.Stats('stats.profile')
    p.sort_stats(SortKey.TIME).print_stats(25)
    exit(0)