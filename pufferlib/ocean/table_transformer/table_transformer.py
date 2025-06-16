'''Table Transformer Environment
 * use this as an environment for Improving Table Cell/Column/Row bbox detection
 * -- approaches
 * -- -- Manual reward signal (very difficult as it seems intractable)
 * -- -- Use a VLM to give rewars (implementation heavy)
 *
 * This environment does the first approach
 '''

import gymnasium
import numpy as np

import pufferlib
from pufferlib.ocean.table_transformer import binding

class TableTransformer(pufferlib.PufferEnv):
    def __init__(self, num_envs=1, render_mode=None,
                num_action_boxes=125, iou_inside=0.5, iou_cell=0.5,
                iou_row=0.5, iou_col=0.5, r_text_const_1=5, r_cut_const_1=5,
                r_cell_const_1=5, r_row_const_1=5, r_col_const_1=5,
                img_channels=4, img_height=2600, img_width=3300, buf=None, seed=0):

        self.n_observations = 1
        self.n_actions = num_action_boxes * 5

        self.log_interval = 1

        self.single_observation_space = gymnasium.spaces.Box(low=0, high=1,
            shape=(self.n_observations,), dtype=np.float32)
        self.single_action_space = gymnasium.spaces.Box(low=0, high=6,
            shape=(self.n_actions,), dtype=np.float32)
        
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
            num_action_boxes=num_action_boxes,
            img_channels=img_channels,
            img_height=img_height,
            img_width=img_width,
            iou_inside=iou_inside,
            iou_cell=iou_cell,
            iou_row=iou_row,
            iou_col=iou_col,
            r_text_const_1=r_text_const_1,
            r_cut_const_1=r_cut_const_1,
            r_cell_const_1=r_cell_const_1,
            r_row_const_1=r_row_const_1,
            r_col_const_1=r_col_const_1
        )

    def reset(self, seed=None):
        self.tick = 0      
        if seed is None:
            binding.vec_reset(self.c_envs, 0)
        else:
            binding.vec_reset(self.c_envs, seed)
        return self.observations, []

    def step(self, actions):
        self.actions[:] = np.clip(actions, 0.0, 10.0)
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
    env = TableTransformer(num_envs=num_envs)
    env.reset(seed=0)
    tick = 0

    actions = np.random.random_sample((atn_cache, num_envs, env.n_actions))

    import time
    start = time.time()
    while time.time() - start < timeout:
        atn = actions[tick % atn_cache]
        env.step(atn)
        tick += 1

    sps = num_envs * tick / (time.time() - start)
    print(f'SPS: {sps:,}')
    
    env.close()

if __name__ == '__main__':
    test_performance(num_envs=100, timeout=10, atn_cache=1024)