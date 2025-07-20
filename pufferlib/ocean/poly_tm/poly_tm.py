import gymnasium
import numpy as np

import pufferlib
from pufferlib.ocean.poly_tm import binding


class PolyTM(pufferlib.PufferEnv):
    def __init__(
        self,
        num_envs=1,
        render_mode="auto",
        log_interval=1,
        log_episodes=10,
        work_tape_size=100,
        state_tape_size=100,
        result_tape_size=100,
        work_observation_window=16,
        state_observation_window=16,
        result_observation_window=16,
        tape_alphabet=10,
        move_head=10,
        num_work_heads=3,
        num_state_heads=1,
        num_result_heads=1,
        nen_halt_penalty=1.0,
        invalid_output_penalty=2.0,
        cont_rew_mul=5.0,
        write_rew_multiplier=10.0,
        correctness_reward=20.0,
        incorrectness_penalty=20.0,
        step_rew_multiplier=0.5,
        max_steps = 300,
        problem_size=4,
        buf=None,
        seed=0
        ):
        
        assert tape_alphabet < 255, "tape stores chars!!"
        
        # convert everything to int (okay not everything, but u get the point)
        work_tape_size = int(work_tape_size)
        state_tape_size = int(state_tape_size)
        result_tape_size = int(result_tape_size)
        work_observation_window = int(work_observation_window)
        state_observation_window = int(state_observation_window)
        result_observation_window = int(result_observation_window)
        tape_alphabet = int(tape_alphabet)
        move_head = int(move_head)
        num_work_heads = int(num_work_heads)
        num_state_heads = int(num_state_heads)
        num_result_heads = int(num_result_heads)
        problem_size = int(problem_size)
        
        self.num_obs = ((num_work_heads + num_state_heads + num_result_heads) + problem_size\
                                  + num_work_heads * (2 * work_observation_window + 1) + \
                                    num_state_heads * (2 * state_observation_window + 1) + \
                                    num_result_heads * (2 * result_observation_window + 1))
        
        self.num_heads = num_work_heads + num_state_heads + num_result_heads
        self.num_work_heads = num_work_heads
        self.num_state_heads = num_state_heads
        self.num_result_heads = num_result_heads
        self.problem_size = problem_size
        self.work_tape_size = work_tape_size
        self.state_tape_size = state_tape_size
        self.result_tape_size = result_tape_size
        self.tape_alphabet = tape_alphabet
        self.work_observation_window = work_observation_window
        self.state_observation_window = state_observation_window
        self.result_observation_window = result_observation_window

        self.num_actions = 4
        
        self.log_idx = 0
        
        self.num_agents = num_envs
        self.render_mode = render_mode
        
        self.log_interval = log_interval
        self.log_episodes = log_episodes
        self.logs = 0
        
        self.max_steps = max_steps

        self.single_observation_space = gymnasium.spaces.Box(
            low=-1,
            high=tape_alphabet - 1,
            shape=(self.num_obs,),
            dtype=np.int8
        )
        
        self.single_action_space = gymnasium.spaces.MultiDiscrete(
            [num_work_heads + num_state_heads + num_result_heads, 2, tape_alphabet + 1, 2 * move_head + 1],
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
            work_tape_size=work_tape_size,
            state_tape_size=state_tape_size,
            result_tape_size=result_tape_size,
            work_observation_window=work_observation_window,
            state_observation_window=state_observation_window,
            result_observation_window=result_observation_window,
            tape_alphabet=tape_alphabet,
            move_head=move_head,
            num_work_heads=num_work_heads,
            num_state_heads=num_state_heads,
            num_result_heads=num_result_heads,
            nen_halt_penalty=nen_halt_penalty,
            invalid_output_penalty=invalid_output_penalty,
            cont_rew_mul=cont_rew_mul,
            write_rew_multiplier=write_rew_multiplier,
            correctness_reward=correctness_reward,
            incorrectness_penalty=incorrectness_penalty,
            step_rew_multiplier=step_rew_multiplier,
            max_steps=max_steps,
            problem_size=problem_size
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