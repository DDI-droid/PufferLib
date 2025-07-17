/*
An Env for learning polynomial time oracles in Pufferlib.
*/

#include <stdlib.h>
#include <stdio.h>
#include <stddef.h>
#include <string.h>
#include <math.h>
#include <limits.h>
#include <float.h>
#include <time.h>
#include <assert.h>
#include <stdint.h>
#include "raylib.h"

#define min(a, b) (((a) < (b)) ? (a) : (b))
#define max(a, b) (((a) > (b)) ? (a) : (b))

#define OBS_LEN(env) ((env->num_work_heads + env->num_state_heads + env->num_result_heads) + env->problem_size + \
                                        env->num_work_heads * (2 * env->work_observation_window + 1) + \
                                        env->num_state_heads * (2 * env->state_observation_window + 1) + \
                                        env->num_result_heads * (2 * env->result_observation_window + 1))

typedef struct Log Log;
struct Log{
    float perf; // Recommended 0-1 normalized single real number perf metric
    float score; // Recommended unnormalized single real number perf metric
    float episode_return; // Recommended metric: sum of agent rewards over episode
    float episode_length; // Recommended metric: number of steps of agent episode

    // Any extra fields you add here may be exported to Python in binding.c
    float running_time; // Time taken to solve current env instance
    float correctness; // Is the result of the oracle correct? 0-100
    float work_writes; // how much memory the tm uses


    float n; // Required as the last field 
};

typedef struct PolyTM PolyTM;
struct PolyTM {
    Log log;

    char* observations; 
    int* actions;
    float* rewards;
    float* returns;
    unsigned char* terminals; 
    
    int tick;

    //sweep
    int work_tape_size;
    int state_tape_size;
    int result_tape_size;

    int work_observation_window;
    int state_observation_window;
    int result_observation_window;

    int tape_alphabet;

    int move_head;

    int num_work_heads;
    int num_state_heads;
    int num_result_heads;

    
    float nen_halt_penalty;
    float invalid_output_penalty;
    float cont_rew_mul;
    float cont_rew_div;
    float write_rew_multiplier;
    float correctness_reward;
    float incorrectness_penalty;

    int max_steps;

    char* tape_work;
    char* tape_state;
    char* tape_result;

    int* work_heads;
    int* state_heads;
    int* result_heads;

    char* problem;
    int problem_size;

    char* work_written;
    // problem parameters
    // int max_a;
    // int max_i;
    // int max_u;

    // int num_a;
    // int num_i;

    //problem pointers
    // int* utility;
    
    // int* problem;

    // int* results;

    // int* halt;

    // int *precomputed_us;
    // int *precomputed_um;

    bool ch_work;
    bool ch_state;
    bool ch_result;

    int ch_head_idx;
    int ch_tape_idx;

    uint64_t rng_state;

    //tmp (optimizations mostly)
    char correct_tmp;

    char correct;
};

//function prototypes
float check_soln_correctness(PolyTM*);
float auxilary_rewards(PolyTM*);

static inline int clampi(int v, int lo, int hi) { return v < lo ? lo : v > hi ? hi: v; }

static inline uint32_t rng_u32(uint64_t* state)
{
    uint64_t z = (*state += 0x9e3779b97f4a7c15ull);
    z = (z ^ (z >> 30)) * 0xbf58476d1ce4e5b9ull;
    z = (z ^ (z >> 27)) * 0x94d049bb133111ebull;
    return (uint32_t)(z ^ (z >> 31));
}

void init(PolyTM* env) {
    env->tape_work = (char*)calloc(env->work_tape_size, sizeof(char));
    env->tape_state = (char*)calloc(env->state_tape_size, sizeof(char));
    env->tape_result = (char*)calloc(env->result_tape_size, sizeof(char));

    env->work_heads = (int*)calloc(env->num_work_heads, sizeof(int));
    env->state_heads = (int*)calloc(env->num_state_heads, sizeof(int));
    env->result_heads = (int*)calloc(env->num_result_heads, sizeof(int));

    env->problem = (char*)calloc(env->problem_size, sizeof(char));
    env->work_written = (char*)calloc(env->work_tape_size, sizeof(char));

    // env->problem = env->tape_work;
    // env->utility =env->problem+2;

    // env->results  = env->tape_work + env->tape_size - env->max_i - 1;
    // env->halt = env->tape_work + env->tape_size - 1;

    env->returns = (float*)calloc(1, sizeof(float));

    // env->precomputed_us= (int*)calloc(env->max_a * env->max_a, sizeof(int));
    // env->precomputed_um = (int*)calloc(env->max_a * env->max_a, sizeof(int));

    // assert((env->max_a < env->work_alphabet) && "max agents should be within work alphabet");
    // assert((env->max_i < env->work_alphabet) && "max items should be within work alphabet");
    // assert((env->max_u < env->work_alphabet) && "max utility should be within work alphabet");

    env->rng_state = ((uint64_t)(uintptr_t)env) ^ (uint64_t)time(NULL);
}

void allocate(PolyTM* env) {
    env->observations = (char*)calloc(OBS_LEN(env), sizeof(char));
    env->actions = (int*)calloc(3, sizeof(int));
    env->rewards = (float*)calloc(1, sizeof(float));
    env->terminals = (unsigned char*)calloc(1, sizeof(unsigned char));
    init(env);
}

void free_initialized(PolyTM* env) {
    free(env->tape_work);
    free(env->tape_state);
    free(env->tape_result);
    
    free(env->work_heads);
    free(env->state_heads);
    free(env->result_heads);
    
    free(env->problem);
    free(env->work_written);

    free(env->returns);

    // free(env->precomputed_um);
    // free(env->precomputed_us);
}

void free_allocated(PolyTM* env) {
    free(env->observations);
    free(env->actions);
    free(env->rewards);
    free(env->terminals);
    c_close(env);
}

static inline void tape_window_to_obs(const char* tape, int head, char* dst, int w, int tape_size)
{
    int left = head - w;
    int right = head + w;

    if (left < 0)
        left = 0;
    if (right >= tape_size)
        right = tape_size - 1;
    if (right - left + 1 > 2 * w + 1)
    {
        left = right - (2 * w);
        // printf("hits!!");
    }

    int padL = w - (head - left);
    int span = right - left + 1;
    int padR = (2 * w + 1) - (padL + span);

    memset(dst, -1, padL * sizeof(char));
    memcpy(dst + padL, tape + left, span * sizeof(char));
    memset(dst + padL + span, -1, padR * sizeof(char));
}

void compute_observations(PolyTM* env) {

    if (env->ch_work)
    {
        tape_window_to_obs(env->tape_work, env->work_heads[env->ch_head_idx], env->observations + ((env->num_work_heads + env->num_state_heads + env->num_result_heads) + env->problem_size + env->ch_head_idx * (2 * env->work_observation_window + 1)), env->work_observation_window, env->work_tape_size);

        for (int i = 0; i < env->num_work_heads; i++)
        {
            if ((i != env->ch_head_idx) && (abs(env->ch_tape_idx - env->work_heads[i]) <= env->work_observation_window))
            {
                int tmp_id = (env->num_work_heads + env->num_state_heads + env->num_result_heads) + env->problem_size + i * (2 * env->work_observation_window + 1) + env->work_observation_window;
                
                env->observations[tmp_id + env->ch_tape_idx - env->work_heads[i]] = env->tape_work[env->ch_tape_idx];

            }
        }

        env->observations[env->ch_head_idx] = env->work_heads[env->ch_head_idx];
    }
    else if(env->ch_state)
    {
        tape_window_to_obs(env->tape_state, env->state_heads[env->ch_head_idx], env->observations + ((env->num_work_heads + env->num_state_heads + env->num_result_heads) + env->problem_size + env->num_work_heads * (2 * env->work_observation_window + 1) + env->ch_head_idx * (2 * env->state_observation_window + 1)), env->state_observation_window, env->state_tape_size);

        for (int i = 0; i < env->num_state_heads; i++)
        {
            if ((i != env->ch_head_idx) && (abs(env->ch_tape_idx - env->state_heads[i]) <= env->state_observation_window))
            {
                int tmp_id = (env->num_work_heads + env->num_state_heads + env->num_result_heads) + env->problem_size + env->num_work_heads * (2 * env->work_observation_window + 1) + i * (2 * env->state_observation_window + 1) + env->state_observation_window;

                env->observations[tmp_id + env->ch_tape_idx - env->state_heads[i]] = env->tape_state[env->ch_tape_idx];
            }
        }

        env->observations[env->num_work_heads + env->ch_head_idx] = env->state_heads[env->ch_head_idx];
    }
    else if(env->ch_result)
    {
        tape_window_to_obs(env->tape_result, env->result_heads[env->ch_head_idx], env->observations + ((env->num_work_heads + env->num_state_heads + env->num_result_heads) + env->problem_size + env->num_work_heads * (2 * env->work_observation_window + 1) + env->num_state_heads * (2 * env->state_observation_window + 1) + env->ch_head_idx * (2 * env->result_observation_window + 1)), env->result_observation_window, env->result_tape_size);

        for (int i = 0; i < env->num_result_heads; i++)
        {
            if ((i != env->ch_head_idx) && (abs(env->ch_tape_idx - env->result_heads[i]) <= env->result_observation_window))
            {
                int tmp_id = (env->num_work_heads + env->num_state_heads + env->num_result_heads) + env->problem_size + env->num_work_heads * (2 * env->work_observation_window + 1) + env->num_state_heads * (2 * env->state_observation_window + 1) + i * (2 * env->result_observation_window + 1) + env->result_observation_window;

                env->observations[tmp_id + env->ch_tape_idx - env->result_heads[i]] = env->tape_result[env->ch_tape_idx];
            }
        }

        env->observations[env->num_work_heads + env->num_state_heads + env->ch_head_idx] = env->result_heads[env->ch_head_idx];
    }
    else
    {
        int consumed = 0;

        for (int i = 0; i < env->num_work_heads; i++)
        {
            env->observations[consumed++] = (char)env->work_heads[i];
        }

        for (int i = 0; i < env->num_state_heads; i++)
        {
            env->observations[consumed++] = (char)env->state_heads[i];
        }

        for (int i = 0; i < env->num_result_heads; i++)
        {
            env->observations[consumed++] = (char)env->result_heads[i];
        }

        memcpy(env->observations + consumed, env->problem, env->problem_size * sizeof(char));
        consumed += env->problem_size;

        for (int i = 0; i < env->num_work_heads; i++)
        {
            tape_window_to_obs(env->tape_work, env->work_heads[i], env->observations + consumed, env->work_observation_window, env->work_tape_size);
            consumed += (2 * env->work_observation_window + 1);
        }

        for (int i = 0; i < env->num_state_heads; i++)
        {
            tape_window_to_obs(env->tape_state, env->state_heads[i], env->observations + consumed, env->state_observation_window, env->state_tape_size);
            consumed += (2 * env->state_observation_window + 1);
        }

        for (int i = 0; i < env->num_result_heads; i++)
        {
            tape_window_to_obs(env->tape_result, env->result_heads[i], env->observations + consumed, env->result_observation_window, env->result_tape_size);
            consumed += (2 * env->result_observation_window + 1);
        }
    }

    env->ch_work = env->ch_state = env->ch_result = false;
    env->ch_head_idx = env->ch_tape_idx = -1;
}

void c_reset(PolyTM* env) {
    env->tick = 0;

    memset(env->tape_work, -1, env->work_tape_size * sizeof(char));
    memset(env->tape_state, -1, env->state_tape_size * sizeof(char));
    memset(env->tape_result, -1, env->result_tape_size * sizeof(char));

    // TODO: better initialization
    memset(env->work_heads, 0, env->num_work_heads * sizeof(int));
    memset(env->state_heads, 0, env->num_state_heads * sizeof(int));
    memset(env->result_heads, 0, env->num_result_heads * sizeof(int));

    // problem init
    env->correct_tmp = 0;
    for (int i = 0; i < env->problem_size; i++) {
        env->problem[i] = (char)((rng_u32(&env->rng_state)) % env->tape_alphabet);
        env->correct_tmp = (char)((env->correct_tmp + env->problem[i]) % env->tape_alphabet);
    }

    memset(env->work_written, 0, env->work_tape_size * sizeof(char));

    env->returns[0] = 0.0f;

    env->ch_work = false;
    env->ch_state = false;
    env->ch_result = false;

    env->ch_head_idx = -1;
    env->ch_tape_idx = -1;

    env->correct = 0;

    // memset(env->precomputed_um,  0, env->max_a * env->max_a * sizeof(int));
    // memset(env->precomputed_us,  0, env->max_a * env->max_a * sizeof(int));

    //Main problem init
    //Generate a new problem instance
    /* at the top of main() or vec_init() once per process */

    // env->num_a = 2;
    // env->num_i = rng_u32(&env->rng_state) % (env->max_i - 1) + 2;


    // env->problem[0] = env->num_a;
    // env->problem[1] = env->num_i;

    // for (int i = 0; i < env->num_a * env->num_i; i++) {
    //     env->utility[i] = rng_u32(&env->rng_state) % (env->max_u + 1);
    // }


    // for (int i = 0; i < env->num_a; i++)
    // {
    //     for (int j = 0; j < env->num_a; j++)
    //     {
    //         env->precomputed_um[i * env->max_a + j] = INT_MAX;
    //     }
    // }

    // memset(env->observations, 0, OBS_LEN(env) * sizeof(int));

    compute_observations(env);
}


void c_step(PolyTM* env){
    env->tick += 1;

    env->rewards[0] = 0.0f;

    // env->rewards[0] -= env->nen_halt_penalty;
    // env->returns[0] -= env->nen_halt_penalty;

    // Process actions s_t -> s_t+1
    bool wrote_halt = false;

    if (env->actions[0] < env->num_work_heads) {
        env->ch_head_idx = env->actions[0];

        env->ch_tape_idx = env->work_heads[env->ch_head_idx];

        env->tape_work[env->ch_tape_idx] = (char)env->actions[1];

        env->work_heads[env->ch_head_idx] += env->actions[2] - env->move_head;
        env->work_heads[env->ch_head_idx] = clampi(env->work_heads[env->ch_head_idx], 0, env->work_tape_size - 1);

        env->ch_work = true;
        env->ch_state = false;
        env->ch_result = false;

        env->work_written[env->ch_tape_idx] = 1;
    }
    else if (env->actions[0] < env->num_work_heads + env->num_state_heads){
        env->ch_head_idx = env->actions[0] - env->num_work_heads;

        env->ch_tape_idx = env->state_heads[env->ch_head_idx];

        env->tape_state[env->ch_tape_idx] = (char)env->actions[1];

        env->state_heads[env->ch_head_idx] += env->actions[2] - env->move_head;
        env->state_heads[env->ch_head_idx] = clampi(env->state_heads[env->ch_head_idx], 0, env->state_tape_size - 1);

        wrote_halt = (env->actions[1] == 0);

        env->ch_work = false;
        env->ch_state = true;
        env->ch_result = false;
    }
    else if (env->actions[0] < env->num_work_heads + env->num_state_heads + env->num_result_heads) {
        env->ch_head_idx = env->actions[0] - env->num_work_heads - env->num_state_heads;

        env->ch_tape_idx = env->result_heads[env->ch_head_idx];

        env->tape_result[env->ch_tape_idx] = (char)env->actions[1];

        env->result_heads[env->ch_head_idx] += env->actions[2] - env->move_head;
        env->result_heads[env->ch_head_idx] = clampi(env->result_heads[env->ch_head_idx], 0, env->result_tape_size- 1);

        env->ch_work = false;
        env->ch_state = false;
        env->ch_result = true;
    }
    else{
        printf("Invalid action %d\n", env->actions[0]);
        assert(!"Head index out of range");
    }

    //

    // int correct = (env->problem[0] + env->problem[1]) % env->tape_alphabet;
    // int written = env->tape_result[0];

    // if (written == -1)
    // {
    //     // env->rewards[0] -= 2.0f;
    //     // env->returns[0] -= 2.0f;
    // }
    // else
    // {
    //     int delta = abs(correct - written) % env->tape_alphabet;

    //     float bonus = delta * 0.1f; // range 0.0 … +1.0
        
    //     env->rewards[0] += bonus;
    //     env->returns[0] += bonus;
    // }

    env->terminals[0] = wrote_halt ? 1 : 0;

    if ((!env->terminals[0]) && (env->tick > env->max_steps)) {
        env->rewards[0] -= env->nen_halt_penalty;
        env->returns[0] -= env->nen_halt_penalty;
    }

    if (env->terminals[0]) {
        float res = check_soln_correctness(env);
        float tmp_rew = auxilary_rewards(env);
        
        if (env->tape_result[0] == -1) {
            env->correct = 0;
            env->rewards[0] -= env->invalid_output_penalty;
            env->returns[0] -= env->invalid_output_penalty;
        }
        else{
            env->rewards[0] += res + tmp_rew;
            env->returns[0] += res + tmp_rew;
        }
        
        if (env->correct) {
            env->rewards[0] += env->correctness_reward;
            env->returns[0] += env->correctness_reward;
        } 
        else {
            env->rewards[0] -= env->incorrectness_penalty;
            env->returns[0] -= env->incorrectness_penalty;
        }
        
        
        env->log.perf += env->correct ? 1.0f : 0.0f;
        env->log.score += env->returns[0];
        env->log.episode_length += env->tick;
        env->log.episode_return += env->returns[0];

        env->log.running_time += (float)env->tick;
        env->log.correctness += env->correct ? 100.0f : 0.0f; 
        //env->log.work_writes += (done in the aux rew function)
        env->log.n += 1.0f;

        c_reset(env);
    }
    

    compute_observations(env);
}

// //Main problem correctness
// bool check_soln_correctness(PolyTM* env) {
//     // checks if the allocation if EFX

//     for (int i = 0; i < env->num_a; i++) {
//         for (int j = 0; j < env->num_i; j++){

//             int id = env->results[j];

//             if (id >= env->num_a) {
//                 // Invalid item allocation
//                 return false;
//             }

//             env->precomputed_us[i * env->max_a + id] += env->utility[env->max_i * i + j];

//             env->precomputed_um[i * env->max_a + id] = min(env->precomputed_um[i * env->max_a + id], env->utility[env->max_i * i + j]);
//         }
//     }

//     for (int i = 0; i < env->num_a; i++) {
//         for (int j = 0; j < env->num_a; j++) {
//             if (i == j) continue;

//             if (env->precomputed_us[i * env->max_a + i] < env->precomputed_us[i * env->max_a + j] -  env->precomputed_um[i * env->max_a + j]) {
//                 // Agent i has less utility than agent j, so the allocation is not EFX
//                 return false;
//             }
//         }
//     }

//     // If we reach here, the allocation is EFX

//     return true;
// }

float check_soln_correctness(PolyTM* env)
{
    if (env->tape_result[0] == env->correct_tmp)
    {
        env->correct = 1;
    }
    else{
        env->correct = 0;
    }

    return env->cont_rew_mul - abs(env->tape_result[0] - env->correct_tmp) / env->cont_rew_div;

}

float auxilary_rewards(PolyTM* env)
{
    for (int i = 0; i < env->work_tape_size; ++i)
    {
        env->log.work_writes += (env->work_written[i] == 1 ? 1.0f : 0.0f);
    }

    float rew = (env->write_rew_multiplier - abs(env->tape_work[0] - env->problem[0] - env->problem[1])) + (env->write_rew_multiplier - abs(env->tape_work[1] - env->problem[2] - env->problem[3]));

    return rew;
}

void c_render(PolyTM* env)
{
    ;
}

void c_close(PolyTM* env) {
    if (IsWindowReady())
    {
        CloseWindow();
    }
    free_initialized(env);
}