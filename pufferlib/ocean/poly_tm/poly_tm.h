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

#define OBS_LEN(env) ((env->num_work_heads + env->num_state_heads + env->num_result_heads) + \
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
    float write_corr_1;
    float write_corr_2;


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
    float write_rew_multiplier;
    float correctness_reward;
    float incorrectness_penalty;
    float step_rew_multiplier;

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

    bool ch_work;
    bool ch_state;
    bool ch_result;

    int ch_head_idx;
    int ch_tape_idx;

    //tmp (optimizations mostly)
    char correct_tmp;

    bool correct;
};

//function prototypes
float check_soln_correctness(PolyTM*);
float auxilary_rewards(PolyTM*);

static inline int clampi(int v, int lo, int hi) { return v < lo ? lo : v > hi ? hi: v; }

void init(PolyTM* env) {
    env->tape_work = (char*)calloc(env->work_tape_size, sizeof(char));
    env->tape_state = (char*)calloc(env->state_tape_size, sizeof(char));
    env->tape_result = (char*)calloc(env->result_tape_size, sizeof(char));

    env->work_heads = (int*)calloc(env->num_work_heads, sizeof(int));
    env->state_heads = (int*)calloc(env->num_state_heads, sizeof(int));
    env->result_heads = (int*)calloc(env->num_result_heads, sizeof(int));

    env->problem = env->tape_work;
    env->work_written = (char*)calloc(env->work_tape_size, sizeof(char));

    env->returns = (float*)calloc(1, sizeof(float));
}

void allocate(PolyTM* env) {
    env->observations = (char*)calloc(OBS_LEN(env), sizeof(char));
    env->actions = (int*)calloc(4, sizeof(int));
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
    
    free(env->work_written);

    free(env->returns);
}

void free_allocated(PolyTM* env) {
    free(env->observations);
    free(env->actions);
    free(env->rewards);
    free(env->terminals);
    c_close(env);
}

static inline void tape_window_to_obs(const char* tape, int head_idx, char* dst, int w, int tape_size)
{
    int left = head_idx - w;
    int right = head_idx + w;

    if (left < 0)
        left = 0;
    if (right >= tape_size)
        right = tape_size - 1;
    if (right - left + 1 > 2 * w + 1)
    {
        left = right - (2 * w);
        // printf("hits!!");
    }

    int padL = w - (head_idx - left);
    int span = right - left + 1;
    int padR = (2 * w + 1) - (padL + span);

    memset(dst, -1, padL * sizeof(char));
    memcpy(dst + padL, tape + left, span * sizeof(char));
    memset(dst + padL + span, -1, padR * sizeof(char));
}

void compute_observations(PolyTM* env) {

    if (env->ch_work)
    {
        tape_window_to_obs(env->tape_work, env->work_heads[env->ch_head_idx], env->observations + ((env->num_work_heads + env->num_state_heads + env->num_result_heads) + env->ch_head_idx * (2 * env->work_observation_window + 1)), env->work_observation_window, env->work_tape_size);

        for (int i = 0; i < env->num_work_heads; i++)
        {
            if ((i != env->ch_head_idx) && (abs(env->ch_tape_idx - env->work_heads[i]) <= env->work_observation_window))
            {
                int tmp_id = (env->num_work_heads + env->num_state_heads + env->num_result_heads) + i * (2 * env->work_observation_window + 1) + env->work_observation_window;
                
                env->observations[tmp_id + env->ch_tape_idx - env->work_heads[i]] = env->tape_work[env->ch_tape_idx];

            }
        }

        env->observations[env->ch_head_idx] = env->work_heads[env->ch_head_idx];
    }
    else if(env->ch_state)
    {
        tape_window_to_obs(env->tape_state, env->state_heads[env->ch_head_idx], env->observations + ((env->num_work_heads + env->num_state_heads + env->num_result_heads) + env->num_work_heads * (2 * env->work_observation_window + 1) + env->ch_head_idx * (2 * env->state_observation_window + 1)), env->state_observation_window, env->state_tape_size);

        for (int i = 0; i < env->num_state_heads; i++)
        {
            if ((i != env->ch_head_idx) && (abs(env->ch_tape_idx - env->state_heads[i]) <= env->state_observation_window))
            {
                int tmp_id = (env->num_work_heads + env->num_state_heads + env->num_result_heads) + env->num_work_heads * (2 * env->work_observation_window + 1) + i * (2 * env->state_observation_window + 1) + env->state_observation_window;

                env->observations[tmp_id + env->ch_tape_idx - env->state_heads[i]] = env->tape_state[env->ch_tape_idx];
            }
        }

        env->observations[env->num_work_heads + env->ch_head_idx] = env->state_heads[env->ch_head_idx];
    }
    else if(env->ch_result)
    {
        tape_window_to_obs(env->tape_result, env->result_heads[env->ch_head_idx], env->observations + ((env->num_work_heads + env->num_state_heads + env->num_result_heads) + env->num_work_heads * (2 * env->work_observation_window + 1) + env->num_state_heads * (2 * env->state_observation_window + 1) + env->ch_head_idx * (2 * env->result_observation_window + 1)), env->result_observation_window, env->result_tape_size);

        for (int i = 0; i < env->num_result_heads; i++)
        {
            if ((i != env->ch_head_idx) && (abs(env->ch_tape_idx - env->result_heads[i]) <= env->result_observation_window))
            {
                int tmp_id = (env->num_work_heads + env->num_state_heads + env->num_result_heads) + env->num_work_heads * (2 * env->work_observation_window + 1) + env->num_state_heads * (2 * env->state_observation_window + 1) + i * (2 * env->result_observation_window + 1) + env->result_observation_window;

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

    srand(time(NULL));
    // problem init
    env->correct_tmp = 0;
    for (int i = 0; i < env->problem_size; i++) {
        env->problem[i] = (char)(rand() % env->tape_alphabet);
        env->correct_tmp = (char)((env->correct_tmp + env->problem[i]) % env->tape_alphabet);
    }

    memset(env->work_written, 0, env->work_tape_size * sizeof(char));

    env->returns[0] = 0.0f;

    env->ch_work = false;
    env->ch_state = false;
    env->ch_result = false;

    env->ch_head_idx = -1;
    env->ch_tape_idx = -1;

    env->correct = false;

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

        if ((env->actions[1] == 1) && (env->ch_tape_idx >= env->problem_size))
            env->tape_work[env->ch_tape_idx] = (char)env->actions[2] - 1;

        env->work_heads[env->ch_head_idx] += env->actions[3] - env->move_head;
        env->work_heads[env->ch_head_idx] = clampi(env->work_heads[env->ch_head_idx], 0, env->work_tape_size - 1);

        env->ch_work = true;
        env->ch_state = false;
        env->ch_result = false;

        if ((env->actions[1] == 1) && (env->ch_tape_idx >= env->problem_size))
        {
            if (env->actions[2] == 0)
                env->work_written[env->ch_tape_idx] = 0;
            else
                env->work_written[env->ch_tape_idx] = 1;
        }
    }
    else if (env->actions[0] < env->num_work_heads + env->num_state_heads){
        env->ch_head_idx = env->actions[0] - env->num_work_heads;

        env->ch_tape_idx = env->state_heads[env->ch_head_idx];

        if (env->actions[1] == 1)
            env->tape_state[env->ch_tape_idx] = (char)env->actions[2] - 1;

        env->state_heads[env->ch_head_idx] += env->actions[3] - env->move_head;
        env->state_heads[env->ch_head_idx] = clampi(env->state_heads[env->ch_head_idx], 0, env->state_tape_size - 1);

        if (env->actions[1] == 1)
            wrote_halt = (env->actions[2] == 0);

        env->ch_work = false;
        env->ch_state = true;
        env->ch_result = false;
    }
    else if (env->actions[0] < env->num_work_heads + env->num_state_heads + env->num_result_heads) {
        env->ch_head_idx = env->actions[0] - env->num_work_heads - env->num_state_heads;

        env->ch_tape_idx = env->result_heads[env->ch_head_idx];

        if (env->actions[1] == 1)
            env->tape_result[env->ch_tape_idx] = (char)env->actions[2] - 1;

        env->result_heads[env->ch_head_idx] += env->actions[3] - env->move_head;
        env->result_heads[env->ch_head_idx] = clampi(env->result_heads[env->ch_head_idx], 0, env->result_tape_size- 1);

        env->ch_work = false;
        env->ch_state = false;
        env->ch_result = true;
    }
    else{
        printf("Invalid action %d\n", env->actions[0]);
        assert(!"Head index out of range");
    }

    env->terminals[0] = wrote_halt ? 1 : 0;

    // float step_rew = env->step_rew_multiplier * auxilary_rewards(env);
    // env->rewards[0] += step_rew;
    // env->returns[0] += step_rew;

    if ((!env->terminals[0]) && (env->tick > env->max_steps)) {
        // env->rewards[0] -= env->nen_halt_penalty;
        // env->returns[0] -= env->nen_halt_penalty;
        env->terminals[0] = 1;
    }

    if (env->terminals[0]) {
        // float res = check_soln_correctness(env);
        float tmp_rew = auxilary_rewards(env);
        
        // if (env->tape_result[0] == -1) {
        //     env->rewards[0] -= env->invalid_output_penalty;
        //     env->returns[0] -= env->invalid_output_penalty;
        // }
            
        env->rewards[0] += tmp_rew;
        env->returns[0] += tmp_rew;

        // env->rewards[0] += env->max_steps - env->tick;
        // env->returns[0] += env->max_steps - env->tick;
        
        if (env->correct) {
            // env->rewards[0] += env->correctness_reward;
            // env->returns[0] += env->correctness_reward;
        } 
        else {
            // env->rewards[0] -= env->incorrectness_penalty;
            // env->returns[0] -= env->incorrectness_penalty;
        }
        
        
        env->log.perf += env->correct ? 1.0f : 0.0f;
        env->log.score += env->returns[0];
        env->log.episode_length += env->tick;
        env->log.episode_return += env->returns[0];

        env->log.running_time += (float)env->tick;
        env->log.correctness += env->correct ? 100.0f : 0.0f; 
        
        for (int i = 0; i < env->work_tape_size; ++i)
        {
            env->log.work_writes += (env->work_written[i] == 1 ? 1.0f : 0.0f);
        }

        if (env->tape_work[4] != -1)
            env->log.write_corr_1 += (env->tape_work[4] == (env->problem[0] + env->problem[1]) % env->tape_alphabet) ? 100.0f : 0.0f;
        else
            env->log.write_corr_1 += 0.0f;

        if (env->tape_work[5] != -1)
            env->log.write_corr_2 += (env->tape_work[5] == (env->problem[2] + env->problem[3]) % env->tape_alphabet) ? 100.0f : 0.0f;
        else
            env->log.write_corr_2 += 0.0f;

        env->log.n += 1.0f;

        c_reset(env);
    }
    
    compute_observations(env);
}

float check_soln_correctness(PolyTM* env)
{
    if (env->tape_result[0] == -1)
    {
        env->correct = false;
        return 0;
    }
    else if (env->tape_result[0] == env->correct_tmp)
    {
        env->correct = true;
        return env->cont_rew_mul;
    }
    else{
        env->correct = false;
        return env->cont_rew_mul - abs(env->tape_result[0] - env->correct_tmp);
    }
}

float auxilary_rewards(PolyTM* env)
{
    float rew = 0;
    
    if (env->tape_work[4] == (env->problem[0] + env->problem[1]) % env->tape_alphabet)
        rew += env->write_rew_multiplier;
     
    if (env->tape_work[5] == (env->problem[2] + env->problem[3]) % env->tape_alphabet)    
        rew += env->write_rew_multiplier;

    return rew;
}

void c_render(PolyTM* env)
{
    //TODO ;-;: gud render here :)
}

void c_close(PolyTM* env) {
    if (IsWindowReady())
    {
        CloseWindow();
    }
    free_initialized(env);
}