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
#include<stdint.h>
#include "raylib.h"

#define min(a, b) (((a) < (b)) ? (a) : (b))
#define max(a, b) (((a) > (b)) ? (a) : (b))

#define OBS_LEN(env) (4 * (env)->observation_window + 2)

typedef struct Log Log;
struct Log{
    float perf; // Recommended 0-1 normalized single real number perf metric
    float score; // Recommended unnormalized single real number perf metric
    float episode_return; // Recommended metric: sum of agent rewards over episode
    float episode_length; // Recommended metric: number of steps of agent episode

    // Any extra fields you add here may be exported to Python in binding.c
    float running_time; // Time taken to solve current env instance
    float correctness; // Is the result of the oracle correct? 0-1


    float n; // Required as the last field 
};

typedef struct PolyTM PolyTM;
struct PolyTM {
    Log log;

    int* observations; 
    int* actions;
    float* rewards;
    float* returns;
    unsigned char* terminals; 
    
    int tick;

    //sweep
    float nen_halt_penalty;
    float correctness_reward;
    float incorrectness_penalty;

    int tape_size;
    int observation_window;

    int work_alphabet;
    int state_alphabet;

    int move_state;
    int move_work;

    int max_steps;

    //problem parameters
    int max_a;
    int max_i;
    int max_u;
    
    int num_a;
    int num_i;
    //

    int *tape_work;
    int *tape_state;

    int head_work;
    int head_state;

    int state;

    int* utility;
    
    int* problem;

    int* results;

    int* halt;

    int *precomputed_us;
    int *precomputed_um;

    uint64_t rng_state;
};

//function prototypes
bool check_soln_correctness(PolyTM*);
bool check_correctness_side_1(PolyTM*);

static inline int clampi(int v, int lo, int hi) { return v < lo ? lo : v > hi ? hi: v; }

static inline uint32_t rng_u32(uint64_t *state)
{
    uint64_t z = (*state += 0x9e3779b97f4a7c15ull);
    z = (z ^ (z >> 30)) * 0xbf58476d1ce4e5b9ull;
    z = (z ^ (z >> 27)) * 0x94d049bb133111ebull;
    return (uint32_t)(z ^ (z >> 31));
}

void init(PolyTM* env) {
    env->tape_work = (int*)calloc(env->tape_size, sizeof(int));
    env->tape_state = (int*)calloc(env->tape_size, sizeof(int));

    env->problem = env->tape_work;
    env->utility =env->problem+2;

    env->results  = env->tape_work + env->tape_size - env->max_i - 1;
    env->halt = env->tape_work + env->tape_size - 1;

    env->returns = (float*)calloc(1, sizeof(float));

    env->precomputed_us= (int*)calloc(env->max_a * env->max_a, sizeof(int));
    env->precomputed_um = (int*)calloc(env->max_a * env->max_a, sizeof(int));

    assert((env->max_a < env->work_alphabet) && "max agents should be within work alphabet");
    assert((env->max_i < env->work_alphabet) && "max items should be within work alphabet");
    assert((env->max_u < env->work_alphabet) && "max utility should be within work alphabet");

    env->rng_state = ((uint64_t)(uintptr_t)env) ^ (uint64_t)time(NULL);
}

void allocate(PolyTM* env) {
    env->observations = (int*)calloc(OBS_LEN(env), sizeof(int));
    env->actions = (int*)calloc(4, sizeof(int));
    env->rewards = (float*)calloc(1, sizeof(float));
    env->terminals = (unsigned char*)calloc(1, sizeof(unsigned char));
    init(env);
}

void free_initialized(PolyTM* env) {
    free(env->tape_work);
    free(env->tape_state);
    free(env->returns);

    free(env->precomputed_um);
    free(env->precomputed_us);
}

void free_allocated(PolyTM* env) {
    free(env->observations);
    free(env->actions);
    free(env->rewards);
    free(env->terminals);
    free_initialized(env);
}

static inline void tape_window_to_obs(const int *tape, int head, int *dst, int w, int tape_size)
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

    memset(dst, 0, padL * sizeof(int));
    memcpy(dst + padL, tape + left, span * sizeof(int));
    memset(dst + padL + span, 0, padR * sizeof(int));
}

void compute_observations(PolyTM* env) {

    const int window = 2 * env->observation_window + 1;

    tape_window_to_obs(env->tape_work, env->head_work, env->observations, env->observation_window, env->tape_size);
    tape_window_to_obs(env->tape_state, env->head_state, env->observations + window, env->observation_window, env->tape_size);

}

void c_reset(PolyTM* env) {
    env->tick = 0;
    memset(env->observations, 0, (OBS_LEN(env)) * sizeof(int));

    memset(env->tape_work, -1, env->tape_size * sizeof(int));
    memset(env->tape_state, -1, env->tape_size * sizeof(int));

    env->head_work = env->tape_size / 2;
    env->head_state = env->tape_size / 2;

    env->returns[0] = 0.0f;

    memset(env->precomputed_um,  0, env->max_a * env->max_a * sizeof(int));
    memset(env->precomputed_us,  0, env->max_a * env->max_a * sizeof(int));

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

    //Side problem-1 init
    env->problem[0] = rng_u32(&env->rng_state) % (10);
    env->problem[1] = rng_u32(&env->rng_state) % (10);

    compute_observations(env);
}


void c_step(PolyTM* env){
    env->tick += 1;

    env->rewards[0] = 0.0f;

    // env->rewards[0] -= env->nen_halt_penalty;
    // env->returns[0] -= env->nen_halt_penalty;

    // Process actions s_t -> s_t+1

    env->tape_work[env->head_work] = env->actions[0];

    env->head_work += (int)env->actions[1] - env->move_work;
    env->head_work = clampi(env->head_work, 0, env->tape_size - 1);

    env->tape_state[env->head_state] = env->actions[2];

    env->head_state += (int)env->actions[3]  - env->move_state;
    env->head_state = clampi(env->head_state, 0, env->tape_size - 1);

    //

    int correct = (env->problem[0] + env->problem[1]) % env->work_alphabet;
    int written = env->results[0];

    if (written < 0)
    {
        env->rewards[0] -= 2.0f;
        env->returns[0] -= 2.0f;
    }
    else
    {
        int delta = abs(correct - written) % env->work_alphabet;

        float bonus = delta * 0.01f; // range 0.0 … +1.0
        
        env->rewards[0] += bonus;
        env->returns[0] += bonus;
    }

    env->terminals[0] = env->halt[0] > 0 ? 1 : 0;

    if (!env->terminals[0]) {
        if (env->tick >= env->max_steps) {
            env->terminals[0] = 1;
        } else {
            env->terminals[0] = 0;
        }
    }

    if (env->terminals[0]) {

        //Main problem halt
        bool res = check_correctness_side_1(env);
        
        if (env->results[0] == -1) {
            res = false;
            env->rewards[0] -= 2.0f;
            env->returns[0] -= 2.0f;
        }
        
        if (res) {
            env->rewards[0] += env->correctness_reward;
            env->returns[0] += env->correctness_reward;
        } 
        else {
            env->rewards[0] -= env->incorrectness_penalty;
            env->returns[0] -= env->incorrectness_penalty;
        }
        
        
        env->log.perf += res ? 1.0f : 0.0f;
        env->log.score += env->returns[0];
        env->log.episode_length += env->tick;
        env->log.episode_return += env->returns[0];

        env->log.running_time += (float)env->tick;
        env->log.correctness += res ? 100.0f : 0.0f; 
        env->log.n += 1.0f;

        c_reset(env);
    }
    

    compute_observations(env);
}

//Main problem correctness
bool check_soln_correctness(PolyTM* env) {
    // checks if the allocation if EFX

    for (int i = 0; i < env->num_a; i++) {
        for (int j = 0; j < env->num_i; j++){

            int id = env->results[j];

            if (id >= env->num_a) {
                // Invalid item allocation
                return false;
            }

            env->precomputed_us[i * env->max_a + id] += env->utility[env->max_i * i + j];

            env->precomputed_um[i * env->max_a + id] = min(env->precomputed_um[i * env->max_a + id], env->utility[env->max_i * i + j]);
        }
    }

    for (int i = 0; i < env->num_a; i++) {
        for (int j = 0; j < env->num_a; j++) {
            if (i == j) continue;

            if (env->precomputed_us[i * env->max_a + i] < env->precomputed_us[i * env->max_a + j] -  env->precomputed_um[i * env->max_a + j]) {
                // Agent i has less utility than agent j, so the allocation is not EFX
                return false;
            }
        }
    }

    // If we reach here, the allocation is EFX

    return true;
}

// side problem-1
bool check_correctness_side_1(PolyTM* env)
{
    if (env->results[0] == (env->problem[0] +  env->problem[1]) % env->work_alphabet)
    {
        return true;
    }
    else
    {
        return false;
    }
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