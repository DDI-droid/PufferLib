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
#include "raylib.h"

#define min(a, b) (((a) < (b)) ? (a) : (b))

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

typedef struct PolyTime PolyTime;
struct PolyTime {
    Log log;

    /*
    observations [0] - num_agents
    observations[1] - num_items
    observations[2 - 2 + max_agents*max_items;] - U(A_i, I_j),  Utility vector for each agent i and item j
    observations[2 + max_agents*max_items; - 2 + max_agents*max_items; + TAPE_SIZE] - Memory tape
    observations[2 + max_agents*max_items; + TAPE_SIZE - 2 + max_agents*max_items; + TAPE_SIZE + max_items;] - Results allocation
    observations[2 + max_agents*max_items; + TAPE_SIZE + max_items;] - Halt flag (0 or 1)
    */
    float* observations; 
    float* actions;
    float* rewards;
    unsigned char* terminals; 
    
    int tick;

    float* returns;

    //sweep
    float nen_halt_penalty;
    float correctness_reward;
    float incorrectness_penalty;


    int max_agents;
    int max_items;
    int tape_size;
    int max_utility;

    int num_agents;
    int num_items;
    unsigned int* utility;
    
    /*
    Problem structure:
    problem[0] - num_agents
    problem[1] - num_items
    problem[2 - 2 + max_agents*max_items;] - U(A_i, I_j),  Utility vector for each agent i and item j
    */
    
    unsigned int* problem;

    float* tape;

    /*
    Results structure:
    results[0 - max_items;] - allocation
    results[max_items;] - halt flag (0 or 1)
    */
    unsigned int* results;

    int *precomputed_us;
    int *precomputed_um;
};

//function prototypes
bool check_soln_correctness(PolyTime*);

void copy_bits_ui_f(const unsigned int* src, float* dst, size_t count)
{
    _Static_assert(sizeof(unsigned int) == sizeof(float), "unsigned int and float must be the same size");

    memcpy(dst, src, count * sizeof(unsigned int));
}

void copy_bits_f_ui(const float* src, unsigned int* dst, size_t count)
{
    _Static_assert(sizeof(unsigned int) == sizeof(float), "unsigned int and float must be the same size");

    memcpy(dst, src, count * sizeof(float));
}

void init(PolyTime* env) {
    env->problem = (unsigned int*)calloc(2 + env->max_agents * env->max_items, sizeof(unsigned int));
    env->tape = (float*)calloc(env->tape_size, sizeof(float));
    env->results = (unsigned int*)calloc(env->max_items + 1, sizeof(unsigned int));
    env->returns = (float*)calloc(1, sizeof(float));
    env->utility = env->problem + 2;

    env->precomputed_us= (int*)calloc(env->max_agents * env->max_agents, sizeof(int));
    env->precomputed_um = (int*)calloc(env->max_agents * env->max_agents, sizeof(int));

}

void allocate(PolyTime* env) {
    env->observations = (float*)calloc(2 + env->max_agents * env->max_items + env->tape_size + env->max_items + 1, sizeof(float));
    env->actions = (float*)calloc(env->tape_size + env->max_items + 1, sizeof(float));
    env->rewards = (float*)calloc(1, sizeof(float));
    env->terminals = (unsigned char*)calloc(1, sizeof(unsigned char));
    init(env);
}

void free_initialized(PolyTime* env) {
    free(env->problem);
    free(env->tape);
    free(env->results);
    free(env->returns);

    free(env->precomputed_um);
    free(env->precomputed_us);
}

void free_allocated(PolyTime* env) {
    free(env->observations);
    free(env->actions);
    free(env->rewards);
    free(env->terminals);
    free_initialized(env);
}

void compute_observations(PolyTime* env) {
    int obs_index = 0;

   env->observations[obs_index++] = (float)env->num_agents;

    env->observations[obs_index++] = (float)env->num_items;

    // Utility vector U(A_i, I_j) TODO:  no normalization here, but will add in nns
    copy_bits_ui_f(env->utility, env->observations + obs_index, env->max_agents * env->max_items);
    obs_index += env->max_agents * env->max_items;

    // Memory tape TODO: idk if I should normalize this
    memcpy(env->observations + obs_index, env->tape, env->tape_size * sizeof(float));
    obs_index += env->tape_size;

    // Results allocation
    copy_bits_ui_f(env->results, env->observations + obs_index, env->max_items);
    obs_index += env->max_items;

    // Halt flag
    env->observations[obs_index++] = env->results[env->max_items] ? 1.0f : 0.0f;
}

void c_reset(PolyTime* env) {
    env->tick = 0;
    memset(env->observations, 0, (2 + env->max_agents * env->max_items + env->tape_size +
                    env->max_items + 1) * sizeof(float));

    memset(env->actions, 0, (env->tape_size + env->max_items + 1) * sizeof(float));
    memset(env->rewards, 0, 1 * sizeof(float));
    memset(env->terminals, 0, 1 * sizeof(unsigned char));

    //Generate a new problem instance
    srand((unsigned int)time(NULL));


    env->num_agents = rand() % (env->max_agents) + 1;
    env->num_items = rand() % (env->max_items) + 1;


    env->problem[0] = env->num_agents;
    env->problem[1] = env->num_items;

    for (int i = 0; i < env->num_agents * env->num_items; i++) {
        env->utility[i] = rand() % (env->max_utility + 1);
    }


    memset(env->tape, 0, env->tape_size * sizeof(float));

    memset(env->results, 0, (env->max_items + 1) * sizeof(unsigned int));

    compute_observations(env);
}


void c_step(PolyTime* env){
    env->tick += 1;

    env->rewards[0] = -env->nen_halt_penalty;

    // Process actions s_t -> s_t+1
    memcpy(env->tape, env->actions, env->tape_size * sizeof(float));

    copy_bits_f_ui(env->actions + env->tape_size, env->results, env->max_items);
    // Allocation TODO ? data type ?

    env->results[env->max_items] = (unsigned int)env->actions[env->tape_size + env->max_items]; // Halt flag


    if (env->results[env->max_items]) {

        bool res = check_soln_correctness(env);

        if (res) {
            env->rewards[0] += env->correctness_reward;
        } 
        else {
            env->rewards[0] -= env->incorrectness_penalty;
        }

        env->returns[0] += env->rewards[0];
        
        env->log.perf += res ? 1.0f : 0.0f; // Performance metric
        env->log.score += env->rewards[0]; // Score metric TODO: change score
        env->log.episode_length += env->tick; // Episode length metric
        env->log.episode_return += env->returns[0]; // Episode return metric

        env->log.running_time += (float)env->tick; // Running time metric
        env->log.correctness += res ? 1.0f : 0.0f; // Correctness metric


        env->terminals[0] = 1;
        c_reset(env);
    } 
    else
    {
        env->returns[0] += env->rewards[0];
        compute_observations(env);
    }

}

bool check_soln_correctness(PolyTime* env) {
    // checks if the allocation if EFX

    for (int i = 0; i < env->num_agents; i++) {
        for (int j = 0; j < env->num_agents; j++) {
            env->precomputed_um[i * env->max_agents + j] = INT_MAX;
        }
    }

    for (int i = 0; i < env->num_agents; i++) {
        for (int j = 0; j < env->num_items; j++){
            if (env->results[j] >= env->num_agents) {
                // Invalid allocation
                return false;
            }
            env->precomputed_us[i * env->max_agents + env->results[j]] += env->utility[env->max_items * i + j];

            env->precomputed_um[i * env->max_agents + env->results[j]] = min(env->precomputed_um[i * env->max_agents + env->results[j]], env->utility[env->max_items * i + j]);
        }
    }

    for (int i = 0; i < env->num_agents; i++) {
        for (int j = 0; j < env->num_agents; j++) {
            if (i == j) continue;

            if (env->precomputed_us[i * env->max_agents + i] < env->precomputed_us[i * env->max_agents + j] -  env->precomputed_um[i * env->max_agents + j]) {
                // Agent i has less utility than agent j, so the allocation is not EFX
                return false;
            }
        }
    }

    // If we reach here, the allocation is EFX

    return true;
}

void c_render()
{
    ;
}

void c_close(PolyTime* env) {
    if (IsWindowReady())
    {
        CloseWindow();
    }
    free_initialized(env);
}