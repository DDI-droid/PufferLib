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

#define OBS_LEN(env) (env->tape_size)

#define ATN_LEN(env) (3 + 1)

typedef struct Log Log;
struct Log{
    float perf;
    float score;
    float episode_return;
    float episode_length;

    float running_time;
    float correctness;
    float memory_written;
    float task_1;
    float task_2;

    float n;
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

    int* tape;
    int tape_size;

    int result_idx;
    
    int* problem;
    int problem_size;

    int* result;
    char* halt;

    char* tape_mask;

    int max_steps;
};

typedef enum {
    OP_NOOP,
    OP_ADD,
    OP_MULT
} OP;

// Function prototypes
void c_close(PolyTM* env);

void init(PolyTM* env) {
    env->tape = (int*)calloc(env->tape_size, sizeof(int));

    env->problem = env->tape;
    env->result = env->tape + env->tape_size - 3;

    env->tape_mask = (char*)calloc(env->tape_size, sizeof(char));
    env->halt = env->tape_mask + env->tape_size - 1;

    env->returns = (float*)calloc(1, sizeof(float));
}

void allocate(PolyTM* env) {
    env->observations = (int*)calloc(OBS_LEN(env), sizeof(int));
    env->actions = (int*)calloc(ATN_LEN(env), sizeof(int));
    env->rewards = (float*)calloc(1, sizeof(float));
    env->terminals = (unsigned char*)calloc(1, sizeof(unsigned char));
    init(env);
}

void free_initialized(PolyTM* env) {
    free(env->tape);
    free(env->tape_mask);
    free(env->returns);
}

void free_allocated(PolyTM* env) {
    free(env->observations);
    free(env->actions);
    free(env->rewards);
    free(env->terminals);
    c_close(env);
}

void compute_observations(PolyTM* env) {
    if (env->result_idx >= 0 && env->result_idx < env->tape_size) {
        env->observations[env->result_idx] = env->tape[env->result_idx];
    } else {
        memcpy(env->observations, env->tape, env->tape_size * sizeof(int));
    }
}

void c_reset(PolyTM* env) {
    env->tick = 0;

    memset(env->tape, 0, env->tape_size * sizeof(int));

    srand(time(NULL));
    
    for (int i = 0; i < env->problem_size; i++)
        env->problem[i] = rand() % 10;

    env->returns[0] = 0.0f;

    memset(env->tape_mask, 0, env->tape_size * sizeof(char));

    env->result_idx = -1;

    compute_observations(env);
}


void c_step(PolyTM* env){
    env->tick += 1;

    env->rewards[0] = 0.0f;

    env->result_idx = env->actions[0];
    
    int operand1 = env->actions[1];
    int operand2 = env->actions[2];

    OP operation = (OP)env->actions[3];

    if (env->result_idx >= 0 && env->result_idx < env->tape_size &&
        operand1 >= 0 && operand1 < env->tape_size &&
        operand2 >= 0 && operand2 < env->tape_size) {
        
        switch (operation) {
            case OP_NOOP:
                break;
            case OP_ADD:
                env->tape[env->result_idx] = env->tape[operand1] + env->tape[operand2];
                break;
            case OP_MULT:
                env->tape[env->result_idx] = env->tape[operand1] * env->tape[operand2];
                break;
        }

        if (operation != OP_NOOP)
            env->tape_mask[env->result_idx] = 1;
    }
    else{
        printf("Operation out of bounds!!\n");
    }

    env->terminals[0] = env->halt[0] ? 1 : 0;


    if ((!env->terminals[0]) && (env->tick > env->max_steps))
        env->terminals[0] = 1;

    if (env->terminals[0]) {
        if (env->result[0] == env->problem[0] + env->problem[1])
        {
            env->rewards[0] += 0.5f;
            env->returns[0] += 0.5f;

            env->log.task_1 += 100.0f;
        }

        if (env->result[1] == env->problem[2] + env->problem[3])
        {
            env->rewards[0] += 0.5f;
            env->returns[0] += 0.5f;

            env->log.task_2 += 100.0f;
        }
        
        env->log.perf += env->rewards[0];
        env->log.score += env->returns[0];
        env->log.episode_length += env->tick;
        env->log.episode_return += env->returns[0];

        env->log.running_time += (float)env->tick;
        env->log.correctness += env->rewards[0] * 100; 
        
        for (int i = 0; i < env->tape_size; i++)
        {
            env->log.memory_written += env->tape_mask[i];
        }

        env->log.n++;

        c_reset(env);
    }
    
    compute_observations(env);
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