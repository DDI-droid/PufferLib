#include "poly_tm.h"

#define Env PolyTM
#include "../env_binding.h"

static int my_init(Env* env, PyObject* args, PyObject* kwargs) {

    env->work_tape_size = unpack(kwargs, "work_tape_size");
    env->state_tape_size = unpack(kwargs, "state_tape_size");
    env->result_tape_size = unpack(kwargs, "result_tape_size");

    env->work_observation_window = unpack(kwargs, "work_observation_window");
    env->state_observation_window = unpack(kwargs, "state_observation_window");
    env->result_observation_window = unpack(kwargs, "result_observation_window");

    env->tape_alphabet = unpack(kwargs, "tape_alphabet");

    env->move_head = unpack(kwargs, "move_head");

    env->num_work_heads = unpack(kwargs, "num_work_heads");
    env->num_state_heads = unpack(kwargs, "num_state_heads");
    env->num_result_heads = unpack(kwargs, "num_result_heads");
    
    env->nen_halt_penalty = unpack(kwargs, "nen_halt_penalty");
    env->invalid_output_penalty = unpack(kwargs, "invalid_output_penalty");
    env->correctness_reward = unpack(kwargs, "correctness_reward");
    env->incorrectness_penalty = unpack(kwargs, "incorrectness_penalty");

    env->max_steps = unpack(kwargs, "max_steps");

    env->problem_size = unpack(kwargs, "problem_size");

    // env->max_a = unpack(kwargs, "max_a");
    // env->max_i = unpack(kwargs, "max_i");
    // env->max_u = unpack(kwargs, "max_u");

    init(env);
    return 0;
}

static int my_log(PyObject* dict, Log* log)
{
    assign_to_dict(dict, "perf", log->perf);
    assign_to_dict(dict, "score", log->score);
    assign_to_dict(dict, "episode_return", log->episode_return);
    assign_to_dict(dict, "episode_length", log->episode_length);
    assign_to_dict(dict, "running_time", log->running_time);
    assign_to_dict(dict, "correctness", log->correctness);
    assign_to_dict(dict, "n", log->n);
    return 0;
}