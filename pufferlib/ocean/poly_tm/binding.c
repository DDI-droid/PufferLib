#include "poly_tm.h"

#define Env PolyTM
#include "../env_binding.h"

static int my_init(Env* env, PyObject* args, PyObject* kwargs) {
    
    env->nen_halt_penalty = unpack(kwargs, "nen_halt_penalty");
    env->correctness_reward = unpack(kwargs, "correctness_reward");
    env->incorrectness_penalty = unpack(kwargs, "incorrectness_penalty");

    env->tape_size = unpack(kwargs, "tape_size");
    env->observation_window = unpack(kwargs, "observation_window");
    env->work_alphabet = unpack(kwargs, "work_alphabet");
    env->state_alphabet = unpack(kwargs, "state_alphabet");
    env->move_state = unpack(kwargs, "move_state");
    env->move_work = unpack(kwargs, "move_work");


    env->max_a = unpack(kwargs, "max_a");
    env->max_i = unpack(kwargs, "max_i");
    env->max_u = unpack(kwargs, "max_u");

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