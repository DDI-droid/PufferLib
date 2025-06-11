#include "poly_time.h"

#define Env PolyTime
#include "../env_binding.h"

static int my_init(Env* env, PyObject* args, PyObject* kwargs) {
    
    env->nen_halt_penalty = unpack(kwargs, "nen_halt_penalty");
    env->correctness_reward = unpack(kwargs, "correctness_reward");
    env->incorrectness_penalty = unpack(kwargs, "incorrectness_penalty");

    env->max_agents = unpack(kwargs, "max_agents");
    env->max_items = unpack(kwargs, "max_items");
    env->tape_size = unpack(kwargs, "tape_size");
    env->max_utility = unpack(kwargs, "max_utility");

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
    //assign_to_dict(dict, "n", log->n);
    return 0;
}