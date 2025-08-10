#include "poly_tm.h"

#define Env PolyTM
#include "../env_binding.h"

static int my_init(Env* env, PyObject* args, PyObject* kwargs) {

    env->tape_size = unpack(kwargs, "tape_size");

    env->problem_size = unpack(kwargs, "problem_size");

    env->max_steps = unpack(kwargs, "max_steps");

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
    assign_to_dict(dict, "memory_written", log->memory_written);
    assign_to_dict(dict, "task_1", log->task_1);
    assign_to_dict(dict, "task_2", log->task_2);
    return 0;
}