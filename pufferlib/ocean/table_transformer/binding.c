#include "table_transformer.h"

#define Env TableTransformer
#include "../env_binding.h"

static int my_init(Env *env, PyObject *args, PyObject *kwargs)
{
    env->step_penalty = unpack(kwargs, "step_penalty");
    env->d_position = unpack(kwargs, "d_position");


    env->img_width = unpack(kwargs, "img_width");
    env->img_height = unpack(kwargs, "img_height");

    // sweep

    env->r_text_const_1 = unpack(kwargs, "r_text_const_1");
    env->r_cut_const_1 = unpack(kwargs, "r_cut_const_1");
    env->r_row_const_1 = unpack(kwargs, "r_row_const_1");

    env->min_steps = unpack(kwargs, "min_steps");
    env->max_steps = unpack(kwargs, "max_steps");

    env->stable_r_coeff = unpack(kwargs, "stable_r_coeff");
    env->done_r_coeff = unpack(kwargs, "done_r_coeff");


    init(env);
    return 0;
}

static int my_log(PyObject *dict, Log *log)
{
    assign_to_dict(dict, "perf", log->perf);
    assign_to_dict(dict, "score", log->score);
    assign_to_dict(dict, "episode_return", log->episode_return);
    assign_to_dict(dict, "episode_length", log->episode_length);
    assign_to_dict(dict, "n_cut", log->n_cut);
    assign_to_dict(dict, "n_clustered", log->n_clustered);
    assign_to_dict(dict, "n", log->n);
    return 0;
}