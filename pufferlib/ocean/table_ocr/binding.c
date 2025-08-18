#include "table_ocr.h"

#define Env TableOCR
#include "../env_binding.h"

static int my_init(Env *env, PyObject *args, PyObject *kwargs)
{
    env->d_position = unpack(kwargs, "d_position");

    env->img_width = unpack(kwargs, "img_width");
    env->img_height = unpack(kwargs, "img_height");

    env->min_steps = unpack(kwargs, "min_steps");
    env->max_steps = unpack(kwargs, "max_steps");

    env->stable_r_coeff = unpack(kwargs, "stable_r_coeff");

    env->epsilon_del = unpack(kwargs, "epsilon_del");

    init(env);
    return 0;
}

static int my_log(PyObject *dict, Log *log)
{
    assign_to_dict(dict, "perf", log->perf);
    assign_to_dict(dict, "score", log->score);
    assign_to_dict(dict, "episode_return", log->episode_return);
    assign_to_dict(dict, "episode_length", log->episode_length);
    assign_to_dict(dict, "n_deleted_rows", log->n_deleted_rows);
    assign_to_dict(dict, "n_good_rows", log->n_good_rows);
    assign_to_dict(dict, "init_good_rows", log->init_good_rows);
    assign_to_dict(dict, "init_perf", log->init_perf);
    assign_to_dict(dict, "n_good_words", log->n_good_words);
    assign_to_dict(dict, "word_perf", log->word_perf);
    assign_to_dict(dict, "init_good_words", log->init_good_words);
    assign_to_dict(dict, "init_word_perf", log->init_word_perf);
    assign_to_dict(dict, "n", log->n);
    return 0;
}