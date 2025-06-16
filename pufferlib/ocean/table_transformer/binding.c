#include "table_transformer.h"

#define Env TableTransformer
#include "../env_binding.h"

static int my_init(Env *env, PyObject *args, PyObject *kwargs)
{
    env->num_action_boxes = unpack(kwargs, "num_action_boxes");

    env->img_width = unpack(kwargs, "img_width");
    env->img_height = unpack(kwargs, "img_height");
    env->img_channels = unpack(kwargs, "img_channels");

    // sweep
    env->iou_inside = unpack(kwargs, "iou_inside");
    env->iou_cell = unpack(kwargs, "iou_cell");
    env->iou_row = unpack(kwargs, "iou_row");
    env->iou_col = unpack(kwargs, "iou_col");

    env->r_text_const_1 = unpack(kwargs, "r_text_const_1");
    env->r_cut_const_1 = unpack(kwargs, "r_cut_const_1");
    env->r_cell_const_1 = unpack(kwargs, "r_cell_const_1");
    env->r_row_const_1 = unpack(kwargs, "r_row_const_1");
    env->r_col_const_1 = unpack(kwargs, "r_col_const_1");
    return 0;
}

static int my_log(PyObject *dict, Log *log)
{
    assign_to_dict(dict, "perf", log->perf);
    assign_to_dict(dict, "score", log->score);
    assign_to_dict(dict, "episode_return", log->episode_return);
    assign_to_dict(dict, "episode_length", log->episode_length);
    assign_to_dict(dict, "n", log->n);
    return 0;
}