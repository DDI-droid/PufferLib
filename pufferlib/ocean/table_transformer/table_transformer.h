/* Table Transformer Environment
 * use this as an environment for Improving Table Cell/Column/Row bbox detection
 * -- approaches
 * -- -- Manual reward signal (very difficult as it seems intractable)
 * -- -- Use a VLM to give rewards (implementation heavy + need a good enough vlm... although it would work even if the vlm rewards are somewhat jank)
 *
 * This environment does the first approach
 */

#include <stdio.h>
#include <math.h>
#include <stdlib.h>
#include <string.h>
#include "raylib.h"
#include "json.h"

// const char* IMAGE_PATH = "/home/anshult/Documents/Ocean/pufferlib/pufferlib/resources/table_structure.jpg";
const char *WORD_BOXES_PATH = "/media/user/EXT_DRIVE/Anshul/Ocean/pufferlib/pufferlib/resources/table_transformer/word_boxes.json";

// Required struct. Only use floats!
typedef struct Log Log;
struct Log
{
    float perf;           // Recommended 0-1 normalized single real number perf metric
    float score;          // Recommended unnormalized single real number perf metric
    float episode_return; // Recommended metric: sum of agent rewards over episode
    float episode_length; // Recommended metric: number of steps of agent episode
    // Any extra fields you add here may be exported to Python in binding.c
    float n; // Required as the last field
};

typedef struct Client Client;
struct Client
{
    int pass;
};
// Required that you have some struct for your env
// Recommended that you name it the same as the env file
typedef struct TableTransformer TableTransformer;
struct TableTransformer
{
    Log log;
    unsigned char *observations;
    float *actions;
    float *rewards;
    unsigned char *terminals;

    int img_width;
    int img_height;
    int img_channels;

    int num_action_boxes;

    // action output
    float *cell_boxes;
    float *word_boxes;
    int n_cell_boxes;
    int n_word_boxes;

    // sweep params
    float iou_inside;
    float iou_cell;
    float iou_row;
    float iou_col;

    // float w_oob;
    // float w_rspan; reward weights same as the following
    // float w_cspan;

    float r_text_const_1;
    float r_cut_const_1;
    float r_cell_const_1;
    float r_row_const_1;
    float r_col_const_1;
    //

    int tick;

    int max_rps;
    int rps;

    // bool obs_buffer_initialized;
};

// function declarations
void c_close(TableTransformer *);

static char *read_file(const char *path, size_t *out_len)
{
    FILE *f = fopen(path, "rb");
    if (!f)
    {
        perror("fopen");
        return NULL;
    }
    fseek(f, 0, SEEK_END);
    size_t len = ftell(f);
    fseek(f, 0, SEEK_SET);
    char *buf = malloc(len + 1);
    if (!buf)
    {
        fclose(f);
        return NULL;
    }
    if (fread(buf, 1, len, f) != len)
    {
        free(buf);
        fclose(f);
        return NULL;
    }
    buf[len] = '\0';
    fclose(f);
    if (out_len)
        *out_len = len;
    return buf;
}

void init(TableTransformer *env)
{
    // Set the image path here
    // env->img_path = IMAGE_PATH;

    // env->image = LoadImage(env->img_path);
    // if (env->image.data == NULL)
    // {
    //     fprintf(stderr, "Failed to load image: %s\n", env->img_path);
    //     return;
    // }

    // env->img = env->image.data;
    // env->img_width = env->image.width;
    // env->img_height = env->image.height;
    // env->img_channels = env->image.format == PIXELFORMAT_UNCOMPRESSED_GRAYSCALE ? 1 : (env->image.format == PIXELFORMAT_UNCOMPRESSED_R8G8B8A8 ? 4 : 3);

    env->cell_boxes = env->actions;

    size_t json_len;
    char *json_txt = read_file(WORD_BOXES_PATH, &json_len);

    if (json_txt == NULL)
    {
        fprintf(stderr, "Failed to read JSON file: %s\n", WORD_BOXES_PATH);
        return;
    }

    struct json_value_s *root = json_parse(json_txt, json_len);
    if (!root)
    {
        fprintf(stderr, "Failed to parse JSON file: %s\n", WORD_BOXES_PATH);
        free(json_txt);
        return;
    }

    struct json_object_s *root_obj = json_value_as_object(root);
    if (!root_obj)
    {
        fprintf(stderr, "Root is not a JSON object\n");
        free(root);
        free(json_txt);
        return;
    }
    struct json_value_s *wb_val = NULL;
    struct json_object_element_s *element = root_obj->start;
    while (element)
    {
        if (strcmp(element->name->string, "word_boxes") == 0)
        {
            wb_val = element->value;
            break;
        }
        element = element->next;
    }
    if (!wb_val)
    {
        fprintf(stderr, "No 'word_boxes' key found in JSON\n");
        free(root);
        free(json_txt);
        return;
    }

    struct json_array_s *wb_arr = json_value_as_array(wb_val);
    if (!wb_arr)
    {
        fprintf(stderr, "'word_boxes' is not an array\n");
        free(root);
        free(json_txt);
        return;
    }

    env->n_word_boxes = wb_arr->length;

    env->word_boxes = calloc(env->n_word_boxes * 4, sizeof(int));

    struct json_array_element_s *array_element = wb_arr->start;
    for (int i = 0; i < env->n_word_boxes && array_element; ++i)
    {
        struct json_value_s *box = array_element->value;
        struct json_array_s *coords = json_value_as_array(box);

        if (!coords)
        {
            fprintf(stderr, "Box at index %d is not an array\n", i);
            continue;
        }

        struct json_array_element_s *coord_element = coords->start;
        for (int j = 0; j < 4 && coord_element; ++j)
        {
            struct json_value_s *c = coord_element->value;
            struct json_number_s *number = json_value_as_number(c);

            if (number)
            {
                env->word_boxes[i * 4 + j] = (float)strtol(number->number, NULL, 10) /
                                             (j % 2 == 0 ? env->img_width : env->img_height);
            }
            else
            {
                fprintf(stderr, "Coordinate [%d][%d] is not a number\n", i, j);
                env->word_boxes[i * 4 + j] = 0;
            }

            coord_element = coord_element->next;
        }

        array_element = array_element->next;
    }

    env->n_cell_boxes = env->num_action_boxes;

    free(root);
    free(json_txt);

    env->max_rps = 0;
    env->rps = 0;
}

// for c/c++ testing
void allocate(TableTransformer *env)
{
    env->observations = (unsigned char *)calloc(1, sizeof(char));
    env->actions = (float *)calloc(env->num_action_boxes * 5, sizeof(float));
    env->rewards = (float *)calloc(1, sizeof(float));
    env->terminals = (unsigned char *)calloc(1, sizeof(char));

    init(env);
}

void free_initialized(TableTransformer *env)
{
    // UnloadImage(env->image);
    free(env->word_boxes);
}

void free_allocated(TableTransformer *env)
{
    free(env->observations);
    free(env->actions);
    free(env->rewards);
    free(env->terminals);

    c_close(env);
}

static inline float rect_area(const float *b)
{
    float w = b[2] - b[0];
    float h = b[3] - b[1];
    return (w > 0 && h > 0) ? w * h : 0;
}

static inline float intersection(const float *a, const float *b)
{
    float x1 = fmaxf(a[0], b[0]);
    float y1 = fmaxf(a[1], b[1]);
    float x2 = fminf(a[2], b[2]);
    float y2 = fminf(a[3], b[3]);
    float w = x2 - x1;
    float h = y2 - y1;
    return (w > 0 && h > 0) ? w * h : 0;
}

// TODO: confirm implementation
static inline float pairwise_iou(const float *a, const float *b)
{
    float x1 = fmaxf(a[0], b[0]);
    float y1 = fmaxf(a[1], b[1]);
    float x2 = fminf(a[2], b[2]);
    float y2 = fminf(a[3], b[3]);

    float iw = fmaxf(0.0f, x2 - x1);
    float ih = fmaxf(0.0f, y2 - y1);
    float inter = iw * ih;

    float area_a = (a[2] - a[0]) * (a[3] - a[1]);
    float area_b = (b[2] - b[0]) * (b[3] - b[1]);
    float union_ = area_a + area_b - inter + 1e-6f;

    return inter / union_;
}

float cell_text_reward_hier(TableTransformer *env)
{

    unsigned char *table_mask = (unsigned char *)calloc(env->n_cell_boxes, sizeof(char));
    unsigned char *cell_mask = (unsigned char *)calloc(env->n_cell_boxes, sizeof(char));

    int n_cells = 0;

    for (int i = 0; i < env->n_cell_boxes; ++i)
    {
        if (env->cell_boxes[i * 5 + 4] == 0)
        {
            table_mask[i] = 1;
        }
        else // TODO confirm: assuming there are only tables and cells and nothing else
        {
            cell_mask[i] = 1;
            n_cells++;
        }
    }

    if (n_cells == 0)
    {
        free(table_mask);
        free(cell_mask);
        return -5.0f; 
    }

    // 2) word-inside-cell metrics
    float r_text = -1, r_cut = -1;

    if (n_cells > 0)
    {
        int inside = 0, cut = 0;

        for (int i = 0; i < env->n_word_boxes; ++i)
        {
            float *wbox = (float *)&env->word_boxes[4 * i];
            float warea = rect_area(wbox);
            float max_ioi = 0.0f;

            for (int j = 0; j < env->n_cell_boxes; ++j)
            {
                if (!cell_mask[j])
                    continue;

                float *cbox = (float *)&env->cell_boxes[5 * j];
                float inter = intersection(wbox, cbox);
                float ioi = inter / (warea + 1e-6f);

                if (ioi > max_ioi)
                    max_ioi = ioi;
            }

            if (max_ioi >= env->iou_inside)
            {
                inside++;
                env->rps++;
            }
            else if (max_ioi > 0)
            {
                cut++;
                env->rps--;
            }
            env->max_rps++;
        }

        float frac_in = (float)inside / env->n_word_boxes;
        float fran_cut = (float)cut / env->n_word_boxes;

        r_text = env->r_text_const_1 * frac_in; // - env->r_text_const_2; // sweep can mislead due to r_text_const_2

        r_cut = -env->r_cut_const_1 * fran_cut;
    }
    // else
    // {
    //     return 0.0f;
    // }

    // 3) overall cell-cell overlap
    float r_cell= 0.0;
    float overlap = 0.0f;

    // if (n_cells > 1)
    // {
    //     for (int i = 0; i < env->n_cell_boxes; ++i)
    //     {
    //         if (!cell_mask[i])
    //             continue;

    //         float *box_i = (float *)&env->cell_boxes[5 * i];

    //         for (int j = i + 1; j < env->n_cell_boxes; ++j)
    //         {
    //             if (!cell_mask[j])
    //                 continue;

    //             float *box_j = (float *)&env->cell_boxes[5 * j];
    //             float iou = pairwise_iou(box_i, box_j);

    //             if (iou > env->iou_cell)
    //             {
    //                 overlap++;
    //                 env->rps--;
    //             }
    //         }
    //     }

    //     float frac_cc = (float)overlap / (n_cells * (n_cells - 1));

    //     r_cell = -env->r_cell_const_1 * frac_cc;
    // }
    // else
    // {
    //     return 0.0;
    // }

    // 4) row-row and col-col: reuse cell-cell logic but per label
    // float r_row = 0, r_col = 0;

    // if (n_cells > 1)
    // {
    //     float bad_r = 0, bad_c = 0;
    //     int num_rows = 0, num_columns = 0;

    //     for (int i = 0; i < env->n_cell_boxes; ++i)
    //         if (env->cell_boxes[i * 5 + 4] == 2)
    //             num_rows++; // TODO: check labels
    //     for (int i = 0; i < env->n_cell_boxes; ++i)
    //         if (env->cell_boxes[i * 5 + 4] == 1)
    //             num_columns++; // TODO: check labels

    //     if (num_rows > 1)
    //     {
    //         for (int i = 0; i < env->n_cell_boxes; ++i)
    //         {
    //             if (env->cell_boxes[i * 5 + 4] == 2)
    //             {
    //                 float *bi = (float *)&env->cell_boxes[5 * i];
    //                 for (int j = i + 1; j < env->n_cell_boxes; ++j)
    //                 {
    //                     if (env->cell_boxes[j * 5 + 4] == 2)
    //                     {
    //                         float *bj = (float *)&env->cell_boxes[5 * j];
    //                         if (pairwise_iou(bi, bj) > env->iou_row)
    //                         {
    //                             bad_r++;
    //                             env->rps--;
    //                         }
    //                     }
    //                 }
    //             }
    //         }
    //         r_row = -env->r_row_const_1 * (bad_r / (num_rows * (num_rows - 1)));
    //     }
    //     // else
    //     // {

    //     // }

    //     if (num_columns > 1)
    //     {
    //         for (int i = 0; i < env->n_cell_boxes; ++i)
    //         {
    //             if (env->cell_boxes[i * 5 + 4] == 1)
    //             {
    //                 float *bi = (float *)&env->cell_boxes[5 * i];
    //                 for (int j = i + 1; j < env->n_cell_boxes; ++j)
    //                 {
    //                     if (env->cell_boxes[5 * j + 4] == 1)
    //                     {
    //                         float *bj = (float *)&env->cell_boxes[5 * j];
    //                         if (pairwise_iou(bi, bj) > env->iou_col)
    //                         {
    //                             bad_c++;
    //                             env->rps--;
    //                         }
    //                     }
    //                 }
    //             }
    //             r_col = -env->r_col_const_1 * (bad_c / (num_columns * (num_columns - 1)));
    //         }
    //     }
        // else
        // {

        // }
    //}
    // else
    // {

    // }

    // TODO: Section 5, 6, 7

    free(table_mask);
    free(cell_mask);

    return r_text + r_cut + r_cell;
}

void compute_observations(TableTransformer *env)
{
    // if (env->obs_buffer_initialized) {
    //     return;
    // }

    // memset(env->observations, 0, env->max_channels * env->max_height * env->max_width * sizeof(char));

    // if (env->max_channels < env->img_channels || env->max_height < env->img_height || env->max_width < env->img_width) {
    //     fprintf(stderr, "Error: Image dimensions exceed max dimensions.\n");
    //     return;
    // }

    // // Copy image data to observations
    // for (int i = 0; i < env->img_height; ++i) {
    //     for (int j = 0; j < env->img_width; ++j) {
    //         for (int c = 0; c < env->img_channels; ++c) {

    //             unsigned char image_pixel = env->img[(i * env->img_width + j) * env->img_channels + c];

    //             env->observations[(i * env->max_width + j) * env->max_channels + c] = image_pixel;
    //         }
    //     }
    // }

    // env->obs_buffer_initialized = true;
    env->observations[0] = 1;
}

// Required function
void c_reset(TableTransformer *env)
{
    env->tick = 0;

    env->max_rps = 0;
    env->rps = 0;

    compute_observations(env);
}

// Required function TODO
void c_step(TableTransformer *env)
{
    env->tick += 1;

    for (int k = 0; k < env->num_action_boxes * 5; ++k)
    {
        float a = env->actions[k];
        if (!isfinite(a) || a < 0.0001f || a > 10.0001f)
            a = 0.f;
        env->actions[k] = fminf(fmaxf(a, 0.f), 10.f);
    }

    env->rewards[0] = cell_text_reward_hier(env);
    env->terminals[0] = 1;

    // log
    env->log.perf += (float)(env->rps + 1e-6) ;
    env->log.score += env->rewards[0];
    env->log.episode_return += env->rewards[0];
    env->log.episode_length += 1;
    env->log.n += 1;
    
    c_reset(env);
    compute_observations(env);
}

// Required function. Should handle creating the client on first call Not required here
void c_render(TableTransformer *env)
{
    ;
}

// Required function. Should clean up anything you allocated
// Do not free env->observations, actions, rewards, terminals
void c_close(TableTransformer *env)
{
    if (IsWindowReady())
    {
        CloseWindow();
    }
    free_initialized(env);
}