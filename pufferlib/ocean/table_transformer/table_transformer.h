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
#include <assert.h>
#include <errno.h>
#include <limits.h>
#include "raylib.h"
#include "raymath.h"


const char* WORD_PATH = "/media/dpa/data/Anshul/Ocean/pufferlib/pufferlib/resources/table_transformer/table_p1_words.txt";
const char* CELL_BOXES_PATH = "/media/dpa/data/Anshul/Ocean/pufferlib/pufferlib/resources/table_transformer/table_p1_rows.txt";
const char* CLUSTERS_PATH = "/media/dpa/data/Anshul/Ocean/pufferlib/pufferlib/resources/table_transformer/table_p1_clusters.txt";

#define max(a, b) ((a) > (b) ? (a) : (b))
#define min(a, b) ((a) < (b) ? (a) : (b))

static inline Rectangle scale_rect(const float *b, float scale, int off_x, int off_y)
{
    return (Rectangle){
        b[0] * scale + off_x,
        b[1] * scale + off_y,
        (b[2] - b[0]) * scale,
        (b[3] - b[1]) * scale
    };
}

// Required struct. Only use floats!
typedef struct Log Log;
struct Log
{
    float perf;           // Recommended 0-1 normalized single real number perf metric
    float score;          // Recommended unnormalized single real number perf metric
    float episode_return; // Recommended metric: sum of agent rewards over episode
    float episode_length; // Recommended metric: number of steps of agent episode
    // Any extra fields you add here may be exported to Python in binding.c
    float n_cut;
    float n_clustered;
    float max_rps;
    float rps;
    float n; // Required as the last field
};

typedef struct { float y1, y2; } Span;

typedef struct Client Client;
struct Client
{
    Texture2D table; // Texture for rendering table

    float zoom;
    float min_zoom;
    float max_zoom;
    Vector2 offset;
};
// Required that you have some struct for your env
// Recommended that you name it the same as the env file
typedef struct TableTransformer TableTransformer;
struct TableTransformer
{
    Log log;
    float* observations;
    char* actions;
    float* rewards;
    float* returns;
    unsigned char* terminals;

    float* cell_boxes;
    float* word_boxes;
    int* cluster_idx;
    int n_cell_boxes;
    int n_word_boxes;
    int n_clusters;
    int num_clusters;
    int max_clstr_size;

    int img_width;
    int img_height;
    // sweep params

    float r_text_const_1;
    float r_cut_const_1;
    float r_row_const_1;
    
    float step_penalty;

    float d_position;

    int min_steps;
    int max_steps;

    //

    int tick;

    int max_rps;
    int rps;
    int n_cut;
    int n_clustered;

    float* state_pos;
    //unsigned int* state_sts;

    int n_row;//, n_col;
    
    float* row_y1; 
    float* row_y2;
    // float* col_x1; 
    // float* col_x2;

    int size_x;
    int size_y;

    unsigned char start_flag;

    Client* client; // For rendering

    Span *spans; // For precomp

    float stable_r_coeff;
    float done_r_coeff;

    int* cluster_comp;
    int* cluster_comp_write_head;

    int* cell_freq_map;
    int* cell_max_freq_map;
};

// function declarations
void c_close(TableTransformer *);

static int cmp_span_by_top(const void *a, const void *b)
{
    const float diff = ((const Span *)a)->y1 - ((const Span *)b)->y1;
    return (diff > 0.f) - (diff < 0.f);     /* faster than branching */
}

static void precomp(TableTransformer *env)
{
    const int n_cells = env->n_cell_boxes;
    const float EPS   = 1e-3f;

    int n_spans = 0;

    for (int i = 0; i < n_cells; ++i) {
        float y1 = env->state_pos[4 * i + 1];
        float y2 = env->state_pos[4 * i + 3];
        if (y2 < y1) { float tmp = y1; y1 = y2; y2 = tmp; }  /* sanity */
        env->spans[n_spans++] = (Span){ y1, y2 };
    }

    /* 2. Sort by top coordinate ---------------------------------------- */
    qsort(env->spans, n_spans, sizeof(Span), cmp_span_by_top);


    env->n_row = n_spans;
    for (int i = 0; i < n_spans; ++i) {
        env->row_y1[i] = env->spans[i].y1;
        env->row_y2[i] = env->spans[i].y2;
    }
}


static char* slurp_file(const char* path) {
    FILE* f = fopen(path, "rb");
    if (!f) { perror("fopen"); exit(1); }
    fseek(f, 0, SEEK_END);
    long sz = ftell(f);
    fseek(f, 0, SEEK_SET);
    char* buf = malloc(sz + 1);
    if (!buf) { perror("malloc"); exit(1); }
    if (fread(buf, 1, sz, f) != (size_t)sz) {
        perror("fread");
        exit(1);
    }
    fclose(f);
    buf[sz] = '\0';
    return buf;
}

// Parse every float in buf into a dynamically growing array.
// Returns malloc’d float*, sets *out_count to number of elements.
float* parse_floats_strtof(const char* buf, int* out_count) {
    size_t cap = 16, len = 0;
    float* arr = malloc(cap * sizeof *arr);
    if (!arr) { perror("malloc"); exit(1); }

    char* end;
    const char* p = buf;
    while (*p) {
        float v = strtof(p, &end);
        if (end == p) {
            p++;
        } else {
            if (len == cap) {
                cap *= 2;
                arr = realloc(arr, cap * sizeof *arr);
                if (!arr) { perror("realloc"); exit(1); }
            }
            arr[len++] = v;
            p = end;
        }
    }

    *out_count = len;
    return arr;
}

int* parse_ints_strtol(const char* buf, int* out_count)
{
    size_t cap = 16, len = 0;
    int* arr = malloc(cap * sizeof *arr);
    if (!arr) { perror("malloc"); exit(1); }

    char* end;
    const char* p = buf;

    while (*p)
    {
        errno = 0;                       /* clear for each attempt */
        long v = strtol(p, &end, 10);    /* base-10 by default */

        if (end == p)                  /* no digits consumed ⇒ skip char */
        {
            ++p;
        }
        else
        {
            if ((v > INT_MAX) || (v < INT_MIN) || errno == ERANGE)
            {
                fprintf(stderr, "integer out of range: %ld\n", v);
            }
            else
            {
                if (len == cap)
                {
                    cap *= 2;
                    arr = realloc(arr, cap * sizeof *arr);
                    if (!arr) { perror("realloc"); exit(1); }
                }
                arr[len++] = (int)v;
            }
            p = end;
        }
    }

    *out_count = (int)len;
    return arr;
}

void init(TableTransformer *env)
{
    char* word_buf = slurp_file(WORD_PATH);
    env->word_boxes = parse_floats_strtof(word_buf, &env->n_word_boxes);
    free(word_buf);
    env->n_word_boxes /= 4;

    char* cell_buf = slurp_file(CELL_BOXES_PATH);
    env->cell_boxes = parse_floats_strtof(cell_buf, &env->n_cell_boxes);
    free(cell_buf);
    env->n_cell_boxes /= 4;

    char* clusters_buf = slurp_file(CLUSTERS_PATH);
    env->cluster_idx = parse_ints_strtol(clusters_buf, &env->n_clusters);
    free(clusters_buf);

    env->num_clusters = 0;
    env->max_clstr_size = 0;

    for (int i = 0; i < env->n_clusters; ++i) {
        env->num_clusters = max(env->num_clusters, env->cluster_idx[i] + 1);
    }

    int* tmp_arr = calloc(env->num_clusters, sizeof(int));
    for (int i = 0; i < env->n_word_boxes; ++i) {
        tmp_arr[env->cluster_idx[i]]++;
    }

    for (int i = 0; i < env->num_clusters; ++i) {
        env->max_clstr_size = max(env->max_clstr_size, tmp_arr[i]);
    }
    free(tmp_arr);

    env->max_rps = 0;
    env->rps = 0;
    env->n_cut = 0;
    env->n_clustered = 0;

    env->state_pos = (float*)calloc(env->n_cell_boxes * 4, sizeof(float));

    env->returns = (float*)calloc(1, sizeof(float));


    env->row_y1 = malloc(env->n_cell_boxes * sizeof(float));
    env->row_y2 = malloc(env->n_cell_boxes * sizeof(float));

    env->spans = malloc((env->n_cell_boxes + 1) * sizeof(Span));

    env->cluster_comp = malloc(env->num_clusters * env->max_clstr_size * sizeof(int));
    // memset(env->cluster_comp, -1, env->num_clusters * env->max_clstr_size * sizeof(int));
    
    env->cluster_comp_write_head = calloc(env->num_clusters, sizeof(int));
    env->cell_freq_map = calloc(env->n_cell_boxes, sizeof(int));
    env->cell_max_freq_map = calloc(env->n_cell_boxes, sizeof(int));
}

// for c/c++ testing
void allocate(TableTransformer *env)
{
    init(env);

    env->observations = (float*)calloc(4 * env->n_cell_boxes, sizeof(float));
    env->actions = (char*)calloc(env->n_cell_boxes + 1, sizeof(char));
    env->rewards = (float*)calloc(1, sizeof(float));
    env->terminals = (unsigned char *)calloc(1, sizeof(unsigned char));
}

void free_initialized(TableTransformer *env)
{
    free(env->state_pos);
    free(env->cell_boxes);
    free(env->word_boxes);
    free(env->cluster_idx);
    free(env->cluster_comp);
    free(env->cluster_comp_write_head);
    free(env->cell_freq_map);
    free(env->cell_max_freq_map);
    free(env->spans);
    free(env->returns);
    free(env->row_y1); 
    free(env->row_y2);

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

static inline float cut_reward_linear(TableTransformer *env)
{
    int bad = 0;
    int ir = 0;
    const int nW = env->n_word_boxes;

    for (int w = 0; w < nW; ++w) {
        float y1 = env->word_boxes[4*w+1];
        float y2 = env->word_boxes[4*w+3];
        float x1 = env->word_boxes[4*w+0];
        float x2 = env->word_boxes[4*w+2];

        ir = 0; 

        while ((ir < env->n_row) && (env->row_y2[ir] <= y1)) ++ir;


        int row_cut = 0;

        while ((ir < env->n_row) && (((env->row_y2[ir] < y2) && (env->row_y2[ir] > y1)) || ((env->row_y1[ir] < y2) && (env->row_y1[ir] > y1)))) {
            row_cut++;
            ir++;
        }

        bad += row_cut;
        env->n_cut += row_cut;
        env->rps -= row_cut;
    }
    
    return env->r_row_const_1 - env->r_cut_const_1 * bad;
}

static inline float cluster_reward(TableTransformer* env)
{
    const int nW = env->n_word_boxes;
    int ir = 0;

    memset(env->cluster_comp_write_head, 0, env->num_clusters * sizeof(int));
    memset(env->cell_freq_map, 0, env->n_cell_boxes * sizeof(int));
    memset(env->cell_max_freq_map, 0, env->n_cell_boxes * sizeof(int));
    //memset(env->cluster_comp, -1, env->num_clusters * env->max_clstr_size * sizeof(int));


    for (int i = 0; i < nW; ++i)
    {
        float y1 = env->word_boxes[4 * i + 1];
        float y2 = env->word_boxes[4 * i + 3];
        float x1 = env->word_boxes[4 * i + 0];
        float x2 = env->word_boxes[4 * i + 2];

        int cluster_id = env->cluster_idx[i];

        ir = 0; 

        while ((ir < env->n_row) && (env->row_y2[ir] <= y1)) ++ir;

        if ((ir < env->n_row) && (env->row_y2[ir] >= y2) && (env->row_y1[ir] <= y1))
        {
            env->cluster_comp[env->max_clstr_size * cluster_id + env->cluster_comp_write_head[cluster_id]] = ir;
            env->cluster_comp_write_head[cluster_id]++;
        }
    }

    int cluster_rew = 0;

    int max_freq_idx = 0;
    int max_freq = 0;
    for (int i = 0; i < env->num_clusters; i++)
    {
        memset(env->cell_freq_map, 0, env->n_cell_boxes * sizeof(int));
        max_freq_idx = 0;
        max_freq = 0;

        for (int j = 0; j < env->cluster_comp_write_head[i]; j++)
        {
            int cell_idx = env->cluster_comp[i * env->max_clstr_size + j];
            env->cell_freq_map[cell_idx]++;

            if (env->cell_freq_map[cell_idx] > max_freq)
            {
                max_freq = env->cell_freq_map[cell_idx];
                max_freq_idx = cell_idx;
            }
        }

        if (max_freq > env->cell_max_freq_map[max_freq_idx])
        {
            env->cell_max_freq_map[max_freq_idx] = max_freq;
        }    
    }

    for (int i = 0; i < env->n_cell_boxes; i++)
    {
        if (env->cell_max_freq_map[i] > 0)
        {
            cluster_rew += env->cell_max_freq_map[i];
        }
    }

    env->rps += cluster_rew;

    env->max_rps = nW;

    env->n_clustered = cluster_rew;

    return env->r_text_const_1 * cluster_rew;
}

static inline float compute_reward(TableTransformer *env)
{
    env->n_cut = 0;
    env->rps = 0;
    env->max_rps = 0;
    env->n_clustered = 0;

    
    precomp(env);

    float r_comp = cut_reward_linear(env) + cluster_reward(env);
    return r_comp;
}

void compute_observations(TableTransformer *env)
{
    memcpy(env->observations, env->state_pos, env->n_cell_boxes * 4 * sizeof(float));
}

// Required function
void c_reset(TableTransformer *env)
{
    env->tick = 0;
    env->rps = 0;
    env->max_rps = 0;
    env->n_cut = 0;
    env->n_clustered = 0;


    memcpy(env->state_pos, env->cell_boxes, env->n_cell_boxes * 4 * sizeof(float));

    env->start_flag = 1;

    env->returns[0] = 0.0f;

    compute_observations(env);
}

// Required function TODO
void c_step(TableTransformer *env)
{
    env->tick += 1;

    env->rewards[0] = 0.0f;

    env->rewards[0] += -1 * env->step_penalty;
    env->returns[0] += -1 * env->step_penalty;

    int tmp = -1;

    for (int i = 0; i < env->n_cell_boxes; ++i)
    {

        tmp = env->actions[i] == 0 ? -env->d_position : env->d_position;

        if ((tmp < 0) && (env->state_pos[4 * i + 3] + tmp < env->state_pos[4 * i + 1]))
            continue;

        if ((tmp > 0) && (i < env->n_cell_boxes - 1))
        {
            if (env->state_pos[4 * i + 5]  + tmp > env->state_pos[4 * i + 7])
                continue;
        }

        env->state_pos[4 * i + 3] += tmp;
        if (i < env->n_cell_boxes - 1)
            env->state_pos[4 * i + 5] += tmp;

        env->state_pos[4 * i + 3] = fmaxf(0.0f, fminf(env->state_pos[4 * i + 3], (float)env->img_height));
        env->state_pos[4 * i + 5] = fmaxf(0.0f, fminf(env->state_pos[4 * i + 5], (float)env->img_height));
    }


    
    float r_comp = env->stable_r_coeff * compute_reward(env);
    env->rewards[0] += r_comp;
    env->returns[0] += r_comp;

    if (env->tick <= env->min_steps)
        env->terminals[0] = 0;
    else if (env->tick > env->max_steps)
        env->terminals[0] = 1;
    else
        env->terminals[0] = env->actions[env->n_cell_boxes] == 1 ? 1 : 0;

    if (env->terminals[0])
    {
        float r_comp = env->done_r_coeff * compute_reward(env);

        env->rewards[0] += r_comp;
        env->returns[0] += r_comp;

        env->log.perf += (env->max_rps > 0) ? (float)env->rps / env->max_rps : 0.0f;
        env->log.score += env->returns[0];
        env->log.episode_return += env->returns[0];
        env->log.episode_length += (float)env->tick;
        env->log.n_cut += (float)env->n_cut;
        env->log.n_clustered += (float)env->n_clustered;
        env->log.max_rps += (float)env->max_rps;
        env->log.rps += (float)env->rps;
        env->log.n++;

        c_reset(env);

    }

    compute_observations(env);

    if (env->tick > 1)
        env->start_flag = 0;

}

// Required function. Should handle creating the client on first call Not required here
void c_render(TableTransformer *env)
{
    /* 0. ------------------------------------------------------------------ */
    if (env->client == NULL) {
        SetConfigFlags(FLAG_WINDOW_RESIZABLE | FLAG_VSYNC_HINT);
        int winW = GetMonitorWidth(0) / 2;
        int winH = GetMonitorHeight(0) / 2;
        InitWindow(winW, winH,
                   "Surya overlay (rows = blue, words = red)");
        SetWindowMinSize(400, 300);

        env->client           = calloc(1, sizeof(Client));
        env->client->table    = LoadTexture(
            "/home/anshult/Documents/Ocean/pufferlib/"
            "pufferlib/resources/table_transformer/table_p1.png");

        env->client->zoom     = 1.0f;
        env->client->min_zoom = 0.20f;
        env->client->max_zoom = 5.00f;
        env->client->offset   = (Vector2){ 0, 0 };
    }

    /* 1. ------------------------------------------------------------------ */
    if (WindowShouldClose()) {
        UnloadTexture(env->client->table);
        CloseWindow();
        exit(0);
    }

    /* 2. ------------------------------------------------------------------ */
    float base = fminf((float)GetScreenWidth()  / env->img_width,
                       (float)GetScreenHeight() / env->img_height);

    /* 3. ------------------------------------------------------------------ */
    float wheel = GetMouseWheelMove();
    if (wheel == 0) {
        if (IsKeyPressed(KEY_EQUAL)  || IsKeyPressed(KEY_KP_ADD))      wheel = +1;
        if (IsKeyPressed(KEY_MINUS)  || IsKeyPressed(KEY_KP_SUBTRACT)) wheel = -1;
    }
    if (wheel != 0) {
        Vector2 mouse = GetMousePosition();
        Vector2 img   = {
            (mouse.x - env->client->offset.x) / (base * env->client->zoom),
            (mouse.y - env->client->offset.y) / (base * env->client->zoom)
        };

        env->client->zoom *= 1.0f + wheel * 0.10f;
        env->client->zoom  = Clamp(env->client->zoom,
                                   env->client->min_zoom,
                                   env->client->max_zoom);

        env->client->offset.x = mouse.x - img.x * base * env->client->zoom;
        env->client->offset.y = mouse.y - img.y * base * env->client->zoom;
    }

    /* 4. ------------------------------------------------------------------ */
    static bool    dragging = false;
    static Vector2 dragOrig = { 0 };

    if (IsMouseButtonPressed(MOUSE_BUTTON_MIDDLE)) {
        dragging  = true;
        dragOrig  = GetMousePosition();
    }
    if (IsMouseButtonReleased(MOUSE_BUTTON_MIDDLE)) dragging = false;

    if (dragging) {
        Vector2 now   = GetMousePosition();
        Vector2 delta = Vector2Subtract(now, dragOrig);
        env->client->offset = Vector2Add(env->client->offset, delta);
        dragOrig = now;
    }

    /* 5. ------------------------------------------------------------------ */
    float w = env->img_width  * base * env->client->zoom;
    float h = env->img_height * base * env->client->zoom;
    if (w < GetScreenWidth())  env->client->offset.x = (GetScreenWidth()  - w) / 2.0f;
    if (h < GetScreenHeight()) env->client->offset.y = (GetScreenHeight() - h) / 2.0f;

    /* 6.  draw **only** at episode end ------------------------------------ */
    if (!env->terminals[0]) {
        BeginDrawing();
        ClearBackground((Color){ 6, 24, 24, 255 });

        DrawTextureEx(env->client->table,
                      env->client->offset,
                      0.0f,
                      base * env->client->zoom,
                      WHITE);

        for (int i = 0; i < env->n_word_boxes; ++i) {
            Rectangle r = scale_rect(&env->word_boxes[4 * i],
                                     base * env->client->zoom,
                                     (int)env->client->offset.x,
                                     (int)env->client->offset.y);
            DrawRectangleLinesEx(r, 1, GREEN);
        }
        for (int i = 0; i < env->n_cell_boxes; ++i) {
            Rectangle r = scale_rect(&env->state_pos[4 * i],
                                     base * env->client->zoom,
                                     (int)env->client->offset.x,
                                     (int)env->client->offset.y);
            //DrawRectangleRec    (r, (Color){ 0, 0, 255, 80 });
            // if (env->terminals[0])
            //     DrawRectangleLinesEx(r, 2, BLUE);
            // else if (env->start_flag)
            DrawRectangleLinesEx(r, 2, RED);
            // else
            //     DrawRectangleLinesEx(r, 2, GREEN);
        }

        EndDrawing();

        // printf("%d\n", env->tick);

        // printf("row1: (%f %f) (%f %f)\n", env->state_pos[0],
        //        env->state_pos[1], env->state_pos[2], env->state_pos[3]);

        // printf("row2: (%f %f) (%f %f)\n", env->state_pos[4],
        //        env->state_pos[5], env->state_pos[6], env->state_pos[7]);

        // printf("row3: (%f %f) (%f %f)\n", env->state_pos[8],
        //        env->state_pos[9], env->state_pos[10], env->state_pos[11]);

        // printf("row4: (%f %f) (%f %f)\n", env->state_pos[12],
        //        env->state_pos[13], env->state_pos[14], env->state_pos[15]);

        char fname[64];
        snprintf(fname, sizeof fname, "render++_%06d.png", env->tick);
        TakeScreenshot(fname);
    }
    else{
        printf("Episode ended: %d steps, %d cuts, rps = %d/%d\n",
               env->tick, env->n_cut, env->rps, env->max_rps);
        printf("Perf: %.3f, Score: %.3f, Return: %.3f, Length: %.3f\n",
               env->log.perf, env->log.score,
               env->log.episode_return, env->log.episode_length);
    }
}

// Required function. Should clean up anything you allocated
// Do not free env->observations, actions, rewards, terminals
void c_close(TableTransformer *env)
{
    if (env->client != NULL) {
        UnloadTexture(env->client->table);
        CloseWindow();
        free(env->client);
    }

    free_initialized(env);
}