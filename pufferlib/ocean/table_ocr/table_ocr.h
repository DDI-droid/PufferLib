/* Table OCR Environment
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

const char* WORD_PATH = "/media/dpa/data/Anshul/Ocean_ocr/helper/prepped/JPMCC 2016-JP2_Camelback Crossing_20231231_p1_words.txt";
const char* CELL_BOXES_PATH = "/media/dpa/data/Anshul/Ocean_ocr/helper/prepped/JPMCC 2016-JP2_Camelback Crossing_20231231_p1_rows.txt";
const char* CLUSTERS_PATH = "/media/dpa/data/Anshul/Ocean_ocr/helper/prepped/JPMCC 2016-JP2_Camelback Crossing_20231231_p1_clusters.txt";

#define max(a, b) ((a) > (b) ? (a) : (b))
#define min(a, b) ((a) < (b) ? (a) : (b))

static inline Rectangle scale_rect(const float *b, float image_width, float scale, int off_x, int off_y)
{
    return (Rectangle){
        off_x,
        b[0] * scale + off_y,
        6500 * scale,
        (b[1] - b[0]) * scale
    };
}

typedef struct Log Log;
struct Log
{
    float perf;
    float score;
    float episode_return;
    float episode_length;
    float n_deleted;
    float n_clustered;
    float init_clustered;
    float n; // Required as the last field
};

typedef struct { float y1, y2; } Span;

typedef struct Client Client;
struct Client
{
    Texture2D table;
    float zoom;
    float min_zoom;
    float max_zoom;
    Vector2 offset;
};

typedef struct TableOCR TableOCR;
struct TableOCR
{
    Log log;
    float* observations;
    int* actions;
    float* rewards;
    float* returns;
    unsigned char* terminals;

    int img_width;
    int img_height;
    
    float d_position;

    float stable_r_coeff;

    int min_steps;
    int max_steps;

    float epsilon_cell;
    float epsilon_del;

    float* cell_boxes;
    float* word_boxes;
    int* cluster_idx;
    int n_cell_boxes;
    int n_word_boxes;
    int n_clusters;

    int num_clusters;
    int max_clstr_size;

    int tick;

    int max_rps;
    int rps;
    int n_clustered;
    int init_clustered;

    Span* spans;

    float* state_pos;

    int* cluster_comp;
    int* cluster_comp_write_head;

    int* cell_freq_map;
    int* cell_max_freq_map;
    int* cell_max_written;

    int* cell_next;
    int* cell_prev;
    int cell_start;

    int* cell_deleted;

    int num_deleted;

    Client* client;    
};

// function declarations
void c_close(TableOCR*);

static int cmp_span_by_top(const void* a, const void* b)
{
    const Span* span1 = (const Span*)a;
    const Span* span2 = (const Span*)b;
    float diff = span1->y1 - span2->y1;
    return (diff > 0.f) - (diff < 0.f);
}

static void precomp(TableOCR* env)
{
    const int n_cells = env->n_cell_boxes;
    int n_spans = 0;

    for (int i = 0; i < n_cells; ++i) {
        float y1 = env->state_pos[2 * i];
        float y2 = env->state_pos[2 * i + 1];
        if (y2 < y1) { float tmp = y1; y1 = y2; y2 = tmp; }
        env->spans[n_spans++] = (Span){ y1, y2 };
    }

    qsort(env->spans, n_spans, sizeof(Span), cmp_span_by_top);


    for (int i = 0; i < n_spans; ++i) {
        env->state_pos[2 * i] = env->spans[i].y1;
        env->state_pos[2 * i + 1] = env->spans[i].y2;
    }

    for (int i = 1; i < 2 * n_spans; ++i)
    {
        env->state_pos[i] = max(env->state_pos[i], env->state_pos[i-1]);
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

void init(TableOCR* env)
{
    char* word_buf = slurp_file(WORD_PATH);
    env->word_boxes = parse_floats_strtof(word_buf, &env->n_word_boxes);
    free(word_buf);
    env->n_word_boxes /= 2;

    char* cell_buf = slurp_file(CELL_BOXES_PATH);
    env->cell_boxes = parse_floats_strtof(cell_buf, &env->n_cell_boxes);
    free(cell_buf);
    env->n_cell_boxes /= 2;

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
    env->n_clustered = 0;
    env->init_clustered = 0;
    env->num_deleted = 0;

    env->state_pos = (float*)calloc(env->n_cell_boxes * 2, sizeof(float));

    env->returns = (float*)calloc(1, sizeof(float));

    env->spans = calloc(env->n_cell_boxes, sizeof(Span));

    env->cluster_comp = calloc(env->num_clusters * env->max_clstr_size, sizeof(int));
    
    env->cluster_comp_write_head = calloc(env->num_clusters, sizeof(int));
    env->cell_freq_map = calloc(env->n_cell_boxes, sizeof(int));
    env->cell_max_freq_map = calloc(env->n_cell_boxes, sizeof(int));
    env->cell_max_written = calloc(env->n_cell_boxes, sizeof(int));

    env->cell_next = calloc(env->n_cell_boxes, sizeof(int));
    env->cell_prev = calloc(env->n_cell_boxes, sizeof(int));

    env->cell_deleted = calloc(env->n_cell_boxes, sizeof(int));
}

// for c/c++ testing
void allocate(TableOCR* env)
{
    init(env);

    env->observations = (float*)calloc(2 * env->n_cell_boxes, sizeof(float));
    env->actions = (int*)calloc(2 * env->n_cell_boxes + 1, sizeof(int));
    env->rewards = (float*)calloc(1, sizeof(float));
    env->terminals = (unsigned char *)calloc(1, sizeof(unsigned char));
}

void free_initialized(TableOCR* env)
{
    free(env->state_pos);
    free(env->cell_boxes);
    free(env->word_boxes);
    free(env->cluster_idx);
    free(env->cluster_comp);
    free(env->cluster_comp_write_head);
    free(env->cell_freq_map);
    free(env->cell_max_freq_map);
    free(env->cell_next);
    free(env->cell_prev);
    free(env->spans);
    free(env->returns);
    free(env->cell_deleted);
}

void free_allocated(TableOCR* env)
{
    free(env->observations);
    free(env->actions);
    free(env->rewards);
    free(env->terminals);

    c_close(env);
}

static inline float cluster_reward(TableOCR* env)
{
    const int nW = env->n_word_boxes;
    int ir = 0;

    memset(env->cluster_comp_write_head, 0, env->num_clusters * sizeof(int));
    memset(env->cell_freq_map, 0, env->n_cell_boxes * sizeof(int));
    memset(env->cell_max_freq_map, 0, env->n_cell_boxes * sizeof(int));
    memset(env->cell_max_written, 0, env->n_cell_boxes * sizeof(int));
    memset(env->cluster_comp, -1, env->num_clusters * env->max_clstr_size * sizeof(int));


    for (int i = 0; i < nW; ++i)
    {
        float y1 = env->word_boxes[2 * i];
        float y2 = env->word_boxes[2 * i + 1];

        int cluster_id = env->cluster_idx[i];

        ir = env->cell_start;

        if (ir == -1)
        {
            printf("cell start is -1!!\n");
        }

        while ((ir != -1) && (env->state_pos[2 * ir + 1] <= y1)) // leq makes sense here
        {
            ir = env->cell_next[ir];
        }

        while ((ir != -1) && (env->state_pos[2 * ir] < y2)) // less than makes sense here
        {
            if ((env->state_pos[2 * ir] <= y1 + env->epsilon_cell) && (env->state_pos[2 * ir + 1] >= y2 - env->epsilon_cell) && (env->cell_deleted[ir] == 0))
            {
                env->cluster_comp[env->max_clstr_size * cluster_id + env->cluster_comp_write_head[cluster_id]] = ir;
                env->cluster_comp_write_head[cluster_id]++;
                break;
            }
            ir = env->cell_next[ir];
        }
    }

    int cluster_rew = 0;

    int max_freq = 0;
    int max_freq_idx = -1;

    for (int i = 0; i < env->num_clusters; i++)
    {
        memset(env->cell_freq_map, 0, env->n_cell_boxes * sizeof(int));
        max_freq = 0;
        max_freq_idx = -1;

        for (int j = 0; j < env->cluster_comp_write_head[i]; j++)
        {
            int cell_idx = env->cluster_comp[i * env->max_clstr_size + j];
            
            if (cell_idx < 0 || cell_idx >= env->n_cell_boxes || env->cell_deleted[cell_idx] == 1)
            {
                fprintf(stderr, "Invalid cell index %d for cluster %d\n", cell_idx, i);
                continue;
            }
            
            env->cell_freq_map[cell_idx]++;
        }

        int cnt = 0;
        for (int j = 0; j < env->n_cell_boxes; ++j)
        {
            if (env->cell_freq_map[j] > 0)
            {
                cnt++;
                max_freq_idx = j;
                max_freq = env->cell_freq_map[j];
            }
        }

        if (cnt == 1)
        {
            if (env->cell_max_written[max_freq_idx] == 0)
            {
                env->cell_max_freq_map[max_freq_idx] = max_freq;
                env->cell_max_written[max_freq_idx] = 1;
            }
            else
            {
                env->cell_max_freq_map[max_freq_idx] = 0;
            }
        }
        else if (cnt == 0)
        {
            continue;
        }        
        else
        {
            for (int j = 0; j < env->n_cell_boxes; ++j)
            {
                if (env->cell_freq_map[j] > 0)
                {
                    if (env->cell_max_written[j] == 0)
                    {
                        env->cell_max_written[j] = 1;
                    }
                    else
                    {
                        env->cell_max_freq_map[j] = 0;
                        env->cell_max_written[j] = 1;
                    }
                }
            }
        }

        
    }

    for (int i = 0; i < env->n_cell_boxes; ++i)
    {
        cluster_rew += env->cell_max_freq_map[i];
    }

    env->rps = cluster_rew;

    env->n_clustered = cluster_rew;

    env->max_rps = nW;

    return (float)cluster_rew / nW;
}

static inline float compute_reward(TableOCR* env)
{
    env->rps = 0;
    env->max_rps = 0;
    env->n_clustered = 0;

    float r_comp = cluster_reward(env);
    return r_comp;
}

void compute_observations(TableOCR* env)
{
    memcpy(env->observations, env->state_pos, env->n_cell_boxes * 2 * sizeof(float));
}

void c_reset(TableOCR* env)
{
    env->tick = 0;

    env->rps = 0;
    env->max_rps = 0;
    env->n_clustered = 0;
    env->num_deleted = 0;

    env->cell_start = 0;


    memcpy(env->state_pos, env->cell_boxes, env->n_cell_boxes * 2 * sizeof(float));

    for (int i = 0; i < env->n_cell_boxes; ++i)
    {
        if (i > 0)
        {
            env->cell_prev[i] = i - 1;
        }
        else{
            env->cell_prev[i] = -1;
        }

        if (i < env->n_cell_boxes - 1)
        {
            env->cell_next[i] = i + 1;
        }
        else
        {
            env->cell_next[i] = -1;
        }
    }

    memset(env->cell_deleted, 0, env->n_cell_boxes * sizeof(int));    

    env->returns[0] = 0.0f;

    precomp(env);

    compute_reward(env);
    env->init_clustered = env->n_clustered;

    compute_observations(env);
}

void c_step(TableOCR* env)
{
    env->tick += 1;

    env->rewards[0] = 0.0f;

    for (int i = 0; i < 2 * env->n_cell_boxes; ++i)
    {
        if (env->cell_deleted[i / 2])
            continue;

        if (env->actions[i] == 0)
            continue;
        
        else if (env->actions[i] == 1)
        {
            if (i % 2 == 0)
            {
                if (env->cell_prev[i / 2] == -1)
                {
                    env->state_pos[i] = max(env->state_pos[i] - env->d_position, 0.0f);
                }
                else
                {
                    env->state_pos[i] = max(env->state_pos[i] - env->d_position, env->state_pos[2 * env->cell_prev[i / 2] + 1]);
                }
            }
            else
            {
                env->state_pos[i] = max(env->state_pos[i] - env->d_position, env->state_pos[i-1]);
            }
        }
        else if (env->actions[i] == 2)
        {
            if (i % 2 == 0)
            {
                env->state_pos[i] = min(env->state_pos[i] + env->d_position, env->state_pos[i+1]);
            }
            else
            {
                if (env->cell_next[i / 2] == -1)
                {
                    env->state_pos[i] = min(env->state_pos[i] + env->d_position, (float)env->img_height);
                }
                else
                {
                    env->state_pos[i] = min(env->state_pos[i] + env->d_position, env->state_pos[2 * env->cell_next[i / 2]]);
                }
            }
        }
        else
            printf("Invalid action %d for cell %d\n", env->actions[i], i);
    }

    for (int i = 0; i < env->n_cell_boxes; ++i)
    {
        if ((env->cell_deleted[i] == 0) && (env->state_pos[2 * i + 1] - env->state_pos[2 * i] < env->epsilon_del))
        {
            //row deletion
            env->cell_deleted[i] = 1;
            env->num_deleted++;

            env->state_pos[2 * i] = -1.0f;
            env->state_pos[2 * i + 1] = -1.0f;

            if (env->cell_next[i] != -1)
                env->cell_prev[env->cell_next[i]] = env->cell_prev[i];

            if (env->cell_prev[i] != -1)
                env->cell_next[env->cell_prev[i]] = env->cell_next[i];

            if (env->cell_start == i)
                env->cell_start = env->cell_next[i];
        }
    }
    
    /// per-step reward
    // env->stable_r_coeff * 

    if (env->tick <= env->min_steps)
        env->terminals[0] = 0;
    else if (env->tick > env->max_steps)
        env->terminals[0] = 1;
    else
        env->terminals[0] = env->actions[2 * env->n_cell_boxes] == 1 ? 1 : 0;

    if (env->terminals[0])
    {
        float r_comp = compute_reward(env);

        env->rewards[0] += r_comp;
        env->returns[0] += r_comp;

        env->log.perf += (env->max_rps > 0) ? (float)env->rps / env->max_rps : 0.0f;
        env->log.score += env->returns[0];
        env->log.episode_return += env->returns[0];
        env->log.episode_length += (float)env->tick;
        env->log.n_deleted += (float)env->num_deleted;
        env->log.n_clustered += (float)env->n_clustered;
        env->log.init_clustered += (float)env->init_clustered;
        env->log.n++;

        c_reset(env);

    }

    compute_observations(env);
}

// Required function. Should handle creating the client on first call Not required here
void c_render(TableOCR *env)
{
    /* 0. ------------------------------------------------------------------ */
    if (env->client == NULL) {
        SetConfigFlags(FLAG_WINDOW_RESIZABLE | FLAG_VSYNC_HINT);
        int winW = GetMonitorWidth(0) / 2;
        int winH = GetMonitorHeight(0) / 2;

        InitWindow(winW, winH,
                   "Table OCR (rows = blue, words = red)");
        SetWindowMinSize(400, 300);

        env->client           = calloc(1, sizeof(Client));
        env->client->table    = LoadTexture(
            "/media/dpa/data/Anshul/Ocean_ocr/helper/prepped/JPMCC 2016-JP2_Camelback Crossing_20231231_p1.png"
        );

        env->client->zoom     = 0.3f;
        env->client->min_zoom = 0.20f;
        env->client->max_zoom = 5.00f;
        env->client->offset   = (Vector2){ 100, 100 };
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
    BeginDrawing();
    ClearBackground((Color){ 6, 24, 24, 255 });

    DrawTextureEx(env->client->table,
                    env->client->offset,
                    0.0f,
                    base * env->client->zoom,
                    WHITE);

    for (int i = 0; i < env->n_cell_boxes; ++i) {
        Rectangle r = scale_rect(&env->state_pos[2 * i],
                                    (float)env->img_width,
                                    base * env->client->zoom,
                                    (int)env->client->offset.x,
                                    (int)env->client->offset.y);

        DrawRectangleLinesEx(r, 2, BLUE);

    }

    EndDrawing();

    char fname[64];
    snprintf(fname, sizeof fname, "render++_%06d.png", env->tick);
    TakeScreenshot(fname);

    if (env->terminals[0]) {
        printf("Episode ended: %d steps, rps = %d/%d\n",
               env->tick, env->rps, env->max_rps);
        printf("Perf: %.3f, Score: %.3f, Return: %.3f, Length: %.3f\n",
               env->log.perf, env->log.score,
               env->log.episode_return, env->log.episode_length);
    }
}

// Required function. Should clean up anything you allocated
// Do not free env->observations, actions, rewards, terminals
void c_close(TableOCR *env)
{
    if (env->client != NULL) {
        UnloadTexture(env->client->table);
        CloseWindow();
        free(env->client);
    }

    free_initialized(env);
}