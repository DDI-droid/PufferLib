/* Table OCR Environment
 * use this as an environment for Improving Table Cell/Column/Row bbox detection
 * -- approaches
 * -- -- Manual reward signal (very difficult as it seems intractable)
 * -- -- Use a VLM to give rewars (implementation heavy)
 *
 * This environment does the first approach
 */

#include "table_ocr.h"
#include <time.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

int main()
{
    TableOCR env = {
        .img_channels = 4,
        .img_height = 1200,
        .img_width = 2200,

        .step_penalty = 0.01f,
        .d_position = 0.01f,

        .iou_inside = 0.4f,
        .iou_cell = 0.4f,
        .iou_row = 0.4f,
        .iou_col = 0.4f,

        .r_text_const_1 = 0.4f,
        .r_cut_const_1 = 0.4f,
        .r_cell_const_1 = 0.4f,
        .r_row_const_1 = 0.4f,
        .r_col_const_1 = 0.4f,
    };

    allocate(&env);

    c_reset(&env);
    // c_render(&env);

    int i = 0;
    int start = time(NULL);
    int num_steps = 0;
    
    while (i < 400000)
    {

        // Simulate some actions
        for (int j = 0; j < 4 * env.n_cell_boxes + env.n_cell_boxes + 1; ++j)
        {
            env.actions[j] = rand() % 2; // Random actions for testing
        }

        c_step(&env);
        // c_render(&env);
        i++;

        num_steps++;
    }

    int end = time(NULL);
    float sps = num_steps / (float)(end - start);
    printf("Test Environment SPS: %f\n", sps);

    free_allocated(&env);
}