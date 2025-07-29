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
        .d_position = 5.0f,

        .img_height = 1200,
        .img_width = 2200,

        .r_text_const_1 = 0.4f,

        .min_steps = 10,
        .max_steps = 1000,

        .stable_r_coeff = 0.1f,
        .done_r_coeff = 1.0f,

        .epsilon_cell = 5.0f,
        .epsilon_del = 10.0f
    };

    allocate(&env);

    c_reset(&env);
    // c_render(&env);

    int i = 0;
    int start = time(NULL);
    int num_steps = 0;
    
    while (i < 400000)
    {
        for (int j = 0; j < 2 * env.n_cell_boxes; ++j)
        {
            env.actions[j] = rand() % 3;
        }

        env.actions[2 * env.n_cell_boxes] = rand() % 2;

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