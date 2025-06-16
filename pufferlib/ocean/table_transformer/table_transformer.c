/* Table Transformer Environment
 * use this as an environment for Improving Table Cell/Column/Row bbox detection
 * -- approaches
 * -- -- Manual reward signal (very difficult as it seems intractable)
 * -- -- Use a VLM to give rewars (implementation heavy)
 *
 * This environment does the first approach
 */

#include "table_transformer.h"

int main()
{
    TableTransformer env = {
        .img_channels = 4,
        .img_height = 2600,
        .img_width = 3300,

        .num_action_boxes = 125,

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
    while (i < 40)
    {

        c_step(&env);
        // c_render(&env);
        i++;
    }

    free_allocated(&env);
}