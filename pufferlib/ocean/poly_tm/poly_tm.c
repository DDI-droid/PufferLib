#include "poly_tm.h"


void test_performance(int timeout) {
    PolyTM env = {
        .work_tape_size = 256,
        .state_tape_size = 256,
        .result_tape_size = 256,

        .work_observation_window = 8,
        .state_observation_window = 8,
        .result_observation_window = 8,

        .tape_alphabet = 10,
        .move_head = 10,

        .num_work_heads = 1,
        .num_state_heads = 1,
        .num_result_heads = 1,

        .nen_halt_penalty = 5.0f,
        .invalid_output_penalty = 10.0f,
        .correctness_reward = 10.0f,
        .incorrectness_penalty = 5.0f,

        .max_steps = 1000,

        .problem_size = 10,

    };

    allocate(&env);
    c_reset(&env);

    int start = time(NULL);
    int num_steps = 0;
    while (time(NULL) - start < timeout) {
        
        env.actions[0] = 0;

        c_step(&env);
        num_steps++;
    }

    int end = time(NULL);
    float sps = num_steps / (end - start);
    printf("Test Environment SPS: %f\n", sps);
    free_allocated(&env);
}

int main()
{
    test_performance(10);
}