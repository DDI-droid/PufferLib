#include "poly_tm.h"


void test_performance(int timeout) {
    PolyTM env = {
        .nen_halt_penalty = 5.0f,
        .correctness_reward = 10.0f,
        .incorrectness_penalty = 5.0f,

        .tape_size = 512,
        .observation_window = 16,
        .work_alphabet = 256,
        .state_alphabet = 256,
        .move_state = 10,
        .move_work = 10,
        
        .max_a = 4,
        .max_i = 50,
        .max_u = 50,

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