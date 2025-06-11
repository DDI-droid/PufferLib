#include "poly_time.h"


void test_performance(int timeout) {
    PolyTime env = {
        .nen_halt_penalty = 5.0f,
        .correctness_reward = 10.0f,
        .incorrectness_penalty = 5.0f,

        .max_agents=500,
        .max_items=500,
        .tape_size=1024,
        .max_utility=1000,
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