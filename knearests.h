// Sylvain Lefebvre 2017-10-04
#pragma once

struct kn_problem;

kn_problem   *kn_prepare(float *points, int numpoints);
void          kn_solve(kn_problem *kn);
void          kn_free(kn_problem **kn);

float        *kn_get_points(kn_problem *kn);
unsigned int *kn_get_knearests(kn_problem *kn);
unsigned int *kn_get_permutation(kn_problem *kn);

void kn_print_stats(kn_problem *kn);

