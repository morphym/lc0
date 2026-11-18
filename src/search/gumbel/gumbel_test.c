/* SPDX-License-Identifier: GPL-3.0-or-later */
#include "search/gumbel/gumbel.h"
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#define CHECK(x) do { if (!(x)) { fprintf(stderr, "%s:%d: %s\n", __FILE__, __LINE__, #x); exit(1); } } while (0)
typedef struct {
  int depth, action, count, fail_depth, branching;
  int interior_visits[2];
  float prior[GAZ_MAX_ACTIONS], rewards[GAZ_MAX_ACTIONS];
  float draw;
} game;
static int evaluate(void* ctx, float* value, float* draw, float* p) {
  game* g = ctx;
  *draw = g->draw;
  if (g->fail_depth && g->depth == g->fail_depth) return -1;
  if (g->depth) {
    *value = g->depth % 2 ? -g->rewards[g->action] : g->rewards[g->action];
    if (!g->branching || g->depth == 5) return 0;
    p[0] = 0.7f; p[1] = 0.3f;
    return 2;
  }
  *value = 0;
  memcpy(p, g->prior, (size_t)g->count * sizeof(float));
  return g->count;
}
static int push(void* ctx, int a) {
  game* g = ctx;
  if (!g->depth) g->action = a;
  if (g->depth == 1) ++g->interior_visits[a];
  ++g->depth;
  return 1;
}
static void pop(void* ctx) { --((game*)ctx)->depth; }
static gaz_config config(int n) {
  gaz_config cfg = {n, 16, GAZ_MAX_DEPTH, 50, 0.1, 0, 123};
  return cfg;
}
static gaz_search* create(game* g, gaz_config cfg) {
  return gaz_create(&cfg, (gaz_callbacks){g, evaluate, push, pop});
}
static void finish(gaz_search* s, game* g, int budget) {
  for (int i = 0; i < budget; ++i) { CHECK(gaz_step(s) == 1); CHECK(g->depth == 0); }
  CHECK(gaz_step(s) == 0);
  CHECK(gaz_simulations(s) == budget);
  int sum = 0;
  for (int i = 0; i < g->count; ++i) sum += gaz_action_visits(s, i);
  CHECK(sum == budget);
}
static void test_budgets(void) {
  for (int count = 1; count <= 35; ++count) {
    for (int budget = 0; budget <= 150; ++budget) {
      game g = {.count = count};
      for (int i = 0; i < count; ++i) {
        g.prior[i] = 1.0f / (i + 1);
        g.rewards[i] = (float)i / count;
      }
      gaz_search* s = create(&g, config(budget));
      CHECK(s);
      finish(s, &g, budget);
      CHECK(gaz_best_action(s) >= 0 && gaz_best_action(s) < count);
      float p[GAZ_MAX_ACTIONS];
      CHECK(gaz_policy(s, p, GAZ_MAX_ACTIONS) == count);
      double total = 0;
      for (int i = 0; i < count; ++i) { CHECK(isfinite(p[i]) && p[i] >= 0); total += p[i]; }
      CHECK(fabs(total - 1) < 1e-6);
      gaz_destroy(s);
    }
  }
}
static void test_value_and_policy(void) {
  game g = {.count = 2, .prior = {0.8f, 0.2f}, .rewards = {-1, 1}};
  gaz_search* s = create(&g, config(64));
  CHECK(s);
  finish(s, &g, 64);
  CHECK(gaz_best_action(s) == 1);
  CHECK(gaz_action_value(s, 0) == -1 && gaz_action_value(s, 1) == 1);
  float p[2];
  CHECK(gaz_policy(s, p, 2) == 2);
  CHECK(p[1] > p[0]);
  gaz_destroy(s);
  /* All equal values preserve the network policy, including zero budget. */
  g.rewards[0] = g.rewards[1] = 0;
  s = create(&g, config(0));
  CHECK(gaz_best_action(s) == 0);
  CHECK(gaz_policy(s, p, 2) == 2);
  CHECK(fabs(p[0] - 0.8) < 1e-6);
  gaz_destroy(s);
}
static void test_schedule(void) {
  game g = {.count = 8};
  for (int i = 0; i < 8; ++i) g.prior[i] = 1;
  gaz_config cfg = config(24);
  cfg.max_considered = 4;
  gaz_search* s = create(&g, cfg);
  CHECK(s);
  for (int i = 0; i < 12; ++i) CHECK(gaz_step(s) == 1);
  for (int i = 0; i < 8; ++i) CHECK(gaz_action_visits(s, i) == (i < 4 ? 3 : 0));
  for (int i = 0; i < 12; ++i) CHECK(gaz_step(s) == 1);
  CHECK(gaz_action_visits(s, 0) == 9 && gaz_action_visits(s, 1) == 9);
  CHECK(gaz_action_visits(s, 2) == 3 && gaz_action_visits(s, 3) == 3);
  gaz_destroy(s);
}
static void test_noise(void) {
  game a = {.count = 4, .prior = {1, 2, 3, 4}}, b = a;
  gaz_config cfg = config(30); cfg.noise_scale = 1;
  gaz_search *x = create(&a, cfg), *y = create(&b, cfg);
  CHECK(x && y);
  finish(x, &a, 30); finish(y, &b, 30);
  CHECK(gaz_best_action(x) == gaz_best_action(y));
  for (int i = 0; i < 4; ++i) CHECK(gaz_action_visits(x, i) == gaz_action_visits(y, i));
  gaz_destroy(x); gaz_destroy(y);
  int sampled[4] = {0};
  for (int seed = 0; seed < 1000; ++seed) {
    cfg.seed = (uint64_t)seed; cfg.simulations = 0;
    x = create(&a, cfg); CHECK(x);
    ++sampled[gaz_best_action(x)]; gaz_destroy(x);
  }
  for (int i = 0; i < 4; ++i) CHECK(sampled[i] > 30);
}
static void test_completion_and_interior(void) {
  game g = {.count = 3, .prior = {0.6f, 0.3f, 0.1f}, .rewards = {0.5f, 0, 0}};
  gaz_search* s = create(&g, config(1)); CHECK(s);
  CHECK(gaz_step(s) == 1);
  float p[3]; CHECK(gaz_policy(s, p, 3) == 3);
  /* Mixed unvisited value = (raw 0 + one visit * 0.5) / 2 = 0.25.
     Rescale [0.5, 0.25, 0.25] to [1, 0, 0], then sigma = 51 * 0.1. */
  double expected = 0.6 * exp(5.1) / (0.6 * exp(5.1) + 0.4);
  CHECK(fabs(p[0] - expected) < 1e-6);
  CHECK(fabs(p[1] / p[2] - 3) < 1e-5);
  gaz_destroy(s);
  memset(&g, 0, sizeof(g));
  g.count = 1; g.prior[0] = 1; g.branching = 1;
  gaz_config cfg = config(101); cfg.c_scale = 0;
  s = create(&g, cfg); CHECK(s);
  finish(s, &g, 101);
  CHECK(g.interior_visits[0] == 70 && g.interior_visits[1] == 30);
  gaz_destroy(s);
}
static void test_continuation_and_wdl(void) {
  game g = {.count = 7, .prior = {1, 1, 1, 1, 1, 1, 1},
            .rewards = {0.25f, -0.25f, 0, 0, 0, 0, 0}, .draw = 0.5f};
  gaz_search* s = create(&g, config(3)); CHECK(s);
  CHECK(!gaz_begin_round(s, 5));
  int total = 0;
  for (int round = 0; round < 20; ++round) {
    const int budget = round ? round * 7 + 1 : 3;
    if (round) CHECK(gaz_begin_round(s, budget));
    for (int i = 0; i < budget; ++i) CHECK(gaz_step(s) == 1);
    CHECK(gaz_step(s) == 0 && g.depth == 0);
    total += budget;
    CHECK(gaz_simulations(s) == total);
    int sum = 0;
    for (int i = 0; i < g.count; ++i) {
      sum += gaz_action_visits(s, i);
      CHECK(gaz_action_draw(s, i) == 0.5f);
      CHECK(fabs(gaz_action_prior(s, i) - 1.0 / 7) < 1e-6);
    }
    CHECK(sum == total);
    CHECK(gaz_best_action(s) >= 0);
  }
  int pv[2] = {-1, 999};
  CHECK(gaz_principal_variation(s, 0, pv, 1) == 1);
  CHECK(pv[0] == 0 && pv[1] == 999);
  CHECK(gaz_action_raw_value(s, 0) == 0.25f);
  CHECK(!gaz_begin_round(s, 0));
  gaz_destroy(s);
}
static void test_errors_and_depth(void) {
  game g = {.count = 2, .prior = {0, 0}, .rewards = {1, -1}, .branching = 1};
  gaz_config cfg = config(200); cfg.max_depth = 3;
  gaz_search* s = create(&g, cfg); CHECK(s);
  finish(s, &g, 200);
  CHECK(gaz_depth(s) == 3 && gaz_best_action(s) == 0);
  gaz_destroy(s);
  g.fail_depth = 2;
  s = create(&g, cfg); CHECK(s);
  CHECK(gaz_step(s) == 1);
  CHECK(gaz_step(s) == 1);
  CHECK(gaz_step(s) == -1);
  CHECK(g.depth == 0 && gaz_simulations(s) == 2);
  gaz_destroy(s);
  g.fail_depth = 0; g.count = 0;
  s = create(&g, cfg); CHECK(s);
  CHECK(gaz_step(s) == 0 && gaz_best_action(s) == -1);
  CHECK(gaz_policy(s, NULL, 0) == 0);
  gaz_destroy(s);
  cfg.max_depth = GAZ_MAX_DEPTH + 1; CHECK(!create(&g, cfg));
  cfg = config(5); g.count = 1; g.prior[0] = NAN; CHECK(!create(&g, cfg));
}
int main(void) {
  test_budgets(); test_value_and_policy(); test_schedule(); test_noise();
  test_completion_and_interior(); test_continuation_and_wdl(); test_errors_and_depth();
  puts("Gumbel C search tests passed");
  return 0;
}
