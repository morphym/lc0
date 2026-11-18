/* SPDX-License-Identifier: GPL-3.0-or-later */
#ifndef LCZERO_SEARCH_GUMBEL_H_
#define LCZERO_SEARCH_GUMBEL_H_
#include <stdint.h>
#ifdef __cplusplus
extern "C" {
#endif

/* Legal-action order belongs to the caller; no network-index remapping here. */
#define GAZ_MAX_ACTIONS 256
#define GAZ_MAX_DEPTH 256
typedef struct {
  void* context;
  /* Returns legal action count, zero for terminal, -1 on failure. Value is
     always from the player-to-move perspective; draw is a probability in [0,1],
     and probabilities are in action order. */
  int (*evaluate)(void*, float* value, float* draw, float* probabilities);
  int (*push)(void*, int action);
  void (*pop)(void*);
  /* Optional: translate a legal-action index into a caller move token while
     the just-evaluated position is current. push receives that token. */
  int (*action_token)(void*, int action);
} gaz_callbacks;
typedef struct {
  int simulations;
  int max_considered;
  int max_depth;
  double c_visit, c_scale, noise_scale;
  uint64_t seed;
} gaz_config;
typedef struct gaz_search gaz_search;
typedef struct {
  int count;
  float value, draw;
  float probabilities[GAZ_MAX_ACTIONS];
  int actions[GAZ_MAX_ACTIONS];
} gaz_evaluation;
typedef struct {
  void* context;
  /* Called at the selected leaf, before pop. Fill count/actions immediately;
     value/draw/probabilities may be filled by flush. Return 0 on failure. */
  int (*enqueue)(void*, gaz_evaluation*);
  int (*flush)(void*);
} gaz_batch_callbacks;
/* Reserve distinct leaves and evaluate them together. Never crosses a root
   halving boundary with unresolved values. Returns simulations completed,
   zero at budget exhaustion/terminal root, or -1 on error. Batch=1 matches
   gaz_step exactly. Failed batches undo reservations and schedule changes. */
int gaz_step_batch(gaz_search*, int max_batch, gaz_batch_callbacks);
/* create evaluates the root. NULL means invalid configuration/evaluation/OOM. */
gaz_search* gaz_create(const gaz_config*, gaz_callbacks);
void gaz_destroy(gaz_search*);
/* Start another finite sequential-halving round after exhaustion, retaining
   evaluations/visits and fixed Gumbel noise. Returns 0 if not possible. */
int gaz_begin_round(gaz_search*, int simulations);
/* Root statistics and PV are in caller-supplied legal-action order. */
float gaz_action_draw(const gaz_search*, int action);
float gaz_action_prior(const gaz_search*, int action);
float gaz_action_raw_value(const gaz_search*, int action);
int gaz_principal_variation(const gaz_search*, int action, int* actions, int capacity);
/* One simulation, restores callback position even on error. 1=done, 0=budget
   exhausted/terminal, -1=callback failure/OOM. */
int gaz_step(gaz_search*);
/* -1 for terminal root; result uses Gumbel + log prior + transformed Q among
   the most visited candidates, never the raw visit-count argmax. */
int gaz_best_action(const gaz_search*);
int gaz_simulations(const gaz_search*);
int gaz_depth(const gaz_search*);
float gaz_action_value(const gaz_search*, int action);
/* Improved policy softmax(log prior + sigma(completed Q)), without Gumbel. */
int gaz_policy(const gaz_search*, float* probabilities, int capacity);
int gaz_action_visits(const gaz_search*, int action);
#ifdef __cplusplus
}
#endif
#endif
