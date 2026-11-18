/* SPDX-License-Identifier: GPL-3.0-or-later */
#include "search/gumbel/gumbel.h"
#include <float.h>
#include <limits.h>
#include <math.h>
#include <stdlib.h>

typedef struct node node;
typedef struct {
  double prior, log_prior, sum, draw_sum;
  int visits, in_flight, pending, action;
  node* child;
} edge;
struct node {
  float value, draw;
  int count;
  edge* edges;
};
typedef struct {
  edge* path[GAZ_MAX_DEPTH];
  int depth;
  edge* leaf_edge;
  node* expansion;
  gaz_evaluation evaluation;
} pending_leaf;
struct gaz_search {
  gaz_config cfg;
  gaz_callbacks cb;
  node* root;
  pending_leaf* batch;
  int batch_capacity;
  double noise[GAZ_MAX_ACTIONS];
  int round_visits[GAZ_MAX_ACTIONS];
  int done, depth, round_budget;
  int considered, considered_visit, in_cycle, cycles, extra, log2max;
};

static void free_node(node* n) {
  if (!n) return;
  for (int i = 0; i < n->count; ++i) free_node(n->edges[i].child);
  free(n->edges);
  free(n);
}

static node* make_node(int count, float value, float draw, const float* p,
                       const int* actions) {
  if (count < 0 || count > GAZ_MAX_ACTIONS || !isfinite(value) ||
      value < -1.0f || value > 1.0f || !isfinite(draw) || draw < 0 ||
      draw > 1.0f) return NULL;
  node* n = (node*)calloc(1, sizeof(*n));
  if (!n) return NULL;
  n->value = value;
  n->draw = fminf(draw, 1.0f - fabsf(value));
  if (!count) return n;
  n->edges = (edge*)calloc((size_t)count, sizeof(edge));
  if (!n->edges) { free(n); return NULL; }
  n->count = count;
  double sum = 0;
  for (int i = 0; i < count; ++i) {
    if (!isfinite(p[i]) || p[i] < 0) { free_node(n); return NULL; }
    /* Retain finite log priors for probabilities underflowed by a backend. */
    n->edges[i].action = actions ? actions[i] : i;
    n->edges[i].prior = fmax(p[i], FLT_MIN);
    sum += n->edges[i].prior;
  }
  for (int i = 0; i < count; ++i) {
    n->edges[i].prior /= sum;
    n->edges[i].log_prior = log(n->edges[i].prior);
  }
  return n;
}

static node* expand(gaz_callbacks cb) {
  float p[GAZ_MAX_ACTIONS], value, draw;
  int actions[GAZ_MAX_ACTIONS];
  const int count = cb.evaluate(cb.context, &value, &draw, p);
  if (count < 0 || count > GAZ_MAX_ACTIONS) return NULL;
  for (int i = 0; i < count; ++i)
    actions[i] = cb.action_token ? cb.action_token(cb.context, i) : i;
  return make_node(count, value, draw, p, actions);
}

/* SplitMix64; midpoint conversion keeps U strictly in (0,1). */
static double uniform(uint64_t* state) {
  uint64_t z = (*state += UINT64_C(0x9e3779b97f4a7c15));
  z = (z ^ (z >> 30)) * UINT64_C(0xbf58476d1ce4e5b9);
  z = (z ^ (z >> 27)) * UINT64_C(0x94d049bb133111eb);
  z ^= z >> 31;
  return ((double)(z >> 12) + 0.5) * (1.0 / 4503599627370496.0);
}

/* Appendix D mixed-value completion, then min/max rescaling as in mctx. */
static void transformed(const gaz_search* s, const node* n, double* q) {
  double total = 0, mass = 0, weighted = 0;
  int maxvisit = 0;
  for (int i = 0; i < n->count; ++i) {
    const edge* e = &n->edges[i];
    total += e->visits;
    if (e->visits) {
      mass += e->prior;
      weighted += e->prior * e->sum / e->visits;
    }
    if (e->visits > maxvisit) maxvisit = e->visits;
  }
  const double mixed = (n->value + total * (mass ? weighted / mass : 0)) /
                       (total + 1);
  double lo = DBL_MAX, hi = -DBL_MAX;
  for (int i = 0; i < n->count; ++i) {
    const edge* e = &n->edges[i];
    q[i] = e->visits ? e->sum / e->visits : mixed;
    lo = fmin(lo, q[i]);
    hi = fmax(hi, q[i]);
  }
  const double scale = (s->cfg.c_visit + maxvisit) * s->cfg.c_scale /
                       fmax(hi - lo, 1e-8);
  for (int i = 0; i < n->count; ++i) q[i] = (q[i] - lo) * scale;
}

static void policy(const gaz_search* s, const node* n, double* p) {
  transformed(s, n, p);
  double maxlog = -DBL_MAX, total = 0;
  for (int i = 0; i < n->count; ++i) {
    p[i] += n->edges[i].log_prior;
    maxlog = fmax(maxlog, p[i]);
  }
  for (int i = 0; i < n->count; ++i) total += (p[i] = exp(p[i] - maxlog));
  for (int i = 0; i < n->count; ++i) p[i] /= total;
}

static int root_select(const gaz_search* s, int visit) {
  double q[GAZ_MAX_ACTIONS], best = -DBL_MAX;
  int action = -1;
  transformed(s, s->root, q);
  for (int i = 0; i < s->root->count; ++i) {
    const edge* e = &s->root->edges[i];
    double score = s->noise[i] + e->log_prior + q[i];
    if (!e->pending && s->round_visits[i] == visit && score > best) { best = score; action = i; }
  }
  return action;
}

static int interior_select(const gaz_search* s, const node* n) {
  double p[GAZ_MAX_ACTIONS], total = 1, best = -DBL_MAX;
  int action = -1;
  policy(s, n, p);
  for (int i = 0; i < n->count; ++i) total += n->edges[i].visits + n->edges[i].in_flight;
  for (int i = 0; i < n->count; ++i) {
    double score = p[i] - (n->edges[i].visits + n->edges[i].in_flight) / total;
    if (!n->edges[i].pending && score > best) { best = score; action = i; }
  }
  return action;
}

static void schedule_round(gaz_search* s, int budget) {
  s->round_budget = budget;
  s->considered_visit = s->in_cycle = s->cycles = s->log2max = 0;
  for (int i = 0; i < GAZ_MAX_ACTIONS; ++i) s->round_visits[i] = 0;
  s->considered = s->root->count;
  if (s->considered > s->cfg.max_considered) s->considered = s->cfg.max_considered;
  if (s->considered > budget && budget > 0)
    s->considered = budget;
  for (int k = 1; k < s->considered; k *= 2) ++s->log2max;
  s->extra = s->considered > 1 ?
      budget / (s->log2max * s->considered) : 1;
  if (s->extra < 1) s->extra = 1;
}

int gaz_begin_round(gaz_search* s, int simulations) {
  if (!s || !s->root->count || s->done != s->cfg.simulations ||
      simulations < 1 || simulations > INT_MAX - s->done) return 0;
  s->cfg.simulations = s->done + simulations;
  schedule_round(s, simulations);
  return 1;
}

gaz_search* gaz_create(const gaz_config* cfg, gaz_callbacks cb) {
  if (!cfg || !cb.evaluate || !cb.push || !cb.pop || cfg->simulations < 0 ||
      cfg->max_considered < 1 || cfg->max_depth < 1 ||
      cfg->max_depth > GAZ_MAX_DEPTH || !isfinite(cfg->c_visit) ||
      !isfinite(cfg->c_scale) || !isfinite(cfg->noise_scale) ||
      cfg->c_visit < 0 || cfg->c_scale < 0 || cfg->noise_scale < 0) return NULL;
  gaz_search* s = (gaz_search*)calloc(1, sizeof(*s));
  if (!s) return NULL;
  s->cfg = *cfg;
  s->cb = cb;
  s->root = expand(cb);
  if (!s->root) { free(s); return NULL; }
  uint64_t seed = cfg->seed;
  for (int i = 0; i < s->root->count; ++i)
    s->noise[i] = cfg->noise_scale * -log(-log(uniform(&seed)));
  schedule_round(s, cfg->simulations);
  return s;
}

static void advance_schedule(gaz_search* s) {
  if (++s->in_cycle == s->considered) {
    s->in_cycle = 0;
    ++s->considered_visit;
    if (++s->cycles == s->extra && s->considered > 1) {
      s->cycles = 0;
      s->considered /= 2;
      if (s->considered < 2) s->considered = 2;
      s->extra = s->round_budget / (s->log2max * s->considered);
      if (s->extra < 1) s->extra = 1;
    }
  }
}

int gaz_step(gaz_search* s) {
  if (!s || !s->root->count || s->done >= s->cfg.simulations) return 0;
  edge* path[GAZ_MAX_DEPTH];
  node* n = s->root;
  int depth = 0, failed = 0;
  while (n->count && depth < s->cfg.max_depth) {
    int a = depth ? interior_select(s, n) : root_select(s, s->considered_visit);
    if (a < 0 || !s->cb.push(s->cb.context, n->edges[a].action)) { failed = 1; break; }
    edge* e = &n->edges[a];
    path[depth++] = e;
    if (!e->child) {
      e->child = expand(s->cb);
      if (!e->child) { failed = 1; break; }
      n = e->child;
      break;
    }
    n = e->child;
  }
  double value = n->value;
  const double draw = n->draw;
  if (!failed) ++s->round_visits[path[0] - s->root->edges];
  if (depth > s->depth) s->depth = depth;
  while (depth) {
    s->cb.pop(s->cb.context);
    edge* e = path[--depth];
    value = -value;
    if (!failed) { ++e->visits; e->sum += value; e->draw_sum += draw; }
  }
  if (failed) return -1;
  ++s->done;
  advance_schedule(s);
  return 1;
}

int gaz_step_batch(gaz_search* s, int max_batch, gaz_batch_callbacks cb) {
  if (!s || max_batch < 1 || !cb.enqueue || !cb.flush) return -1;
  if (max_batch == 1) return gaz_step(s);
  if (!s->root->count || s->done >= s->cfg.simulations) return 0;
  if (max_batch > GAZ_MAX_ACTIONS) max_batch = GAZ_MAX_ACTIONS;
  if (max_batch > s->cfg.simulations - s->done)
    max_batch = s->cfg.simulations - s->done;
  /* Complete all reservations before eliminating any root candidates. */
  if (s->considered > 2) {
    const int phase_left = (s->extra - s->cycles) * s->considered - s->in_cycle;
    if (max_batch > phase_left) max_batch = phase_left;
  }
  if (s->batch_capacity < max_batch) {
    pending_leaf* batch = (pending_leaf*)realloc(s->batch, (size_t)max_batch * sizeof(*batch));
    if (!batch) return -1;
    s->batch = batch;
    s->batch_capacity = max_batch;
  }
  const gaz_search saved = *s;
  int count = 0, failed = 0, enqueued = 0;
  while (count < max_batch) {
    pending_leaf* r = &s->batch[count];
    r->depth = 0;
    r->leaf_edge = NULL;
    r->expansion = NULL;
    node* n = s->root;
    int collision = 0;
    while (n->count && r->depth < s->cfg.max_depth) {
      const int a = r->depth ? interior_select(s, n) : root_select(s, s->considered_visit);
      if (a < 0) { collision = 1; break; }
      edge* e = &n->edges[a];
      if (!s->cb.push(s->cb.context, e->action)) { failed = 1; break; }
      r->path[r->depth++] = e;
      ++e->in_flight;
      if (!e->child) {
        r->leaf_edge = e;
        e->pending = 1;
        r->evaluation.count = -1;
        for (int i = 0; i < GAZ_MAX_ACTIONS; ++i) r->evaluation.actions[i] = i;
        if (!cb.enqueue(cb.context, &r->evaluation)) failed = 1;
        else ++enqueued;
        break;
      }
      n = e->child;
    }
    for (int i = 0; i < r->depth; ++i) s->cb.pop(s->cb.context);
    if (failed || collision) {
      for (int i = 0; i < r->depth; ++i) --r->path[i]->in_flight;
      if (r->leaf_edge) r->leaf_edge->pending = 0;
      break;
    }
    if (!r->leaf_edge) {
      r->evaluation.value = n->value;
      r->evaluation.draw = n->draw;
    }
    ++s->round_visits[r->path[0] - s->root->edges];
    advance_schedule(s);
    ++count;
  }
  if (!failed && enqueued && !cb.flush(cb.context)) failed = 1;
  if (!failed) {
    /* Validate and allocate every expansion before committing any result. */
    for (int i = 0; i < count; ++i) {
      pending_leaf* r = &s->batch[i];
      if (!r->leaf_edge) continue;
      const gaz_evaluation* v = &r->evaluation;
      r->expansion = make_node(v->count, v->value, v->draw, v->probabilities, v->actions);
      if (!r->expansion) { failed = 1; break; }
    }
  }
  for (int i = 0; i < count; ++i) {
    pending_leaf* r = &s->batch[i];
    if (r->leaf_edge) {
      r->leaf_edge->pending = 0;
      if (!failed) r->leaf_edge->child = r->expansion;
      else free_node(r->expansion);
    }
    double value = failed ? 0 : (r->leaf_edge ? r->expansion->value : r->evaluation.value);
    const double draw = failed ? 0 : (r->leaf_edge ? r->expansion->draw : r->evaluation.draw);
    for (int j = r->depth; j > 0; --j) {
      edge* e = r->path[j - 1];
      --e->in_flight;
      value = -value;
      if (!failed) { ++e->visits; e->sum += value; e->draw_sum += draw; }
    }
    if (!failed && r->depth > s->depth) s->depth = r->depth;
  }
  if (failed) { *s = saved; return -1; }
  s->done += count;
  return count;
}

void gaz_destroy(gaz_search* s) { if (s) { free_node(s->root); free(s->batch); free(s); } }
int gaz_best_action(const gaz_search* s) {
  if (!s || !s->root->count) return -1;
  int maxvisit = 0;
  for (int i = 0; i < s->root->count; ++i)
    if (s->round_visits[i] > maxvisit) maxvisit = s->round_visits[i];
  return root_select(s, maxvisit);
}
int gaz_simulations(const gaz_search* s) { return s ? s->done : 0; }
int gaz_depth(const gaz_search* s) { return s ? s->depth : 0; }
float gaz_action_value(const gaz_search* s, int a) {
  if (!s) return 0;
  if (a < 0 || a >= s->root->count) return s->root->value;
  const edge* e = &s->root->edges[a];
  return e->visits ? (float)(e->sum / e->visits) : s->root->value;
}
int gaz_action_visits(const gaz_search* s, int a) {
  return s && a >= 0 && a < s->root->count ? s->root->edges[a].visits : 0;
}
int gaz_policy(const gaz_search* s, float* p, int capacity) {
  if (!s || capacity < s->root->count || (!p && s->root->count)) return -1;
  double probs[GAZ_MAX_ACTIONS];
  policy(s, s->root, probs);
  for (int i = 0; i < s->root->count; ++i) p[i] = (float)probs[i];
  return s->root->count;
}

float gaz_action_draw(const gaz_search* s, int a) {
  if (!s) return 0;
  if (a < 0 || a >= s->root->count) return s->root->draw;
  const edge* e = &s->root->edges[a];
  return e->visits ? (float)(e->draw_sum / e->visits) : s->root->draw;
}
float gaz_action_prior(const gaz_search* s, int a) {
  return s && a >= 0 && a < s->root->count ? (float)s->root->edges[a].prior : 0;
}
float gaz_action_raw_value(const gaz_search* s, int a) {
  if (!s || a < 0 || a >= s->root->count) return 0;
  const node* child = s->root->edges[a].child;
  return child ? -child->value : s->root->value;
}
int gaz_principal_variation(const gaz_search* s, int a, int* actions, int capacity) {
  if (!s || !actions || capacity < 1 || a < 0 || a >= s->root->count) return 0;
  int length = 0;
  const node* n = s->root;
  while (n && n->count && length < capacity) {
    actions[length++] = a;
    n = n->edges[a].child;
    if (!n || !n->count) break;
    a = 0;
    for (int i = 1; i < n->count; ++i)
      if (n->edges[i].visits > n->edges[a].visits) a = i;
    if (!n->edges[a].visits) break;
  }
  return length;
}
