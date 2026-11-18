/* SPDX-License-Identifier: GPL-3.0-or-later */
#include "search/gumbel/gumbel.h"

#include <algorithm>
#include <atomic>
#include <bit>
#include <chrono>
#include <cmath>
#include <condition_variable>
#include <exception>
#include <iomanip>
#include <limits>
#include <numeric>
#include <sstream>
#include <mutex>
#include <thread>

#include "chess/gamestate.h"
#include "chess/uciloop.h"
#include "neural/backend.h"
#include "neural/encoder.h"
#include "search/register.h"
#include "search/search.h"

namespace lczero {
namespace {
const OptionId kSimulations{"gumbel-simulations", "GumbelSimulations",
    "Simulation budget when go nodes is absent."};
const OptionId kBatchSize{"gumbel-batch-size", "GumbelBatchSize",
    "Maximum simultaneous leaf evaluations; 0 uses backend recommendation (up to 64), 1 is serial."};
const OptionId kConsidered{"gumbel-considered", "GumbelConsideredMoves",
    "Maximum root candidates for sequential halving."};
const OptionId kCVisit{"gumbel-c-visit", "GumbelCVisit", "Q visit-scale offset."};
const OptionId kCScale{"gumbel-c-scale", "GumbelCScale", "Completed Q scale."};
const OptionId kNoise{"gumbel-noise", "GumbelNoiseScale",
    "Root Gumbel scale; zero gives deterministic search."};
const OptionId kSeed{"gumbel-seed", "GumbelSeed", "Reproducible root noise seed."};

const OptionId kLive{"log-live-stats", "LogLiveStats", "Stream per-move statistics during analysis."};
const OptionId kVerbose{"verbose-move-stats", "VerboseMoveStats", "Show root statistics at the end of a search."};
const OptionId kOverhead{"move-overhead", "MoveOverheadMs", "Clock safety margin in milliseconds."};
const OptionId kMultiPv{"multipv", "MultiPV", "Number of principal variations to report."};
const OptionId kScore{"score-type", "ScoreType", "UCI score conversion."};
const OptionId kSmartPruning{"smart-pruning-factor", "SmartPruningFactor",
    "GUI compatibility option; Gumbel uses sequential halving instead of classic smart pruning."};

class GumbelSearch final : public SearchBase {
 public:
  GumbelSearch(UciResponder* responder, const OptionsDict* options)
      : SearchBase(responder), options_(options) {}
  ~GumbelSearch() override { AbortSearch(); WaitSearch(); }
  void SetPosition(const GameState& state) override {
    AbortSearch();
    WaitSearch();
    state_ = state;
  }
  void StartClock() override { clock_start_ = Clock::now(); }
  void StartSearch(const GoParams& params) override {
    AbortSearch();
    WaitSearch();
    stop_ = false;
    abort_ = false;
    if (clock_start_ == Clock::time_point{}) StartClock();
    gaz_config cfg{
        .simulations = std::max(0, params.nodes.value_or(options_->Get<int>(kSimulations))),
        .max_considered = options_->Get<int>(kConsidered),
        .max_depth = std::clamp(params.depth.value_or(GAZ_MAX_DEPTH), 1, GAZ_MAX_DEPTH),
        .c_visit = options_->Get<float>(kCVisit),
        .c_scale = options_->Get<float>(kCScale),
        .noise_scale = options_->Get<float>(kNoise),
        .seed = static_cast<uint64_t>(options_->Get<int>(kSeed))};
    const auto start = clock_start_;
    clock_start_ = {};
    worker_ = std::thread([this, params, cfg, start] { Run(params, cfg, start); });
  }
  void WaitSearch() override { if (worker_.joinable()) worker_.join(); }
  void StopSearch() override {
    { std::lock_guard lock(wait_mutex_); stop_ = true; }
    wake_.notify_all();
  }
  void AbortSearch() override { abort_ = true; StopSearch(); }

 private:
  using Clock = std::chrono::steady_clock;
  MoveList Moves() const {
    return history_.GetLength() == root_length_ ? root_moves_ :
        history_.Last().GetBoard().GenerateLegalMoves();
  }
  // Legal moves are generated once on expansion and stored as tokens on C
  // edges. Replaying a path no longer regenerates legal moves at every ply.
  int Prepare(gaz_evaluation* result) {
    eval_moves_ = Moves();
    const auto& pos = history_.Last();
    const auto& board = pos.GetBoard();
    result->count = 0;
    if (eval_moves_.empty()) {
      result->value = board.IsUnderCheck() ? -1 : 0;
      result->draw = board.IsUnderCheck() ? 0 : 1;
      return 0;
    }
    if (!board.HasMatingMaterial() || pos.GetRule50Ply() >= 100 || pos.GetRepetitions() >= 2) {
      result->value = 0;
      result->draw = 1;
      return 0;
    }
    if (eval_moves_.size() > GAZ_MAX_ACTIONS) throw Exception("Too many legal moves");
    result->count = static_cast<int>(eval_moves_.size());
    for (int i = 0; i < result->count; ++i) result->actions[i] = eval_moves_[i].raw_data();
    return result->count;
  }
  static int Evaluate(void* context, float* value, float* draw, float* p) noexcept {
    auto& self = *static_cast<GumbelSearch*>(context);
    try {
      gaz_evaluation result;
      const int count = self.Prepare(&result);
      if (count) {
        auto computation = self.backend_->CreateComputation();
        const auto status = computation->AddInput(
            {self.history_.GetPositions(), self.eval_moves_},
            {.q = &result.value, .d = &result.draw,
             .p = std::span(result.probabilities, count)});
        if (status == BackendComputation::ENQUEUED_FOR_EVAL) {
          computation->ComputeBlocking();
          ++self.network_evaluations_;
        }
        std::copy_n(result.probabilities, count, p);
      }
      *value = result.value;
      *draw = result.draw;
      return count;
    } catch (...) { self.error_ = std::current_exception(); return -1; }
  }
  static int Enqueue(void* context, gaz_evaluation* result) noexcept {
    auto& self = *static_cast<GumbelSearch*>(context);
    try {
      const int count = self.Prepare(result);
      if (!count) return 1;
      if (!self.computation_) self.computation_ = self.backend_->CreateComputation();
      const auto status = self.computation_->AddInput(
          {self.history_.GetPositions(), self.eval_moves_},
          {.q = &result->value, .d = &result->draw,
           .p = std::span(result->probabilities, count)});
      if (status == BackendComputation::ENQUEUED_FOR_EVAL) ++self.queued_evaluations_;
      return 1;
    } catch (...) { self.error_ = std::current_exception(); return 0; }
  }
  static int Flush(void* context) noexcept {
    auto& self = *static_cast<GumbelSearch*>(context);
    try {
      if (self.queued_evaluations_) {
        self.computation_->ComputeBlocking();
        self.network_evaluations_ += self.queued_evaluations_;
      }
      return 1;
    } catch (...) { self.error_ = std::current_exception(); return 0; }
  }
  static int ActionToken(void* context, int action) {
    return static_cast<GumbelSearch*>(context)->eval_moves_[action].raw_data();
  }
  static int Push(void* context, int token) noexcept {
    auto& self = *static_cast<GumbelSearch*>(context);
    try {
      self.history_.Append(std::bit_cast<Move>(static_cast<uint16_t>(token)));
      return 1;
    } catch (...) { self.error_ = std::current_exception(); return 0; }
  }
  static void Pop(void* context) {
    static_cast<GumbelSearch*>(context)->history_.Pop();
  }
  int Score(float q, float d) const {
    const auto& type = options_->Get<std::string>(kScore);
    if (type == "win_percentage") return std::lround(5000 * (1 + q));
    if (type == "Q" || type == "W-L") return std::lround(10000 * q);
    if (type == "centipawn_2019") return 295 * q / (1 - 0.976953126 * std::pow(q, 14));
    if (type == "centipawn_2018") return 290.680623072 * std::tan(1.548090806 * q);
    if (type == "WDL_mu") {
      const float w = (1 + q - d) / 2, l = (1 - q - d) / 2;
      const float cp = 45 * std::tan(1.56728071628 * q);
      if (backend_->GetAttributes().has_wdl && w > 0.0001f && l > 0.0001f &&
          d > 0.0001f && std::abs(q) + d < 0.996f) {
        const double a = std::log(1 / l - 1), b = std::log(1 / w - 1);
        const double mu = (a - b) / (a + b);
        if (mu != 0 && (std::abs(mu) < 1 || std::abs(cp) < std::abs(100 * mu)))
          return 100 * mu;
      }
      return cp;
    }
    return 90 * std::tan(1.5637541897 * q);
  }
  MoveList PrincipalVariation(const gaz_search* tree, int action) const {
    int actions[GAZ_MAX_DEPTH];
    const int count = gaz_principal_variation(tree, action, actions, GAZ_MAX_DEPTH);
    PositionHistory history(history_);
    MoveList pv;
    for (int i = 0; i < count; ++i) {
      const MoveList moves = i ? history.Last().GetBoard().GenerateLegalMoves() : root_moves_;
      const Move move = moves.at(actions[i]);
      Move white = move;
      if (history.IsBlackToMove()) white.Flip();
      pv.push_back(white);
      history.Append(move);
    }
    return pv;
  }
  Move Report(const gaz_search* tree, Clock::time_point start, bool final) {
    const int best = gaz_best_action(tree);
    const int count = best < 0 ? 0 : static_cast<int>(root_moves_.size());
    float improved[GAZ_MAX_ACTIONS];
    gaz_policy(tree, improved, GAZ_MAX_ACTIONS);
    std::vector<int> order(count);
    std::iota(order.begin(), order.end(), 0);
    std::stable_sort(order.begin(), order.end(), [&](int a, int b) {
      const int na = gaz_action_visits(tree, a), nb = gaz_action_visits(tree, b);
      return na == nb ? improved[a] > improved[b] : na > nb;
    });
    if (best >= 0) {
      const auto it = std::find(order.begin(), order.end(), best);
      std::rotate(order.begin(), it, it + 1);
    }
    const auto elapsed = std::max<int64_t>(0, std::chrono::duration_cast<std::chrono::milliseconds>(
        Clock::now() - start).count());
    const int nodes = gaz_simulations(tree);
    const int nps = static_cast<int>(std::min<int64_t>(std::numeric_limits<int>::max(),
        static_cast<int64_t>(nodes) * 1000 / std::max<int64_t>(1, elapsed)));
    std::vector<ThinkingInfo> infos;
    const int variants = std::min(count, options_->Get<int>(kMultiPv));
    for (int rank = 0; rank < std::max(1, variants); ++rank) {
      const int a = count ? order[rank] : -1;
      const float q = std::clamp(gaz_action_value(tree, a), -1.0f, 1.0f);
      const float d = std::clamp(gaz_action_draw(tree, a), 0.0f, 1 - std::abs(q));
      const int w = std::lround(500 * (1 + q - d)), l = std::lround(500 * (1 - q - d));
      infos.push_back({.depth = gaz_depth(tree), .seldepth = gaz_depth(tree),
          .time = elapsed, .nodes = nodes, .nps = nps,
          .eps = static_cast<int>(std::min<int64_t>(std::numeric_limits<int>::max(),
              network_evaluations_ * 1000 / std::max<int64_t>(1, elapsed))),
          .score = Score(q, d),
          .wdl = ThinkingInfo::WDL{w, 1000 - w - l, l},
          .pv = PrincipalVariation(tree, a), .multipv = rank + 1});
    }
    if (options_->Get<bool>(kLive) || (final && options_->Get<bool>(kVerbose))) {
      // Nibbler ranks verbose rows by arrival order: strongest must be last.
      for (auto it = order.rbegin(); it != order.rend(); ++it) {
        const int a = *it;
        Move move = root_moves_[a];
        if (history_.IsBlackToMove()) move.Flip();
        std::ostringstream row;
        row << std::fixed << std::setprecision(8) << move.ToString(true)
            << " (" << MoveToNNIndex(root_moves_[a], 0) << ") N: "
            << gaz_action_visits(tree, a) << " (+0) (P: " << 100 * gaz_action_prior(tree, a)
            << "%) (WL: " << gaz_action_value(tree, a) << ") (D: " << gaz_action_draw(tree, a)
            << ") (M: -.---) (Q: " << gaz_action_value(tree, a)
            << ") (U: 0.00000000) (S: " << improved[a]
            << ") (V: " << gaz_action_raw_value(tree, a) << ")";
        infos.push_back({.comment = row.str()});
      }
      infos.push_back({.comment = "node (" + std::to_string(count) + ") N: " +
          std::to_string(nodes) + " (+0)"});
    }
    if (!abort_) uci_responder_->OutputThinkingInfo(&infos);
    return best < 0 ? Move{} : root_moves_[best];
  }
  void Run(const GoParams& params, const gaz_config& cfg, Clock::time_point start) {
    Move best;
    try {
      history_ = PositionHistory(state_.GetPositions());
      history_.Reserve(history_.GetLength() + GAZ_MAX_DEPTH);
      root_length_ = history_.GetLength();
      root_moves_ = history_.Last().GetBoard().GenerateLegalMoves();
      if (!params.searchmoves.empty()) {
        MoveList allowed;
        for (const auto& text : params.searchmoves) {
          const Move move = history_.Last().GetBoard().ParseMove(text);
          if (std::find(root_moves_.begin(), root_moves_.end(), move) != root_moves_.end() &&
              std::find(allowed.begin(), allowed.end(), move) == allowed.end())
            allowed.push_back(move);
        }
        root_moves_ = std::move(allowed);
      }
      if (!root_moves_.empty()) best = root_moves_.front();
      error_ = nullptr;
      network_evaluations_ = 0;
      computation_.reset();
      const auto attrs = backend_->GetAttributes();
      const int requested_batch = options_->Get<int>(kBatchSize);
      const int auto_batch = attrs.runs_on_cpu ? 1 : std::clamp(attrs.recommended_batch_size, 1, 64);
      const int batch_size = std::clamp(requested_batch ? requested_batch : auto_batch,
          1, std::clamp(attrs.maximum_batch_size, 1, GAZ_MAX_ACTIONS));
      std::unique_ptr<gaz_search, decltype(&gaz_destroy)> tree(
          gaz_create(&cfg, {this, Evaluate, Push, Pop, ActionToken}), gaz_destroy);
      if (!tree) {
        if (error_) std::rethrow_exception(error_);
        throw Exception("Cannot initialize Gumbel search");
      }
      auto deadline = Clock::time_point::max();
      if (!params.infinite && !params.ponder) {
        std::optional<int64_t> time = params.movetime;
        if (!time) {
          const bool black = history_.IsBlackToMove();
          const auto remaining = black ? params.btime : params.wtime;
          const auto inc = black ? params.binc : params.winc;
          if (remaining) {
            const int64_t available = std::max<int64_t>(1, *remaining - options_->Get<int>(kOverhead));
            time = std::min(available, available / std::max(1, params.movestogo.value_or(30)) +
                                        inc.value_or(0) * 3 / 4);
          }
        }
        if (time) deadline = start + std::chrono::milliseconds(std::max<int64_t>(0, *time));
      }
      const bool continuous = !params.nodes && (params.infinite || params.ponder ||
          params.movetime || params.wtime || params.btime);
      int round_budget = std::max(1, cfg.simulations);
      auto next_report = Clock::now();
      while (!stop_ && Clock::now() < deadline) {
        queued_evaluations_ = 0;
        const int status = batch_size == 1 ? gaz_step(tree.get()) :
            gaz_step_batch(tree.get(), batch_size, {this, Enqueue, Flush});
        // All result pointers belong to C batch scratch; release the backend
        // computation before that scratch can be reused, including on error.
        computation_.reset();
        if (status < 0) {
          if (error_) std::rethrow_exception(error_);
          throw Exception("Gumbel simulation failed");
        }
        if (!status) {
          if (!continuous) break;
          round_budget = std::min(500000, round_budget) * 2;
          if (!gaz_begin_round(tree.get(), round_budget)) break;
        }
        const auto now = Clock::now();
        if (now >= next_report) {
          best = Report(tree.get(), start, false);
          next_report = now + std::chrono::milliseconds(200);
        }
      }
      best = Report(tree.get(), start, true);
    } catch (const std::exception& e) {
      if (!abort_) {
        std::vector<ThinkingInfo> infos{{.comment = std::string("Gumbel search: ") + e.what()}};
        uci_responder_->OutputThinkingInfo(&infos);
      }
    }
    if ((params.infinite || params.ponder) && !stop_) {
      std::unique_lock lock(wait_mutex_);
      wake_.wait(lock, [this] { return stop_.load(); });
    }
    if (!abort_) {
      if (!best.is_null() && state_.CurrentPosition().IsBlackToMove()) best.Flip();
      BestMoveInfo info{best};
      uci_responder_->OutputBestMove(&info);
    }
  }
  const OptionsDict* options_;
  GameState state_;
  PositionHistory history_;
  MoveList root_moves_, eval_moves_;
  std::unique_ptr<BackendComputation> computation_;
  int queued_evaluations_ = 0;
  int64_t network_evaluations_ = 0;
  int root_length_ = 0;
  std::exception_ptr error_;
  Clock::time_point clock_start_{};
  std::thread worker_;
  std::atomic<bool> stop_{false}, abort_{false};
  std::mutex wait_mutex_;
  std::condition_variable wake_;
};

class GumbelSearchFactory final : public SearchFactory {
  std::string_view GetName() const override { return "gumbel"; }
  void PopulateParams(OptionsParser* options) const override {
    options->Add<BoolOption>(kLive) = false;
    options->Add<BoolOption>(kVerbose) = false;
    options->Add<IntOption>(kOverhead, 0, 1000000000) = 20;
    options->Add<IntOption>(kMultiPv, 1, 500) = 1;
    options->Add<ChoiceOption>(kScore, std::vector<std::string>{"centipawn",
        "centipawn_with_drawscore", "centipawn_2019", "centipawn_2018",
        "win_percentage", "Q", "W-L", "WDL_mu"}) = "WDL_mu";
    options->Add<FloatOption>(kSmartPruning, 0, 10) = 0;
    options->Add<IntOption>(kSimulations, 1, 1000000000) = 1024;
    options->Add<IntOption>(kBatchSize, 0, GAZ_MAX_ACTIONS) = 0;
    options->Add<IntOption>(kConsidered, 1, GAZ_MAX_ACTIONS) = 16;
    options->Add<FloatOption>(kCVisit, 0, 10000) = 50;
    options->Add<FloatOption>(kCScale, 0, 100) = 0.1f;
    options->Add<FloatOption>(kNoise, 0, 100) = 1;
    options->Add<IntOption>(kSeed, 0, 2147483647) = 0;
  }
  std::unique_ptr<SearchBase> CreateSearch(UciResponder* responder,
                                         const OptionsDict* options) const override {
    return std::make_unique<GumbelSearch>(responder, options);
  }
};
REGISTER_SEARCH(GumbelSearchFactory)
}  // namespace
}  // namespace lczero
