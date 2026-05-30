"use client";

import { useEffect, useMemo, useState, type FormEvent } from "react";
import axios from "axios";
import { profileApi } from "@/lib/api";

type Question = {
  id: string;
  category: string;
  text: string;
};

type BehaviorResult = {
  message: string;
  model_training?: {
    started: boolean;
    scope: string;
    user_id: string;
  };
  behavior_profile: {
    behavior_array: Record<string, number>;
    question_count: number;
    risk_profile: {
      risk_tolerance: number;
      category: string;
      description: string;
      recommendations: {
        max_position_size: number;
        suggested_stop_loss: number;
        suggested_take_profit: number;
      };
    };
  };
};

type TrainingStatus = {
  status: "idle" | "queued" | "running" | "completed" | "failed";
  message: string;
  updated_at: string;
  queued_update_pending?: boolean;
  model_path?: string;
};

const QUESTIONS: Question[] = [
  { id: "q_experience", category: "Profile", text: "How experienced are you with active trading?" },
  { id: "q_volatility", category: "Profile", text: "How comfortable are you with high short-term volatility?" },
  { id: "q_drawdown_reaction", category: "Profile", text: "How likely are you to hold through a 15% drawdown?" },
  { id: "q_conviction", category: "Profile", text: "How confident are you in following your pre-trade plan?" },
  { id: "q_overtrade_1", category: "Overtrading & Impulse", text: "How often do you open a new trade shortly after closing one?" },
  { id: "q_overtrade_2", category: "Overtrading & Impulse", text: "How often do you revenge-trade after a loss?" },
  { id: "q_overtrade_3", category: "Overtrading & Impulse", text: "How often do you skip trade setup checks due to FOMO?" },
  { id: "q_overtrade_4", category: "Overtrading & Impulse", text: "How consistently do you wait for your exact entry conditions?" },
  { id: "q_hold_1", category: "Overtrading & Impulse", text: "How often do you cut winners too early?" },
  { id: "q_hold_2", category: "Overtrading & Impulse", text: "How often do you hold losers longer than planned?" },
  { id: "q_rest_1", category: "Overtrading & Impulse", text: "Do you take a cooling-off break after consecutive losses?" },
  { id: "q_rest_2", category: "Overtrading & Impulse", text: "How disciplined are you about daily trade limits?" },
  { id: "q_risk_1", category: "Risk & Account Management", text: "How strictly do you limit risk per trade?" },
  { id: "q_risk_2", category: "Risk & Account Management", text: "How consistently do you reduce size during losing streaks?" },
  { id: "q_risk_3", category: "Risk & Account Management", text: "How often do you breach your max account drawdown limit?" },
  { id: "q_risk_4", category: "Risk & Account Management", text: "How aware are you of correlated positions in your portfolio?" },
  { id: "q_risk_5", category: "Risk & Account Management", text: "How often do you journal the reason for each trade?" },
  { id: "q_risk_6", category: "Risk & Account Management", text: "How consistently do you honor hard stop-loss levels?" },
  { id: "q_context_1", category: "Market Context Execution", text: "How often do you trade during major scheduled news events?" },
  { id: "q_context_2", category: "Market Context Execution", text: "How sensitive are you to entry slippage in fast markets?" },
  { id: "q_context_3", category: "Market Context Execution", text: "How consistently do you adapt sizing by market session?" },
  { id: "q_context_4", category: "Market Context Execution", text: "How often do you avoid low-liquidity periods?" },
  { id: "q_context_5", category: "Market Context Execution", text: "How often do you chase momentum after large candles?" },
  { id: "q_context_6", category: "Market Context Execution", text: "How often do you validate spread/volume before entry?" },
  { id: "q_manage_1", category: "Advanced Trade Management", text: "How often do you scale out using partial take-profits?" },
  { id: "q_manage_2", category: "Advanced Trade Management", text: "How quickly do you move stop-loss to breakeven when valid?" },
  { id: "q_manage_3", category: "Advanced Trade Management", text: "How often do you trail stops in trend trades?" },
  { id: "q_manage_4", category: "Advanced Trade Management", text: "How consistently do you follow planned TP ladders?" },
  { id: "q_manage_5", category: "Advanced Trade Management", text: "How often do you exit full size instead of scaling by plan?" },
  { id: "q_manage_6", category: "Advanced Trade Management", text: "How often do emotions override your exit strategy?" },
];

const SCORE_OPTIONS = [
  { value: 1, label: "Very Low" },
  { value: 2, label: "Low" },
  { value: 3, label: "Moderate" },
  { value: 4, label: "High" },
  { value: 5, label: "Very High" },
];

export function RiskQuestionnaire() {
  const [answers, setAnswers] = useState<Record<string, number>>({});
  const [capitalPerTradePct, setCapitalPerTradePct] = useState(8);
  const [tpSlRatio, setTpSlRatio] = useState(2.0);
  const [maxProfitClosePct, setMaxProfitClosePct] = useState(20);
  const [maxTradesPerDay, setMaxTradesPerDay] = useState(6);
  const [postLossRestMin, setPostLossRestMin] = useState(45);
  const [maxDrawdownPct, setMaxDrawdownPct] = useState(15);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [result, setResult] = useState<BehaviorResult | null>(null);
  const [trainingStatus, setTrainingStatus] = useState<TrainingStatus | null>(null);

  const answeredCount = useMemo(
    () => QUESTIONS.filter((question) => answers[question.id] && answers[question.id] > 0).length,
    [answers],
  );

  const groupedQuestions = useMemo(() => {
    const groups = new Map<string, Question[]>();
    for (const question of QUESTIONS) {
      const existing = groups.get(question.category) ?? [];
      existing.push(question);
      groups.set(question.category, existing);
    }
    return Array.from(groups.entries());
  }, []);

  const onChangeAnswer = (questionId: string, value: number) => {
    setAnswers((prev) => ({ ...prev, [questionId]: value }));
  };

  const onSubmit = async (event: FormEvent) => {
    event.preventDefault();
    setError(null);
    setResult(null);

    if (answeredCount < QUESTIONS.length) {
      setError("Please answer all 30 questions before submitting.");
      return;
    }

    const questionScores = QUESTIONS.map((q) => ({ id: q.id, category: q.category, score: answers[q.id] }));

    const payload: Record<string, unknown> = {
      question_scores: questionScores,
      capital_per_trade_pct: capitalPerTradePct,
      tp_sl_ratio: tpSlRatio,
      max_profit_close_pct: maxProfitClosePct,
      max_trades_per_day: maxTradesPerDay,
      post_loss_rest_min: postLossRestMin,
      max_drawdown_pct: maxDrawdownPct,
      loss_streak_reduce_pct: 25,
      intraday_var_pct: 3,
      entry_slippage_bps: 12,
      session_consistency_score: 60,
      news_buffer_min: 30,
      partial_tp_frequency: 2,
      breakeven_trigger_pct: 1,
      breakeven_migration_time_min: 60,
      avg_holding_time_min: 240,
    };

    try {
      setLoading(true);
      const data = await profileApi.submitBehaviorAssessment(payload);
      setResult(data);
      const status = await profileApi.getModelTrainingStatus();
      setTrainingStatus(status);
    } catch (err: unknown) {
      if (axios.isAxiosError(err) && typeof err.response?.data?.detail === "string") {
        setError(err.response.data.detail);
      } else {
        setError("Failed to submit assessment.");
      }
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    if (!trainingStatus) return;
    if (trainingStatus.status !== "queued" && trainingStatus.status !== "running") return;

    const interval = setInterval(async () => {
      try {
        const status = await profileApi.getModelTrainingStatus();
        setTrainingStatus(status);
      } catch {
        // keep last state if polling fails momentarily
      }
    }, 3000);

    return () => clearInterval(interval);
  }, [trainingStatus]);

  return (
    <div className="max-w-5xl mx-auto">
      <div className="mb-6">
        <h1 className="text-3xl font-bold text-white">Behavior & Risk Assessment</h1>
        <p className="text-gray-400 mt-2">
          Complete 30 categorized questions and define execution constraints for feedback-loop training.
        </p>
      </div>

      <div className="mb-6 p-4 bg-gray-900/50 border border-gray-800 rounded-xl">
        <div className="flex justify-between text-sm text-gray-400 mb-2">
          <span>Progress</span>
          <span>{answeredCount}/{QUESTIONS.length}</span>
        </div>
        <div className="h-2 bg-gray-800 rounded-full overflow-hidden">
          <div
            className="h-full bg-gradient-to-r from-blue-500 to-cyan-400 transition-all"
            style={{ width: `${(answeredCount / QUESTIONS.length) * 100}%` }}
          />
        </div>
      </div>

      <form onSubmit={onSubmit} className="space-y-6">
        <div className="p-5 bg-gray-900/50 border border-gray-800 rounded-xl">
          <h2 className="text-lg font-semibold text-white mb-4">Feedback Loop Constraints</h2>
          <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
            <label className="text-sm text-gray-300">
              Percentage of capital in one trade
              <input
                type="number"
                min={1}
                max={100}
                value={capitalPerTradePct}
                onChange={(e) => setCapitalPerTradePct(Number(e.target.value))}
                className="mt-1 w-full px-3 py-2 rounded-lg border border-gray-700 bg-gray-800 text-white"
              />
            </label>
            <label className="text-sm text-gray-300">
              TP/SL ratio target
              <input
                type="number"
                min={0.5}
                max={10}
                step={0.1}
                value={tpSlRatio}
                onChange={(e) => setTpSlRatio(Number(e.target.value))}
                className="mt-1 w-full px-3 py-2 rounded-lg border border-gray-700 bg-gray-800 text-white"
              />
            </label>
            <label className="text-sm text-gray-300">
              Max profit % when closing trade
              <input
                type="number"
                min={1}
                max={100}
                value={maxProfitClosePct}
                onChange={(e) => setMaxProfitClosePct(Number(e.target.value))}
                className="mt-1 w-full px-3 py-2 rounded-lg border border-gray-700 bg-gray-800 text-white"
              />
            </label>
            <label className="text-sm text-gray-300">
              Max trades per day
              <input
                type="number"
                min={1}
                max={40}
                value={maxTradesPerDay}
                onChange={(e) => setMaxTradesPerDay(Number(e.target.value))}
                className="mt-1 w-full px-3 py-2 rounded-lg border border-gray-700 bg-gray-800 text-white"
              />
            </label>
            <label className="text-sm text-gray-300">
              Rest period after loss (minutes)
              <input
                type="number"
                min={0}
                max={1440}
                value={postLossRestMin}
                onChange={(e) => setPostLossRestMin(Number(e.target.value))}
                className="mt-1 w-full px-3 py-2 rounded-lg border border-gray-700 bg-gray-800 text-white"
              />
            </label>
            <label className="text-sm text-gray-300">
              Max account drawdown tolerance (%)
              <input
                type="number"
                min={1}
                max={100}
                value={maxDrawdownPct}
                onChange={(e) => setMaxDrawdownPct(Number(e.target.value))}
                className="mt-1 w-full px-3 py-2 rounded-lg border border-gray-700 bg-gray-800 text-white"
              />
            </label>
          </div>
        </div>

        {groupedQuestions.map(([category, questions]) => (
          <div key={category} className="p-5 bg-gray-900/50 border border-gray-800 rounded-xl space-y-4">
            <h2 className="text-lg font-semibold text-white">{category}</h2>
            {questions.map((question, index) => (
              <div key={question.id} className="p-4 bg-gray-900 border border-gray-800 rounded-xl">
                <p className="text-white font-medium mb-3">
                  {index + 1}. {question.text}
                </p>
                <div className="grid grid-cols-2 md:grid-cols-5 gap-2">
                  {SCORE_OPTIONS.map((option) => (
                    <button
                      key={option.value}
                      type="button"
                      onClick={() => onChangeAnswer(question.id, option.value)}
                      className={`px-3 py-2 rounded-lg border text-sm transition-colors ${
                        answers[question.id] === option.value
                          ? "border-blue-500 bg-blue-500/20 text-blue-300"
                          : "border-gray-700 bg-gray-800 text-gray-300 hover:border-gray-600"
                      }`}
                    >
                      {option.label}
                    </button>
                  ))}
                </div>
              </div>
            ))}
          </div>
        ))}

        {error && (
          <div className="p-3 bg-red-500/10 border border-red-500/30 rounded-lg text-red-400 text-sm">
            {error}
          </div>
        )}

        <button
          type="submit"
          disabled={loading}
          className="w-full py-3 rounded-xl bg-gradient-to-r from-blue-500 to-indigo-500 text-white font-semibold disabled:opacity-50 disabled:cursor-not-allowed"
        >
          {loading ? "Submitting Assessment & Starting PPO Training..." : "Submit Behavior Assessment"}
        </button>
      </form>

      {result && (
        <div className="mt-6 p-6 bg-gray-900/50 border border-gray-800 rounded-xl">
          <h2 className="text-xl font-semibold text-white mb-1">
            {result.behavior_profile.risk_profile.category} Profile
          </h2>
          <p className="text-gray-300 mb-4">{result.behavior_profile.risk_profile.description}</p>

          {result.model_training?.started ? (
            <div className="mb-4 p-3 rounded-lg border border-blue-500/40 bg-blue-500/10 text-blue-300 text-sm space-y-1">
              <p>PPO retraining started for your profile.</p>
              <p className="text-xs text-blue-200">
                Status: {trainingStatus?.status?.toUpperCase() ?? "QUEUED"} · {trainingStatus?.message ?? "Waiting..."}
              </p>
              {trainingStatus?.queued_update_pending ? (
                <p className="text-xs text-blue-200">A newer reassessment update is queued.</p>
              ) : null}
              {trainingStatus?.status === "completed" && trainingStatus?.model_path ? (
                <p className="text-xs text-green-300">Completed: {trainingStatus.model_path}</p>
              ) : null}
              {trainingStatus?.status === "failed" ? (
                <p className="text-xs text-red-300">Training failed. Please retry assessment.</p>
              ) : null}
            </div>
          ) : null}

          <div className="grid grid-cols-1 md:grid-cols-4 gap-3 text-sm">
            <div className="p-3 bg-gray-800 rounded-lg border border-gray-700">
              <p className="text-gray-400">Risk Score</p>
              <p className="text-white font-semibold">
                {(result.behavior_profile.risk_profile.risk_tolerance * 100).toFixed(0)}%
              </p>
            </div>
            <div className="p-3 bg-gray-800 rounded-lg border border-gray-700">
              <p className="text-gray-400">Behavior Features</p>
              <p className="text-white font-semibold">
                {Object.keys(result.behavior_profile.behavior_array || {}).length}
              </p>
            </div>
            <div className="p-3 bg-gray-800 rounded-lg border border-gray-700">
              <p className="text-gray-400">Max Position Size</p>
              <p className="text-white font-semibold">
                {(result.behavior_profile.risk_profile.recommendations.max_position_size * 100).toFixed(0)}%
              </p>
            </div>
            <div className="p-3 bg-gray-800 rounded-lg border border-gray-700">
              <p className="text-gray-400">Suggested TP</p>
              <p className="text-white font-semibold">
                {(result.behavior_profile.risk_profile.recommendations.suggested_take_profit * 100).toFixed(0)}%
              </p>
            </div>
          </div>
        </div>
      )}
    </div>
  );
}
