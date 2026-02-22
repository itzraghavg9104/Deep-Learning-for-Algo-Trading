"use client";

import { useMemo, useState, type FormEvent } from "react";
import axios from "axios";
import { profileApi } from "@/lib/api";

type RiskResult = {
  risk_tolerance: number;
  category: string;
  description: string;
  recommendations: {
    max_position_size: number;
    suggested_stop_loss: number;
    suggested_take_profit: number;
  };
};

const QUESTIONS = [
  "How comfortable are you with short-term portfolio volatility?",
  "What is your preferred holding period for most trades?",
  "How do you react to a 10% drawdown in your portfolio?",
  "What is your primary objective: capital preservation or growth?",
  "How experienced are you with active trading strategies?",
  "How much of your capital can you allocate to higher-risk positions?",
];

const OPTIONS = [
  { value: 1, label: "Very Low" },
  { value: 2, label: "Low" },
  { value: 3, label: "Moderate" },
  { value: 4, label: "High" },
];

export function RiskQuestionnaire() {
  const [answers, setAnswers] = useState<number[]>(Array(QUESTIONS.length).fill(0));
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [result, setResult] = useState<RiskResult | null>(null);

  const answeredCount = useMemo(
    () => answers.filter((value) => value > 0).length,
    [answers],
  );

  const onChangeAnswer = (questionIndex: number, value: number) => {
    const next = [...answers];
    next[questionIndex] = value;
    setAnswers(next);
  };

  const onSubmit = async (event: FormEvent) => {
    event.preventDefault();
    setError(null);
    setResult(null);

    if (answeredCount < QUESTIONS.length) {
      setError("Please answer all questions before submitting.");
      return;
    }

    try {
      setLoading(true);
      const data = await profileApi.submitRiskAssessment(answers);
      setResult(data);
    } catch (error: unknown) {
      if (axios.isAxiosError(error) && typeof error.response?.data?.detail === "string") {
        setError(error.response.data.detail);
      } else {
        setError("Failed to submit assessment.");
      }
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="max-w-4xl mx-auto">
      <div className="mb-6">
        <h1 className="text-3xl font-bold text-white">Risk Assessment</h1>
        <p className="text-gray-400 mt-2">
          Complete the questionnaire to personalize risk-adjusted signals.
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

      <form onSubmit={onSubmit} className="space-y-4">
        {QUESTIONS.map((question, index) => (
          <div key={question} className="p-5 bg-gray-900/50 border border-gray-800 rounded-xl">
            <p className="text-white font-medium mb-4">
              {index + 1}. {question}
            </p>
            <div className="grid grid-cols-2 md:grid-cols-4 gap-2">
              {OPTIONS.map((option) => (
                <button
                  key={option.value}
                  type="button"
                  onClick={() => onChangeAnswer(index, option.value)}
                  className={`px-3 py-2 rounded-lg border text-sm transition-colors ${
                    answers[index] === option.value
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

        {error && (
          <div className="p-3 bg-red-500/10 border border-red-500/30 rounded-lg text-red-400 text-sm">
            {error}
          </div>
        )}

        <button
          type="submit"
          disabled={loading}
          className="w-full py-3 rounded-xl bg-gradient-to-r from-blue-500 to-indigo-500 text-white font-semibold disabled:opacity-50"
        >
          {loading ? "Submitting..." : "Submit Assessment"}
        </button>
      </form>

      {result && (
        <div className="mt-6 p-6 bg-gray-900/50 border border-gray-800 rounded-xl">
          <h2 className="text-xl font-semibold text-white mb-1">{result.category} Profile</h2>
          <p className="text-gray-300 mb-4">{result.description}</p>

          <div className="grid grid-cols-1 md:grid-cols-3 gap-3 text-sm">
            <div className="p-3 bg-gray-800 rounded-lg border border-gray-700">
              <p className="text-gray-400">Risk Score</p>
              <p className="text-white font-semibold">{(result.risk_tolerance * 100).toFixed(0)}%</p>
            </div>
            <div className="p-3 bg-gray-800 rounded-lg border border-gray-700">
              <p className="text-gray-400">Max Position Size</p>
              <p className="text-white font-semibold">
                {(result.recommendations.max_position_size * 100).toFixed(0)}%
              </p>
            </div>
            <div className="p-3 bg-gray-800 rounded-lg border border-gray-700">
              <p className="text-gray-400">Suggested Stop Loss</p>
              <p className="text-white font-semibold">
                {(result.recommendations.suggested_stop_loss * 100).toFixed(0)}%
              </p>
            </div>
          </div>
        </div>
      )}
    </div>
  );
}
