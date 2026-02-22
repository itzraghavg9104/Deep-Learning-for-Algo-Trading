"use client";

import { useEffect, useMemo, useState } from "react";
import axios from "axios";
import Link from "next/link";
import { profileApi } from "@/lib/api";

type ProfileResponse = {
  id: string;
  name: string;
  risk_profile?: {
    tolerance?: number;
    category?: string;
  };
  preferences?: {
    use_sentiment?: boolean;
    preferred_timeframe?: string;
    symbols?: string[];
  };
};

type Preferences = {
  use_sentiment: boolean;
  preferred_timeframe: string;
  symbols: string[];
};

const DEFAULT_PREFERENCES: Preferences = {
  use_sentiment: false,
  preferred_timeframe: "swing",
  symbols: ["RELIANCE.NS", "TCS.NS", "INFY.NS"],
};

export default function ProfilePage() {
  const [profile, setProfile] = useState<ProfileResponse | null>(null);
  const [preferences, setPreferences] = useState<Preferences>(DEFAULT_PREFERENCES);
  const [symbolsInput, setSymbolsInput] = useState(DEFAULT_PREFERENCES.symbols.join(", "));
  const [loading, setLoading] = useState(true);
  const [saving, setSaving] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [success, setSuccess] = useState<string | null>(null);

  const riskPercent = useMemo(() => {
    const tolerance = profile?.risk_profile?.tolerance;
    if (typeof tolerance !== "number") return 50;
    return Math.round(tolerance * 100);
  }, [profile]);

  useEffect(() => {
    const loadProfile = async () => {
      setLoading(true);
      setError(null);

      try {
        const data = (await profileApi.getProfile()) as ProfileResponse;
        setProfile(data);

        const merged = {
          ...DEFAULT_PREFERENCES,
          ...data.preferences,
          symbols: data.preferences?.symbols?.length
            ? data.preferences.symbols
            : DEFAULT_PREFERENCES.symbols,
        };
        setPreferences(merged);
        setSymbolsInput(merged.symbols.join(", "));
      } catch (err: unknown) {
        if (axios.isAxiosError(err) && typeof err.response?.data?.detail === "string") {
          setError(err.response.data.detail);
        } else {
          setError("Failed to load profile.");
        }
      } finally {
        setLoading(false);
      }
    };

    loadProfile();
  }, []);

  const onSavePreferences = async () => {
    setSaving(true);
    setError(null);
    setSuccess(null);

    const symbols = symbolsInput
      .split(",")
      .map((symbol) => symbol.trim().toUpperCase())
      .filter(Boolean);

    const payload = {
      ...preferences,
      symbols,
    };

    try {
      await profileApi.updatePreferences(payload);
      setPreferences(payload);
      setSuccess("Preferences saved.");
    } catch (err: unknown) {
      if (axios.isAxiosError(err) && typeof err.response?.data?.detail === "string") {
        setError(err.response.data.detail);
      } else {
        setError("Failed to save preferences.");
      }
    } finally {
      setSaving(false);
    }
  };

  return (
    <main className="min-h-screen bg-gray-950 p-6">
      <div className="max-w-5xl mx-auto space-y-6">
        <div className="flex items-start justify-between gap-4">
          <div>
            <h1 className="text-3xl font-bold text-white">Profile</h1>
            <p className="text-gray-400 mt-1">Manage risk profile and trading preferences.</p>
          </div>
          <Link
            href="/profile/risk-assessment"
            className="px-4 py-2 rounded-lg bg-blue-500/20 text-blue-300 border border-blue-500/40 hover:bg-blue-500/30"
          >
            Recalculate Risk
          </Link>
        </div>

        {loading ? (
          <div className="p-4 rounded-xl border border-gray-800 bg-gray-900/40 text-gray-300">
            Loading profile...
          </div>
        ) : (
          <>
            <section className="p-5 rounded-xl border border-gray-800 bg-gray-900/40">
              <h2 className="text-lg font-semibold text-white mb-3">User Summary</h2>
              <div className="grid grid-cols-1 md:grid-cols-3 gap-3 text-sm">
                <InfoCard label="User ID" value={profile?.id ?? "N/A"} />
                <InfoCard label="Name" value={profile?.name ?? "N/A"} />
                <InfoCard
                  label="Risk Category"
                  value={profile?.risk_profile?.category ?? "Moderate"}
                />
              </div>

              <div className="mt-4">
                <div className="flex justify-between text-sm text-gray-400 mb-1">
                  <span>Risk Tolerance</span>
                  <span>{riskPercent}%</span>
                </div>
                <div className="h-2 bg-gray-800 rounded-full overflow-hidden">
                  <div
                    className="h-full bg-gradient-to-r from-yellow-500 to-orange-500"
                    style={{ width: `${riskPercent}%` }}
                  />
                </div>
              </div>
            </section>

            <section className="p-5 rounded-xl border border-gray-800 bg-gray-900/40 space-y-4">
              <h2 className="text-lg font-semibold text-white">Preferences</h2>

              <label className="flex items-center gap-2 text-gray-200">
                <input
                  type="checkbox"
                  checked={preferences.use_sentiment}
                  onChange={(event) =>
                    setPreferences((prev) => ({
                      ...prev,
                      use_sentiment: event.target.checked,
                    }))
                  }
                />
                Use sentiment analysis in signal generation
              </label>

              <div>
                <label className="block text-sm text-gray-300 mb-1">Preferred Timeframe</label>
                <select
                  value={preferences.preferred_timeframe}
                  onChange={(event) =>
                    setPreferences((prev) => ({
                      ...prev,
                      preferred_timeframe: event.target.value,
                    }))
                  }
                  className="w-full md:w-64 px-3 py-2 rounded-lg bg-gray-800 border border-gray-700 text-white"
                >
                  <option value="intraday">Intraday</option>
                  <option value="swing">Swing</option>
                  <option value="position">Position</option>
                  <option value="longterm">Long-term</option>
                </select>
              </div>

              <div>
                <label className="block text-sm text-gray-300 mb-1">
                  Watchlist Symbols (comma-separated)
                </label>
                <input
                  value={symbolsInput}
                  onChange={(event) => setSymbolsInput(event.target.value)}
                  className="w-full px-3 py-2 rounded-lg bg-gray-800 border border-gray-700 text-white"
                  placeholder="RELIANCE.NS, TCS.NS, INFY.NS"
                />
              </div>

              {error && (
                <div className="p-3 text-sm rounded-lg border border-red-500/40 bg-red-500/10 text-red-300">
                  {error}
                </div>
              )}

              {success && (
                <div className="p-3 text-sm rounded-lg border border-green-500/40 bg-green-500/10 text-green-300">
                  {success}
                </div>
              )}

              <button
                onClick={onSavePreferences}
                disabled={saving}
                className="px-4 py-2 rounded-lg bg-gradient-to-r from-blue-500 to-indigo-500 text-white font-semibold disabled:opacity-50"
              >
                {saving ? "Saving..." : "Save Preferences"}
              </button>
            </section>
          </>
        )}
      </div>
    </main>
  );
}

function InfoCard({ label, value }: { label: string; value: string }) {
  return (
    <div className="p-3 rounded-lg border border-gray-800 bg-gray-900/50">
      <p className="text-xs text-gray-400">{label}</p>
      <p className="text-base text-white font-semibold">{value}</p>
    </div>
  );
}
