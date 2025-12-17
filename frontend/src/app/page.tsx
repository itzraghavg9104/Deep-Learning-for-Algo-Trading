import Link from "next/link";
import { TrendingUp, Brain, Shield, BarChart3, ArrowRight } from "lucide-react";

export default function HomePage() {
  return (
    <main className="min-h-screen bg-gray-950">
      {/* Hero Section */}
      <div className="relative overflow-hidden">
        {/* Gradient Background */}
        <div className="absolute inset-0 bg-gradient-to-br from-blue-500/10 via-purple-500/10 to-pink-500/10" />
        <div className="absolute top-0 left-1/2 -translate-x-1/2 w-[800px] h-[800px] bg-blue-500/20 rounded-full blur-3xl" />

        <div className="relative max-w-6xl mx-auto px-6 py-24">
          {/* Header */}
          <header className="flex justify-between items-center mb-20">
            <div className="flex items-center gap-3">
              <div className="w-10 h-10 bg-gradient-to-br from-blue-500 to-purple-600 rounded-xl flex items-center justify-center">
                <TrendingUp className="w-6 h-6 text-white" />
              </div>
              <span className="text-xl font-bold text-white">AlgoTrade</span>
            </div>
            <Link
              href="/dashboard"
              className="px-4 py-2 bg-white/10 text-white rounded-lg hover:bg-white/20 transition-all"
            >
              Launch App
            </Link>
          </header>

          {/* Hero Content */}
          <div className="text-center max-w-4xl mx-auto">
            <div className="inline-flex items-center gap-2 px-4 py-2 bg-blue-500/10 border border-blue-500/30 rounded-full text-blue-400 text-sm mb-6">
              <span className="w-2 h-2 bg-green-400 rounded-full animate-pulse" />
              Powered by Deep Reinforcement Learning
            </div>

            <h1 className="text-5xl md:text-7xl font-bold text-white mb-6 leading-tight">
              AI-Powered Trading for{" "}
              <span className="bg-gradient-to-r from-blue-400 via-purple-400 to-pink-400 bg-clip-text text-transparent">
                Indian Markets
              </span>
            </h1>

            <p className="text-xl text-gray-400 mb-10 max-w-2xl mx-auto">
              Algorithmic trading system combining DeepAR probabilistic forecasting
              with PPO reinforcement learning, optimized for NSE/BSE stocks.
            </p>

            <div className="flex flex-col sm:flex-row gap-4 justify-center">
              <Link
                href="/dashboard"
                className="inline-flex items-center justify-center gap-2 px-8 py-4 bg-gradient-to-r from-blue-500 to-purple-600 text-white font-semibold rounded-xl hover:opacity-90 transition-all"
              >
                Get Started
                <ArrowRight className="w-5 h-5" />
              </Link>
              <Link
                href="/profile/risk-assessment"
                className="inline-flex items-center justify-center gap-2 px-8 py-4 bg-white/10 text-white font-semibold rounded-xl border border-white/20 hover:bg-white/20 transition-all"
              >
                Take Risk Assessment
              </Link>
            </div>
          </div>
        </div>
      </div>

      {/* Features Section */}
      <section className="py-24 px-6">
        <div className="max-w-6xl mx-auto">
          <h2 className="text-3xl font-bold text-white text-center mb-4">
            Two-Stage Architecture
          </h2>
          <p className="text-gray-400 text-center mb-16 max-w-2xl mx-auto">
            Mimics the cognitive process of professional traders with prediction and optimization layers.
          </p>

          <div className="grid md:grid-cols-3 gap-6">
            {/* Feature 1 */}
            <div className="p-6 bg-gray-800/30 border border-gray-700 rounded-2xl hover:border-blue-500/50 transition-all group">
              <div className="w-14 h-14 bg-blue-500/20 rounded-xl flex items-center justify-center mb-4 group-hover:scale-110 transition-transform">
                <Brain className="w-7 h-7 text-blue-400" />
              </div>
              <h3 className="text-xl font-semibold text-white mb-2">
                Layer 1: Prediction
              </h3>
              <p className="text-gray-400">
                DeepAR-Attention model for probabilistic price forecasting with 30+ technical indicators.
              </p>
            </div>

            {/* Feature 2 */}
            <div className="p-6 bg-gray-800/30 border border-gray-700 rounded-2xl hover:border-purple-500/50 transition-all group">
              <div className="w-14 h-14 bg-purple-500/20 rounded-xl flex items-center justify-center mb-4 group-hover:scale-110 transition-transform">
                <BarChart3 className="w-7 h-7 text-purple-400" />
              </div>
              <h3 className="text-xl font-semibold text-white mb-2">
                Layer 2: Decision
              </h3>
              <p className="text-gray-400">
                PPO agent optimizing for Sharpe Ratio to make risk-adjusted Buy/Sell/Hold decisions.
              </p>
            </div>

            {/* Feature 3 */}
            <div className="p-6 bg-gray-800/30 border border-gray-700 rounded-2xl hover:border-green-500/50 transition-all group">
              <div className="w-14 h-14 bg-green-500/20 rounded-xl flex items-center justify-center mb-4 group-hover:scale-110 transition-transform">
                <Shield className="w-7 h-7 text-green-400" />
              </div>
              <h3 className="text-xl font-semibold text-white mb-2">
                Trader Behavior
              </h3>
              <p className="text-gray-400">
                Personalized risk tolerance, position sizing, and break-even analysis for each trader.
              </p>
            </div>
          </div>
        </div>
      </section>

      {/* CTA Section */}
      <section className="py-24 px-6">
        <div className="max-w-4xl mx-auto text-center">
          <div className="p-12 bg-gradient-to-br from-blue-500/10 to-purple-500/10 border border-blue-500/20 rounded-3xl">
            <h2 className="text-3xl font-bold text-white mb-4">
              Ready to Start Trading?
            </h2>
            <p className="text-gray-400 mb-8">
              Complete your risk assessment and get personalized trading signals.
            </p>
            <Link
              href="/dashboard"
              className="inline-flex items-center gap-2 px-8 py-4 bg-white text-gray-900 font-semibold rounded-xl hover:bg-gray-100 transition-all"
            >
              Launch Dashboard
              <ArrowRight className="w-5 h-5" />
            </Link>
          </div>
        </div>
      </section>

      {/* Footer */}
      <footer className="py-8 px-6 border-t border-gray-800">
        <div className="max-w-6xl mx-auto flex flex-col md:flex-row justify-between items-center gap-4">
          <div className="text-gray-500 text-sm">
            © 2024 AlgoTrade. Built for Indian Markets (NSE/BSE)
          </div>
          <div className="text-gray-500 text-sm">
            College Major Project
          </div>
        </div>
      </footer>
    </main>
  );
}
