"use client";

import { useEffect, useMemo, useState } from "react";
import { tradesApi } from "@/lib/api";

type Trade = {
  id: string;
  date: string;
  symbol: string;
  action: "BUY" | "SELL";
  quantity: number;
  price: number;
  pnl: number;
};

export default function TradesPage() {
  const [trades, setTrades] = useState<Trade[]>([]);
  const [totalPnl, setTotalPnl] = useState(0);
  const [winRate, setWinRate] = useState(0);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [symbolFilter, setSymbolFilter] = useState("");
  const [fromDate, setFromDate] = useState("");
  const [toDate, setToDate] = useState("");

  useEffect(() => {
    tradesApi
      .getTradeHistory()
      .then((data) => {
        setTrades(data.trades ?? []);
        setTotalPnl(data.total_pnl ?? 0);
        setWinRate(data.win_rate ?? 0);
      })
      .catch(() => setError("Failed to load trade history"))
      .finally(() => setLoading(false));
  }, []);

  const filtered = useMemo(() => {
    return trades.filter((trade) => {
      const symbolOk = symbolFilter
        ? trade.symbol.toLowerCase().includes(symbolFilter.toLowerCase())
        : true;
      const fromOk = fromDate ? trade.date >= fromDate : true;
      const toOk = toDate ? trade.date <= toDate : true;
      return symbolOk && fromOk && toOk;
    });
  }, [trades, symbolFilter, fromDate, toDate]);

  const filteredPnl = useMemo(
    () => filtered.reduce((sum, trade) => sum + trade.pnl, 0),
    [filtered],
  );

  const filteredWinRate = useMemo(
    () =>
      filtered.length
        ? ((filtered.filter((t) => t.pnl > 0).length / filtered.length) * 100).toFixed(0)
        : "0",
    [filtered],
  );

  const exportCsv = () => {
    const header = "id,date,symbol,action,quantity,price,pnl";
    const rows = filtered.map((trade) =>
      [trade.id, trade.date, trade.symbol, trade.action, trade.quantity, trade.price, trade.pnl].join(","),
    );
    const blob = new Blob([[header, ...rows].join("\n")], { type: "text/csv;charset=utf-8;" });
    const url = URL.createObjectURL(blob);
    const link = document.createElement("a");
    link.href = url;
    link.download = "trade-history.csv";
    link.click();
    URL.revokeObjectURL(url);
  };

  if (loading) {
    return (
      <main className="min-h-screen bg-gray-950 p-6">
        <div className="max-w-6xl mx-auto text-center text-gray-400 pt-20">Loading trade history...</div>
      </main>
    );
  }

  if (error) {
    return (
      <main className="min-h-screen bg-gray-950 p-6">
        <div className="max-w-6xl mx-auto text-center text-red-400 pt-20">{error}</div>
      </main>
    );
  }

  return (
    <main className="min-h-screen bg-gray-950 p-6">
      <div className="max-w-6xl mx-auto space-y-5">
        <div>
          <h1 className="text-3xl font-bold text-white">Trade History</h1>
          <p className="text-gray-400 mt-1">
            {trades.length} trades found &middot; Total P&L: INR {totalPnl.toFixed(2)} &middot; Win Rate: {(winRate * 100).toFixed(0)}%
          </p>
        </div>

        <section className="p-4 rounded-xl border border-gray-800 bg-gray-900/40">
          <div className="grid grid-cols-1 md:grid-cols-4 gap-3">
            <input
              value={symbolFilter}
              onChange={(event) => setSymbolFilter(event.target.value)}
              placeholder="Filter symbol (e.g. TCS)"
              className="px-3 py-2 rounded-lg bg-gray-800 border border-gray-700 text-white"
            />
            <input
              type="date"
              value={fromDate}
              onChange={(event) => setFromDate(event.target.value)}
              className="px-3 py-2 rounded-lg bg-gray-800 border border-gray-700 text-white"
            />
            <input
              type="date"
              value={toDate}
              onChange={(event) => setToDate(event.target.value)}
              className="px-3 py-2 rounded-lg bg-gray-800 border border-gray-700 text-white"
            />
            <button
              onClick={exportCsv}
              className="px-3 py-2 rounded-lg bg-blue-500/20 text-blue-300 border border-blue-500/40 hover:bg-blue-500/30"
            >
              Export CSV
            </button>
          </div>
        </section>

        <section className="grid grid-cols-1 md:grid-cols-3 gap-3">
          <StatCard label="Filtered Trades" value={filtered.length.toString()} />
          <StatCard label="Filtered P&L" value={`INR ${filteredPnl.toFixed(2)}`} positive={filteredPnl >= 0} />
          <StatCard label="Win Rate" value={`${filteredWinRate}%`} />
        </section>

        {filtered.length === 0 ? (
          <p className="text-gray-500 text-center py-10">No trades match your filters.</p>
        ) : (
          <section className="p-4 rounded-xl border border-gray-800 bg-gray-900/40 overflow-auto">
            <table className="w-full text-sm">
              <thead>
                <tr className="text-left text-gray-400 border-b border-gray-800">
                  <th className="py-2 pr-3">Date</th>
                  <th className="py-2 pr-3">Symbol</th>
                  <th className="py-2 pr-3">Action</th>
                  <th className="py-2 pr-3">Qty</th>
                  <th className="py-2 pr-3">Price</th>
                  <th className="py-2 pr-3">P&L</th>
                </tr>
              </thead>
              <tbody>
                {filtered.map((trade) => (
                  <tr key={trade.id} className="border-b border-gray-900">
                    <td className="py-2 pr-3 text-gray-200">{trade.date}</td>
                    <td className="py-2 pr-3 text-gray-200">{trade.symbol}</td>
                    <td className={`py-2 pr-3 ${trade.action === "BUY" ? "text-green-300" : "text-red-300"}`}>
                      {trade.action}
                    </td>
                    <td className="py-2 pr-3 text-gray-200">{trade.quantity}</td>
                    <td className="py-2 pr-3 text-gray-200">{trade.price.toFixed(2)}</td>
                    <td className={`py-2 pr-3 ${trade.pnl >= 0 ? "text-green-300" : "text-red-300"}`}>
                      {trade.pnl.toFixed(2)}
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </section>
        )}
      </div>
    </main>
  );
}

function StatCard({ label, value, positive }: { label: string; value: string; positive?: boolean }) {
  return (
    <div className="p-3 rounded-lg border border-gray-800 bg-gray-900/40">
      <p className="text-xs text-gray-400">{label}</p>
      <p className={`text-lg font-semibold ${positive === undefined ? "text-white" : positive ? "text-green-300" : "text-red-300"}`}>
        {value}
      </p>
    </div>
  );
}
