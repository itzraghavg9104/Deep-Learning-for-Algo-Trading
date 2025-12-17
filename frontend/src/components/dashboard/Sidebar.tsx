"use client";

import Link from "next/link";
import { usePathname } from "next/navigation";
import {
    BarChart3,
    LineChart,
    User,
    Settings,
    TrendingUp,
    History,
} from "lucide-react";

const navItems = [
    { href: "/dashboard", label: "Dashboard", icon: BarChart3 },
    { href: "/dashboard/signals", label: "Signals", icon: TrendingUp },
    { href: "/dashboard/backtest", label: "Backtest", icon: History },
    { href: "/dashboard/analysis", label: "Analysis", icon: LineChart },
    { href: "/profile", label: "Profile", icon: User },
    { href: "/settings", label: "Settings", icon: Settings },
];

export function Sidebar() {
    const pathname = usePathname();

    return (
        <aside className="w-64 bg-gray-900/50 backdrop-blur-xl border-r border-gray-800 min-h-screen p-4">
            {/* Logo */}
            <div className="flex items-center gap-3 px-2 mb-8">
                <div className="w-10 h-10 bg-gradient-to-br from-blue-500 to-purple-600 rounded-xl flex items-center justify-center">
                    <TrendingUp className="w-6 h-6 text-white" />
                </div>
                <div>
                    <h1 className="text-lg font-bold text-white">AlgoTrade</h1>
                    <p className="text-xs text-gray-500">NSE/BSE</p>
                </div>
            </div>

            {/* Navigation */}
            <nav className="space-y-1">
                {navItems.map((item) => {
                    const isActive = pathname === item.href;
                    const Icon = item.icon;

                    return (
                        <Link
                            key={item.href}
                            href={item.href}
                            className={`flex items-center gap-3 px-3 py-2.5 rounded-lg transition-all ${isActive
                                    ? "bg-blue-500/20 text-blue-400 border border-blue-500/30"
                                    : "text-gray-400 hover:bg-gray-800 hover:text-white"
                                }`}
                        >
                            <Icon className="w-5 h-5" />
                            <span className="font-medium">{item.label}</span>
                        </Link>
                    );
                })}
            </nav>

            {/* Risk Profile Card */}
            <div className="mt-8 p-4 bg-gradient-to-br from-purple-500/10 to-blue-500/10 border border-purple-500/20 rounded-xl">
                <p className="text-xs text-gray-400 mb-1">Risk Profile</p>
                <p className="text-lg font-semibold text-white">Moderate</p>
                <div className="mt-2 h-2 bg-gray-700 rounded-full overflow-hidden">
                    <div className="h-full w-1/2 bg-gradient-to-r from-yellow-500 to-orange-500 rounded-full" />
                </div>
                <p className="text-xs text-gray-500 mt-2">50% Risk Tolerance</p>
            </div>
        </aside>
    );
}
