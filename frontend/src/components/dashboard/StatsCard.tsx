"use client";

import { ArrowUp, ArrowDown, Activity, Target } from "lucide-react";

interface StatsCardProps {
    title: string;
    value: string;
    change?: number;
    icon: "up" | "down" | "activity" | "target";
    color: "green" | "red" | "blue" | "purple";
}

export function StatsCard({ title, value, change, icon, color }: StatsCardProps) {
    const colorClasses = {
        green: "from-green-500/20 to-emerald-500/20 border-green-500/30",
        red: "from-red-500/20 to-rose-500/20 border-red-500/30",
        blue: "from-blue-500/20 to-cyan-500/20 border-blue-500/30",
        purple: "from-purple-500/20 to-pink-500/20 border-purple-500/30",
    };

    const iconClasses = {
        green: "text-green-400",
        red: "text-red-400",
        blue: "text-blue-400",
        purple: "text-purple-400",
    };

    const icons = {
        up: ArrowUp,
        down: ArrowDown,
        activity: Activity,
        target: Target,
    };

    const Icon = icons[icon];

    return (
        <div
            className={`bg-gradient-to-br ${colorClasses[color]} backdrop-blur-sm border rounded-xl p-5`}
        >
            <div className="flex justify-between items-start">
                <div>
                    <p className="text-sm text-gray-400 mb-1">{title}</p>
                    <p className="text-2xl font-bold text-white">{value}</p>
                    {change !== undefined && (
                        <p
                            className={`text-sm mt-1 ${change >= 0 ? "text-green-400" : "text-red-400"}`}
                        >
                            {change >= 0 ? "+" : ""}
                            {change.toFixed(2)}%
                        </p>
                    )}
                </div>
                <div
                    className={`w-12 h-12 rounded-lg bg-gray-800/50 flex items-center justify-center ${iconClasses[color]}`}
                >
                    <Icon className="w-6 h-6" />
                </div>
            </div>
        </div>
    );
}
