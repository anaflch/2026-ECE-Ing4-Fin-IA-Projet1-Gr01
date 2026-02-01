"use client";

import { BarChart, Bar, XAxis, YAxis, Tooltip, ResponsiveContainer, Cell } from 'recharts';

interface Argument {
    feature: string;
    shap: number;
}

interface ShapChartProps {
    bullishArgs: Argument[];
    bearishArgs: Argument[];
}

export default function ShapChart({ bullishArgs, bearishArgs }: ShapChartProps) {
    // Combine and sort by magnitude
    const data = [...bullishArgs, ...bearishArgs]
        .sort((a, b) => Math.abs(b.shap) - Math.abs(a.shap))
        .slice(0, 10); // Top 10

    return (
        <div className="bg-white p-6 rounded-xl border border-gray-200 shadow-sm mt-6">
            <h3 className="text-lg font-bold text-gray-900 mb-4">Top Feature Impacts (SHAP)</h3>
            <div className="h-[300px] w-full">
                <ResponsiveContainer width="100%" height="100%">
                    <BarChart
                        layout="vertical"
                        data={data}
                        margin={{ top: 5, right: 30, left: 20, bottom: 5 }}
                    >
                        <XAxis type="number" hide />
                        <YAxis
                            dataKey="feature"
                            type="category"
                            width={100}
                            tick={{ fontSize: 12 }}
                        />
                        <Tooltip
                            cursor={{ fill: 'transparent' }}
                            content={({ active, payload }) => {
                                if (active && payload && payload.length) {
                                    const d = payload[0].payload;
                                    return (
                                        <div className="bg-white border p-2 shadow-lg rounded text-sm">
                                            <div className="font-bold">{d.feature}</div>
                                            <div>Impact: {d.shap.toFixed(4)}</div>
                                        </div>
                                    );
                                }
                                return null;
                            }}
                        />
                        <Bar dataKey="shap">
                            {data.map((entry, index) => (
                                <Cell key={`cell-${index}`} fill={entry.shap > 0 ? '#22c55e' : '#ef4444'} />
                            ))}
                        </Bar>
                    </BarChart>
                </ResponsiveContainer>
            </div>
        </div>
    );
}
