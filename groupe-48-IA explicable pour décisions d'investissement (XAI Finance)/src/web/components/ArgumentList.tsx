"use client";

interface Argument {
    feature: string;
    value: number;
    shap: number;
    text: string;
}

interface ArgumentListProps {
    bullishArgs: Argument[];
    bearishArgs: Argument[];
}

export default function ArgumentList({ bullishArgs, bearishArgs }: ArgumentListProps) {
    return (
        <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
            {/* Bearish Column */}
            <div className="bg-red-50/50 p-6 rounded-xl border border-red-100">
                <h3 className="flex items-center gap-2 text-lg font-bold text-red-800 mb-4">
                    <span>🐻</span> Bearish Drivers
                </h3>
                <div className="space-y-3">
                    {bearishArgs.length === 0 && <p className="text-gray-500 italic">No strong bearish signals.</p>}
                    {bearishArgs.map((arg, idx) => (
                        <div key={idx} className="bg-white p-3 rounded-lg shadow-sm border border-red-100">
                            <div className="text-sm font-semibold text-gray-800">{arg.text}</div>
                            <div className="flex justify-between mt-1 text-xs text-gray-500">
                                <span>{arg.feature}</span>
                                <span>Impact: {arg.shap.toFixed(3)}</span>
                            </div>
                        </div>
                    ))}
                </div>
            </div>

            {/* Bullish Column */}
            <div className="bg-green-50/50 p-6 rounded-xl border border-green-100">
                <h3 className="flex items-center gap-2 text-lg font-bold text-green-800 mb-4">
                    <span>qh</span> Bullish Drivers
                </h3>
                <div className="space-y-3">
                    {bullishArgs.length === 0 && <p className="text-gray-500 italic">No strong bullish signals.</p>}
                    {bullishArgs.map((arg, idx) => (
                        <div key={idx} className="bg-white p-3 rounded-lg shadow-sm border border-green-100">
                            <div className="text-sm font-semibold text-gray-800">{arg.text}</div>
                            <div className="flex justify-between mt-1 text-xs text-gray-500">
                                <span>{arg.feature}</span>
                                <span>Impact: {arg.shap.toFixed(3)}</span>
                            </div>
                        </div>
                    ))}
                </div>
            </div>
        </div>
    );
}
