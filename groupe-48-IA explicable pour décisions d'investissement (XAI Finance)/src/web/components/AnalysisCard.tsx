"use client";

interface AnalysisCardProps {
    prediction: string;
    confidence: number;
    baseProb: number;
    currentPrice: number;
}

export default function AnalysisCard({ prediction, confidence, baseProb, currentPrice }: AnalysisCardProps) {
    const isBullish = prediction === "LONG";
    const bgColor = isBullish ? "bg-green-50" : "bg-red-50";
    const textColor = isBullish ? "text-green-700" : "text-red-700";
    const borderColor = isBullish ? "border-green-200" : "border-red-200";

    return (
        <div className={`p-6 rounded-xl border ${borderColor} ${bgColor} shadow-sm`}>
            <div className="flex justify-between items-start mb-4">
                <div>
                    <h2 className="text-sm font-medium text-gray-500 uppercase tracking-wide">Recommendation</h2>
                    <div className={`text-4xl font-bold ${textColor} mt-1`}>{prediction}</div>
                </div>
                <div className="text-right">
                    <h2 className="text-sm font-medium text-gray-500 uppercase tracking-wide">Current Price</h2>
                    <div className="text-2xl font-mono font-semibold text-gray-900">${currentPrice.toFixed(2)}</div>
                </div>
            </div>

            <div className="grid grid-cols-2 gap-4 mt-4 pt-4 border-t border-gray-200/50">
                <div>
                    <div className="text-xs text-gray-500 mb-1">Confidence Model</div>
                    <div className="text-xl font-semibold text-gray-800">{(confidence * 100).toFixed(1)}%</div>
                    <div className="w-full bg-gray-200 h-1.5 rounded-full mt-2">
                        <div
                            className={`h-1.5 rounded-full ${isBullish ? 'bg-green-500' : 'bg-red-500'}`}
                            style={{ width: `${confidence * 100}%` }}
                        />
                    </div>
                </div>
                <div>
                    <div className="text-xs text-gray-500 mb-1">Base Probability</div>
                    <div className="text-xl font-semibold text-gray-800">{baseProb.toFixed(2)}</div>
                </div>
            </div>
        </div>
    );
}
