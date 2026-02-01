"use client";

import { useState } from "react";
import TickerInput from "@/components/TickerInput";
import AnalysisCard from "@/components/AnalysisCard";
import ArgumentList from "@/components/ArgumentList";
import AICommentary from "@/components/AICommentary";
import ShapChart from "@/components/ShapChart";


// Define the shape of our API response
interface AnalysisResult {
  ticker: string;
  date: string;
  prediction: string;
  confidence: number;
  base_probability: number;
  current_price: number;
  bullish_args: any[];
  bearish_args: any[];
  ai_commentary: string;
}

export default function Home() {
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [data, setData] = useState<AnalysisResult | null>(null);

  const handleSearch = async (ticker: string) => {
    setLoading(true);
    setError(null);
    setData(null);

    try {
      const res = await fetch(`http://localhost:8000/analysis/${ticker}`);
      if (!res.ok) {
        const errorData = await res.json();
        throw new Error(errorData.detail || "Failed to fetch analysis");
      }
      const jsonData = await res.json();
      setData(jsonData);
    } catch (err: any) {
      setError(err.message || "An unexpected error occurred");
    } finally {
      setLoading(false);
    }
  };

  return (
    <main className="min-h-screen bg-gray-50 p-8 font-[family-name:var(--font-geist-sans)]">
      <div className="max-w-5xl mx-auto space-y-8">

        {/* Header */}
        <div className="flex flex-col md:flex-row justify-between items-start md:items-center gap-4">
          <div>
            <h1 className="text-3xl font-bold text-gray-900">Market Intelligence</h1>
            <p className="text-gray-500">AI-Powered Financial Argumentation</p>
          </div>
          <TickerInput onSearch={handleSearch} isLoading={loading} />
        </div>

        {/* Error State */}
        {error && (
          <div className="bg-red-50 text-red-700 p-4 rounded-lg border border-red-200">
            {error}
          </div>
        )}

        {/* Content */}
        {data && (
          <div className="space-y-6 animate-in fade-in slide-in-from-bottom-4 duration-500">

            {/* Top Row: AI Verdict & Key Metrics */}
            <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
              <div className="lg:col-span-2">
                <AICommentary text={data.ai_commentary} />
              </div>
              <div className="lg:col-span-1">
                <AnalysisCard
                  prediction={data.prediction}
                  confidence={data.confidence}
                  baseProb={data.base_probability}
                  currentPrice={data.current_price}
                />
              </div>
            </div>

            <div className="border-t border-gray-200 my-8"></div>

            {/* Middle Row: Combat */}
            <div>
              <h2 className="text-xl font-bold text-gray-900 mb-6">Formal Argumentation</h2>
              <ArgumentList
                bullishArgs={data.bullish_args}
                bearishArgs={data.bearish_args}
              />
              <ShapChart
                bullishArgs={data.bullish_args}
                bearishArgs={data.bearish_args}
              />
            </div>

            <div className="text-center text-xs text-gray-400 mt-12">
              Analysis based on data from {data.date}. Not financial advice.
            </div>

          </div>
        )}

        {/* Empty State */}
        {!data && !loading && !error && (
          <div className="text-center py-20 text-gray-400">
            Type a ticker above to start the analysis.
          </div>
        )}
      </div>
    </main>
  );
}
