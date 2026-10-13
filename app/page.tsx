"use client";

import dynamic from "next/dynamic";
import { motion } from "framer-motion";
import { Activity, ArrowDown, ArrowRight, Check, CircleHelp, FlaskConical, Github, LoaderCircle, RotateCcw, ScanSearch, ShieldCheck, Sparkles, WandSparkles } from "lucide-react";
import { useEffect, useState } from "react";

const HeroScene = dynamic(() => import("./components/hero-scene"), {
  ssr: false,
  loading: () => <div className="scene-loading"><div className="loading-orb" /></div>,
});

type AspectResult = {
  aspect: string;
  sentiment: "positive" | "negative" | "unknown" | string;
  score: number;
  evidence: string[];
  method: string;
};
type AnalysisResult = {
  input_text: string;
  aspects: AspectResult[];
  aspect_method: string;
  aspect_method_is_heuristic: boolean;
  review_sentiment: string | null;
  review_model_available: boolean;
  review_model_kind: string | null;
  message: string;
};
const SAMPLE = "The coffee was excellent, but the packaging was terrible. The delivery was quick.";

function sentimentClass(value: string) {
  if (value === "positive") return "positive";
  if (value === "negative") return "negative";
  return "unknown";
}

export default function Home() {
  const [review, setReview] = useState("");
  const [aspectInput, setAspectInput] = useState("");
  const [result, setResult] = useState<AnalysisResult | null>(null);
  const [error, setError] = useState("");
  const [loading, setLoading] = useState(false);
  const [backendStatus, setBackendStatus] = useState<"checking" | "ready" | "warming" | "unavailable">("checking");

  useEffect(() => {
    let active = true;
    async function warmBackend() {
      try {
        const response = await fetch("/api/health", { cache: "no-store" });
        const data = await response.json().catch(() => ({}));
        if (active) setBackendStatus(response.ok && data.status === "ready" ? "ready" : data.status === "warming" ? "warming" : "unavailable");
      } catch {
        if (active) setBackendStatus("unavailable");
      }
    }
    void warmBackend();
    return () => { active = false; };
  }, []);

  async function analyze() {
    setError("");
    setResult(null);
    if (!review.trim()) {
      setError("Add a review first so the model has something to analyze.");
      return;
    }
    setLoading(true);
    try {
      const response = await fetch("/api/analyze", {
        method: "POST",
        headers: { "content-type": "application/json" },
        body: JSON.stringify({
          text: review.trim(),
          aspects: aspectInput.split(",").map((item) => item.trim()).filter(Boolean),
        }),
      });
      const data = await response.json();
      if (!response.ok) {
        setError(typeof data.detail === "string" ? data.detail : data.error || "Analysis failed. Check the inference service.");
        return;
      }
      setResult(data as AnalysisResult);
    } catch {
      setError("The analysis request failed. Please check your connection and try again.");
    } finally {
      setLoading(false);
    }
  }

  const positiveCount = result?.aspects.filter((item) => item.sentiment === "positive").length ?? 0;
  const negativeCount = result?.aspects.filter((item) => item.sentiment === "negative").length ?? 0;
  const unknownCount = result?.aspects.filter((item) => item.sentiment === "unknown").length ?? 0;

  return (
    <main>
      <nav className="topbar">
        <a className="brand" href="#" aria-label="Aspectly home">
          <span className="brand-mark"><ScanSearch size={21} strokeWidth={2.4} /></span>
          <span>aspectly<span className="brand-dot">.</span></span>
        </a>
        <div className="nav-links"><a href="#how-it-works">How it works</a><a href="#analyzer">Analyzer</a></div>
        <a className="nav-github" href="https://github.com/017Mehul/aspect-based-sentiment-analysis" target="_blank" rel="noreferrer">
          <Github size={16} /> <span>Open source</span><ArrowUpRightIcon />
        </a>
      </nav>

      <section className="hero section-wrap">
        <div className="hero-copy">
          <motion.div className="eyebrow" initial={{ opacity: 0, y: 10 }} animate={{ opacity: 1, y: 0 }} transition={{ duration: 0.55 }}>
            <span className="eyebrow-pulse" /> NLP RESEARCH · MADE INTERACTIVE
          </motion.div>
          <motion.h1 initial={{ opacity: 0, y: 16 }} animate={{ opacity: 1, y: 0 }} transition={{ duration: 0.65, delay: 0.08 }}>
            Every review has a <span className="headline-highlight">point of view.</span>
          </motion.h1>
          <motion.p className="hero-description" initial={{ opacity: 0, y: 12 }} animate={{ opacity: 1, y: 0 }} transition={{ duration: 0.6, delay: 0.18 }}>
            Go beyond a single sentiment score. Explore what people feel about the details that matter — one aspect at a time.
          </motion.p>
          <motion.div className="hero-actions" initial={{ opacity: 0, y: 10 }} animate={{ opacity: 1, y: 0 }} transition={{ delay: 0.28 }}>
            <a className="button-primary" href="#analyzer">Analyze a review <ArrowDown size={17} /></a>
            <a className="button-secondary" href="https://github.com/017Mehul/aspect-based-sentiment-analysis" target="_blank" rel="noreferrer">Explore the research <ArrowRight size={16} /></a>
          </motion.div>
          <div className="hero-trust"><span><ShieldCheck size={15} /> Transparent methods</span><span><FlaskConical size={15} /> Python ML pipeline</span><span><Activity size={15} /> Live inference</span></div>
        </div>
        <div className="hero-visual"><HeroScene /><div className="visual-caption"><span className="caption-dot" /> SENTIMENT, WITH CONTEXT</div></div>
        <div className="hero-bottom-line"><span>UNDERSTAND THE WHY, NOT JUST THE SCORE</span><span>01 — ASPECT INTELLIGENCE</span></div>
      </section>

      <section className="analyzer-section" id="analyzer">
        <div className="section-wrap analyzer-wrap">
          <div className="section-heading">
            <div><div className="section-kicker">THE WORKSPACE <span>01 / 03</span></div><h2>Put a review under the lens.</h2></div>
            <p>Paste real customer feedback. The connected Python service will analyze it using the project’s existing pipeline.</p>
          </div>
          <div className="workspace-grid">
            <div className="input-panel panel">
              <div className="panel-top"><span className="panel-index">A</span><div><h3>Review input</h3><p>What did the customer say?</p></div><span className="live-tag"><span /> INPUT</span></div>
              <label className="field-label" htmlFor="review">CUSTOMER REVIEW</label>
              <textarea id="review" value={review} onChange={(event) => setReview(event.target.value)} maxLength={4000} placeholder="e.g. The coffee was excellent, but the packaging was terrible..." />
              <div className="input-meta"><span>{review.length.toLocaleString()} / 4,000 characters</span><button className="text-button" type="button" onClick={() => { setReview(SAMPLE); setAspectInput("coffee, packaging, delivery"); setError(""); setResult(null); }}>Use sample review <ArrowRight size={14} /></button></div>
              <label className="field-label aspect-label" htmlFor="aspects">ASPECTS TO CHECK <span>OPTIONAL</span></label>
              <input id="aspects" value={aspectInput} onChange={(event) => setAspectInput(event.target.value)} placeholder="coffee, packaging, delivery" />
              <p className="field-hint"><CircleHelp size={14} /> Add comma-separated aspect terms, or leave blank to use spaCy extraction.</p>
              <button className="analyze-button" type="button" onClick={analyze} disabled={loading || !review.trim() || backendStatus !== "ready"}>
                {loading ? <><LoaderCircle className="spin" size={18} /> Analyzing with Python…</> : <><WandSparkles size={18} /> Analyze sentiment <ArrowRight size={17} /></>}
              </button>
              {error && <div className="error-box" role="alert"><CircleHelp size={17} /><span>{error}</span></div>}
              {backendStatus !== "ready" && <div className="error-box" role="status"><CircleHelp size={17} /><span>{backendStatus === "checking" || backendStatus === "warming" ? "Waking the Python service on Render. The first connection can take a little while; this page checks it automatically." : "The Python service is not responding yet. Retry the connection after a short wait."}</span>{backendStatus === "unavailable" && <button className="text-button" type="button" onClick={() => window.location.reload()}>Retry</button>}</div>}
              <div className="privacy-note"><ShieldCheck size={15} /> Results are returned by your configured inference service; this interface does not invent predictions.</div>
            </div>

            <div className="results-panel panel">
              <div className="panel-top"><span className="panel-index panel-index-dark">B</span><div><h3>Aspect breakdown</h3><p>Results from the live inference endpoint</p></div><span className={result ? "status-tag status-ready" : "status-tag"}><span /> {result ? "ANALYZED" : "AWAITING INPUT"}</span></div>
              {!result ? (
                <div className="empty-results"><div className="empty-graphic"><div className="empty-ring ring-one" /><div className="empty-ring ring-two" /><ScanSearch size={35} strokeWidth={1.25} /></div><h4>Your insights will appear here.</h4><p>Run an analysis to see aspect-level polarity, supporting words, and any available trained review-level prediction.</p><div className="empty-legend"><span><i className="legend-positive" /> Positive</span><span><i className="legend-negative" /> Negative</span><span><i className="legend-unknown" /> Unknown</span></div></div>
              ) : (
                <div className="result-content">
                  <div className="result-summary">
                    <div><span className="summary-label">ASPECTS RETURNED</span><strong>{result.aspects.length}</strong></div>
                    <div><span className="summary-label">REVIEW MODEL</span><strong className="summary-model">{result.review_model_available ? result.review_sentiment : "Not trained"}</strong></div>
                  </div>
                  <div className="sentiment-counts">
                    <div><span className="count-dot positive-dot" /><strong>{positiveCount}</strong><span>Positive</span></div>
                    <div><span className="count-dot negative-dot" /><strong>{negativeCount}</strong><span>Negative</span></div>
                    <div><span className="count-dot unknown-dot" /><strong>{unknownCount}</strong><span>Unknown</span></div>
                  </div>
                  <div className="aspect-list">
                    {result.aspects.map((item, index) => (
                      <motion.div className="aspect-row" key={item.aspect + index} initial={{ opacity: 0, x: 10 }} animate={{ opacity: 1, x: 0 }} transition={{ delay: index * 0.08 }}>
                        <div className="aspect-row-main"><span className="aspect-number">{String(index + 1).padStart(2, "0")}</span><div className="aspect-name-wrap"><strong>{item.aspect}</strong><span>{item.evidence.length ? `Evidence: ${item.evidence.join(", ")}` : "No lexicon evidence found"}</span></div><span className={`sentiment-pill ${sentimentClass(item.sentiment)}`}>{item.sentiment}</span></div>
                      </motion.div>
                    ))}
                    {!result.aspects.length && <p className="no-aspects">No aspects were detected. Try adding aspect terms in the input panel.</p>}
                  </div>
                  <div className="method-note"><FlaskConical size={16} /><div><strong>Method transparency</strong><p>{result.message}</p><small>{result.aspect_method}{result.aspect_method_is_heuristic ? " · heuristic, not a supervised ABSA model" : ""}</small></div></div>
                  {result.review_model_available && <div className="model-note"><Check size={16} /> Review-level prediction supplied by a trained scikit-learn pipeline.</div>}
                </div>
              )}
            </div>
          </div>
          <div className="workspace-foot"><span><span className="tiny-live-dot" /> INFERENCE STATUS: {loading ? "RUNNING" : result ? "COMPLETE" : backendStatus === "ready" ? "READY" : backendStatus === "checking" || backendStatus === "warming" ? "WAKING SERVICE" : "UNAVAILABLE"}</span><span>NO FABRICATED METRICS · NO PLACEHOLDER PREDICTIONS</span></div>
        </div>
      </section>

      <section className="how-section section-wrap" id="how-it-works">
        <div className="section-heading">
          <div><div className="section-kicker">THE PIPELINE <span>02 / 03</span></div><h2>From raw text to useful signals.</h2></div>
          <p>One interface. An explicit inference boundary. A Python pipeline that remains the source of truth.</p>
        </div>
        <div className="steps-grid">
          <motion.article className="step-card" whileHover={{ y: -5 }} transition={{ duration: 0.2 }}><div className="step-top"><span>01</span><div className="step-icon"><ScanSearch size={20} /></div></div><h3>Extract aspects</h3><p>Use supplied aspect terms or the existing spaCy noun-phrase extraction path when the language model is installed.</p><div className="step-line"><span style={{ width: "36%" }} /></div></motion.article>
          <motion.article className="step-card" whileHover={{ y: -5 }} transition={{ duration: 0.2 }}><div className="step-top"><span>02</span><div className="step-icon"><Activity size={20} /></div></div><h3>Estimate polarity</h3><p>The current aspect demo uses a transparent proximity lexicon. Its heuristic status is always shown with results.</p><div className="step-line"><span style={{ width: "67%" }} /></div></motion.article>
          <motion.article className="step-card" whileHover={{ y: -5 }} transition={{ duration: 0.2 }}><div className="step-top"><span>03</span><div className="step-icon"><Sparkles size={20} /></div></div><h3>Report honestly</h3><p>If a trained review-level artifact exists, the API returns its prediction. Otherwise, the UI says it is unavailable.</p><div className="step-line"><span style={{ width: "100%" }} /></div></motion.article>
        </div>
        <div className="research-note"><div className="research-note-icon"><FlaskConical size={20} /></div><div><strong>Research-grade transparency</strong><p>Review-level star-rating labels do not validate aspect-level sentiment. This UI keeps those tasks separate and does not display benchmark figures as live predictions.</p></div><a href="https://github.com/017Mehul/aspect-based-sentiment-analysis/blob/audit/fix-absa-pipeline/EXPERIMENT_RESULTS.md" target="_blank" rel="noreferrer">Read experiment notes <ArrowRight size={15} /></a></div>
      </section>

      <footer className="footer"><a className="brand" href="#"><span className="brand-mark"><ScanSearch size={18} /></span><span>aspectly<span className="brand-dot">.</span></span></a><span>Built for clearer sentiment, not bigger claims.</span><a href="https://github.com/017Mehul/aspect-based-sentiment-analysis" target="_blank" rel="noreferrer">GitHub <ArrowRight size={14} /></a></footer>
    </main>
  );
}

function ArrowUpRightIcon() {
  return <ArrowRight size={14} className="arrow-up-right" />;
}
