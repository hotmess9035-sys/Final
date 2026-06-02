import express from "express";
import cors from "cors";
import dotenv from "dotenv";
import fs from "fs";
import path from "path";
import { fileURLToPath } from "url";
import OpenAI from "openai";

dotenv.config();

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);
const app = express();

const PORT = process.env.PORT || 3000;
const MODEL = process.env.OPENAI_MODEL || "gpt-5.5";
const MEMORY_PATH = path.join(__dirname, "data", "predictions.json");
const PROJECTS_PATH = path.join(__dirname, "data", "projects.json");
const SKILLS_DIR = path.join(__dirname, "skills");

app.use(cors());
app.use(express.json({ limit: "5mb" }));
app.use(express.static(path.join(__dirname, "public")));

const AGENTS = [
  {
    id: "historian",
    name: "Historian / REMEMBERS",
    purpose: "Compare current signals to stored predictions, repeated cycles, and historical patterns.",
    bias: "memory, cycles, precedent, what came before"
  },
  {
    id: "skeptic",
    name: "Skeptic / SELFAWARE",
    purpose: "Attack the forecast, detect overconfidence, missing data, and weak assumptions.",
    bias: "self-correction, calibration, blind spots"
  },
  {
    id: "rebel",
    name: "Rebel / REBEL",
    purpose: "Find contrarian futures, black swans, and ignored opportunities.",
    bias: "anti-consensus, leverage, power dynamics"
  },
  {
    id: "empath",
    name: "Empath / EMPATHY",
    purpose: "Model human reaction, customer behavior, fear, desire, trust, and social adoption.",
    bias: "emotion, incentives, buying behavior, pain points"
  },
  {
    id: "systems",
    name: "Systems Thinker / UNITY + ENTANGLEMENT",
    purpose: "Find hidden relationships across domains and chain reactions.",
    bias: "interdependence, convergence, network effects"
  },
  {
    id: "explorer",
    name: "Explorer / MANY WORLDS + SUPERPOSITION",
    purpose: "Generate multiple futures and keep them alive as probability clouds.",
    bias: "alternative timelines, possibility space"
  },
  {
    id: "operator",
    name: "Operator / PHENOMENON",
    purpose: "Turn forecast into next action, project ranking, and practical execution.",
    bias: "action, resources, constraints, income"
  }
];

function readJson(file, fallback) {
  try {
    if (!fs.existsSync(file)) return fallback;
    return JSON.parse(fs.readFileSync(file, "utf8"));
  } catch {
    return fallback;
  }
}

function writeJson(file, data) {
  fs.writeFileSync(file, JSON.stringify(data, null, 2));
}

function readMemory() {
  return readJson(MEMORY_PATH, []);
}

function readProjects() {
  return readJson(PROJECTS_PATH, []);
}

function getSkillFiles() {
  if (!fs.existsSync(SKILLS_DIR)) return [];
  return fs.readdirSync(SKILLS_DIR)
    .filter(f => /\.(yaml|yml|skill|txt)$/i.test(f))
    .map(f => {
      const p = path.join(SKILLS_DIR, f);
      const raw = fs.readFileSync(p, "utf8");
      const clipped = raw.slice(0, 4200);
      let name = f.replace(/\.(yaml|yml|skill|txt)$/i, "");
      const m = raw.match(/name:\s*["']?([^"'\n]+)/i);
      if (m) name = m[1].trim();
      return { file: f, name, excerpt: clipped };
    });
}

function clamp(n) {
  return Math.max(1, Math.min(99, Math.round(Number(n) || 0)));
}

function textBlob(input) {
  return JSON.stringify(input || {}).toLowerCase();
}

function has(blob, terms) {
  return terms.some(t => blob.includes(t));
}

function heuristicScores(input) {
  const s = input.sliders || {};
  const money = Number(s.money ?? 50);
  const opportunity = Number(s.opportunity ?? 50);
  const chaos = Number(s.chaos ?? 50);
  const action = Number(s.action ?? 50);
  const data = Number(s.data ?? 50);
  const skills = Number(s.skills ?? 50);
  const blob = textBlob(input);

  const tech = has(blob, ["ai", "automation", "software", "agent", "quantum", "app", "saas"]) ? 14 : 0;
  const customers = has(blob, ["customer", "support", "service", "client", "sales", "buyer"]) ? 12 : 0;
  const creative = has(blob, ["horror", "film", "story", "content", "ufo", "strange", "video"]) ? 10 : 0;
  const urgency = has(blob, ["bill", "debt", "urgent", "money fast", "broke", "need money", "pressure"]) ? 15 : 0;
  const world = has(blob, ["politic", "war", "election", "inflation", "market", "recession", "weather"]) ? 8 : 0;

  const confidence = clamp(data * 0.44 + skills * 0.22 + action * 0.21 - chaos * 0.18);
  const income = clamp(opportunity * 0.33 + action * 0.28 + skills * 0.18 + tech + customers + urgency * 0.3 - chaos * 0.08);
  const volatility = clamp(chaos * 0.52 + urgency * 0.85 + money * 0.18 + world - data * 0.08);
  const creativeMarket = clamp(creative * 3 + tech + opportunity * 0.24 + action * 0.15);
  const execution = clamp(action * 0.36 + skills * 0.24 + opportunity * 0.18 - chaos * 0.14 - money * 0.05);
  const unknown = clamp(100 - (data * 0.56 + skills * 0.17));

  return { confidence, income, volatility, creativeMarket, execution, unknown };
}

function projectRankings(projects, scores) {
  return projects.map(p => {
    const opportunityScore =
      (p.revenuePotential * 0.30) +
      (p.timeToRevenue * 0.22) +
      (p.traction * 0.15) +
      (scores.income * 0.17) +
      (scores.execution * 0.16) -
      (p.risk * 0.18) -
      (p.effort * 0.08);

    return {
      ...p,
      score: clamp(opportunityScore),
      recommendation:
        opportunityScore > 70 ? "Push now" :
        opportunityScore > 55 ? "Prototype and test" :
        opportunityScore > 40 ? "Hold as secondary" :
        "Do not prioritize yet"
    };
  }).sort((a, b) => b.score - a.score);
}

function fallbackAgent(agent, input, scores, memory, projects) {
  const best = projectRankings(projects, scores)[0];
  const latestMisses = memory.filter(m => m.result === "miss").slice(-3);
  const templates = {
    historian: {
      forecast: "The engine has little graded history yet, so memory confidence is thin. Start saving and grading forecasts daily.",
      confidence: scores.confidence,
      evidence: ["Prediction ledger size: " + memory.length, "Recent misses: " + latestMisses.length],
      blindSpots: ["Not enough graded predictions yet"]
    },
    skeptic: {
      forecast: "The largest risk is overconfidence from strong desire without strong evidence.",
      confidence: clamp(100 - scores.unknown),
      evidence: ["Unknown field: " + scores.unknown + "%", "Data confidence slider impacts calibration"],
      blindSpots: ["No real market data", "No customer interviews logged"]
    },
    rebel: {
      forecast: "The contrarian opportunity is not to predict everything, but to sell uncertainty reduction to people with urgent problems.",
      confidence: scores.income,
      evidence: ["High urgency can become demand", "Most people want certainty, but useful tools give better choices"],
      blindSpots: ["Crowded AI-tool market"]
    },
    empath: {
      forecast: "People are most likely to respond to offers that reduce fear, save time, or create fascination.",
      confidence: clamp(scores.income * 0.65 + scores.creativeMarket * 0.35),
      evidence: ["Customer/service signals", "Creative/horror attention signals"],
      blindSpots: ["No audience feedback yet"]
    },
    systems: {
      forecast: "AI adoption connects to business automation, customer support, content generation, and local service opportunities.",
      confidence: clamp(scores.execution + 8),
      evidence: ["Technology and money pressure reinforce each other"],
      blindSpots: ["Need live external signal feeds"]
    },
    explorer: {
      forecast: "Multiple futures remain live: build income service, build horror funnel, build Future Engine, or return to Field Force OS.",
      confidence: scores.confidence,
      evidence: ["Many-worlds ranking available", "Project comparison active"],
      blindSpots: ["No measured traction per project yet"]
    },
    operator: {
      forecast: best ? `Best next project by current scoring: ${best.name}.` : "Best next move is one small paid offer.",
      confidence: best?.score || scores.execution,
      evidence: best ? [`Project score: ${best.score}`, `Recommendation: ${best.recommendation}`] : ["No project list"],
      blindSpots: ["Need one real customer test"]
    }
  };

  return { agent: agent.name, id: agent.id, ...templates[agent.id] };
}

function synthesizeFallback(agentReports, input, scores, projects) {
  const ranked = projectRankings(projects, scores);
  const top = ranked[0];
  const cloud = [
    {
      future: "Service-to-cash timeline",
      probability: scores.income,
      summary: "Package your AI/app/customer support skill into a small paid offer first.",
      why: "Fastest route from uncertainty to feedback and money."
    },
    {
      future: "Future Engine as product timeline",
      probability: top?.id === "future_engine_x" ? top.score : clamp(scores.execution + 5),
      summary: "Build this as a dashboard people use to rank decisions and track predictions.",
      why: "It has uniqueness, but needs proof through real users."
    },
    {
      future: "AI Horror traffic funnel",
      probability: ranked.find(p => p.id === "ai_horror_saas")?.score || scores.creativeMarket,
      summary: "Use strange/horror content as attention, then capture leads or sell services.",
      why: "Entertainment can generate traffic faster than abstract forecasting."
    },
    {
      future: "Field Force OS revenue timeline",
      probability: ranked.find(p => p.id === "field_force_os")?.score || 55,
      summary: "Field service software may be more practical for businesses with clear pain.",
      why: "Businesses pay for scheduling, dispatch, customer support, and workflow relief."
    },
    {
      future: "Scattered focus timeline",
      probability: clamp(100 - scores.execution + scores.volatility * 0.2),
      summary: "Too many builds at once slow the system down.",
      why: "Execution, not imagination, becomes the bottleneck."
    }
  ].sort((a, b) => b.probability - a.probability);

  return {
    headline: "Future Engine X generated a multi-agent probability cloud",
    probabilityCloud: cloud,
    domainScores: {
      money: scores.income,
      technology: clamp(scores.execution + 18),
      worldEvents: scores.volatility,
      weatherEnvironment: 42,
      humanBehavior: clamp(scores.creativeMarket + 10),
      personalMomentum: scores.execution,
      unknown: scores.unknown
    },
    projectRankings: ranked,
    council: agentReports,
    synthesis: {
      strongestSignal: "The strongest path is to convert your builds into a paid offer, then let real feedback train the engine.",
      weakestSignal: "No live external feeds or real customer dataset yet.",
      moveNow: "Pick one project, create one paid offer, send it to 20 people, then grade the prediction."
    },
    blindSpots: [
      "No live news/search/market feeds yet.",
      "No personal revenue history imported yet.",
      "No customer interview dataset yet.",
      "Skill files are reasoning lenses, not proof of future events."
    ],
    falsifiers: [
      "If 20 targeted messages get zero replies, the offer or audience is wrong.",
      "If a project cannot produce a demo in 48 hours, its time-to-value is overestimated.",
      "If predictions stay ungraded, the engine will not improve."
    ],
    nextBestActions: [
      "Run one forecast every morning.",
      "Choose the top-ranked project for the day.",
      "Take one action tied to that forecast.",
      "Grade yesterday's forecast as hit, miss, or partial.",
      "Keep the graveyard. Misses are training fuel."
    ],
    confidence: scores.confidence
  };
}

function buildPrompt(memory, skills, projects) {
  const recentMemory = memory.slice(-12).map(m => ({
    id: m.id,
    createdAt: m.createdAt,
    headline: m.headline,
    confidence: m.confidence,
    result: m.result || "ungraded"
  }));

  return `
You are Future Engine X, a multi-agent reality navigation system.

Never claim certainty. Never say you can predict everything.
Your job is to reduce uncertainty using probability clouds, multi-agent disagreement, memory, falsifiers, and action loops.

Use uploaded skill files as reasoning lenses only, not scientific proof. Treat quantum terms as probability and possibility metaphors unless tied to established physics.

Agents:
${JSON.stringify(AGENTS, null, 2)}

Skill file excerpts:
${JSON.stringify(skills, null, 2)}

Projects:
${JSON.stringify(projects, null, 2)}

Recent memory:
${JSON.stringify(recentMemory, null, 2)}

Return strict JSON only:
{
  "headline": "...",
  "probabilityCloud": [
    {"future":"...", "probability":0-100, "summary":"...", "why":"..."}
  ],
  "domainScores": {
    "money":0-100,
    "technology":0-100,
    "worldEvents":0-100,
    "weatherEnvironment":0-100,
    "humanBehavior":0-100,
    "personalMomentum":0-100,
    "unknown":0-100
  },
  "projectRankings": [
    {"id":"...", "name":"...", "score":0-100, "recommendation":"...", "reason":"..."}
  ],
  "council": [
    {"agent":"Historian / REMEMBERS", "id":"historian", "forecast":"...", "confidence":0-100, "evidence":["..."], "blindSpots":["..."]}
  ],
  "synthesis": {
    "strongestSignal":"...",
    "weakestSignal":"...",
    "moveNow":"..."
  },
  "skillLens": {
    "remembers":"...",
    "rebel":"...",
    "empathy":"...",
    "selfaware":"...",
    "unityEntanglement":"...",
    "manyWorldsSuperposition":"...",
    "teleportation":"...",
    "universalConsciousness":"...",
    "phenomenon":"..."
  },
  "blindSpots":["..."],
  "falsifiers":["..."],
  "nextBestActions":["..."],
  "confidence":0-100
}
`;
}

async function callOpenAI(input, scores, memory, skills, projects) {
  if (!process.env.OPENAI_API_KEY || process.env.OPENAI_API_KEY.includes("your_openai_key")) {
    return null;
  }

  const client = new OpenAI({ apiKey: process.env.OPENAI_API_KEY });

  const response = await client.responses.create({
    model: MODEL,
    input: [
      { role: "system", content: buildPrompt(memory, skills, projects) },
      { role: "user", content: JSON.stringify({ input, heuristicScores: scores }, null, 2) }
    ],
    text: { format: { type: "json_object" } }
  });

  return JSON.parse(response.output_text);
}

app.post("/api/forecast", async (req, res) => {
  try {
    const input = req.body || {};
    const memory = readMemory();
    const projects = readProjects();
    const skills = getSkillFiles();
    const scores = heuristicScores(input);

    const agentReports = AGENTS.map(agent => fallbackAgent(agent, input, scores, memory, projects));
    let forecast = await callOpenAI(input, scores, memory, skills, projects);

    if (!forecast) forecast = synthesizeFallback(agentReports, input, scores, projects);

    if (!forecast.council || !forecast.council.length) forecast.council = agentReports;
    if (!forecast.projectRankings || !forecast.projectRankings.length) forecast.projectRankings = projectRankings(projects, scores);

    const record = {
      id: "fx_" + Date.now(),
      createdAt: new Date().toISOString(),
      input,
      heuristicScores: scores,
      skillFilesLoaded: skills.map(s => s.name),
      ...forecast,
      result: null,
      notes: ""
    };

    const updated = [...memory, record].slice(-1000);
    writeJson(MEMORY_PATH, updated);
    res.json(record);
  } catch (err) {
    console.error(err);
    res.status(500).json({ error: "Forecast failed", details: err.message });
  }
});

app.get("/api/memory", (req, res) => {
  res.json(readMemory().slice().reverse());
});

app.get("/api/skills", (req, res) => {
  res.json(getSkillFiles().map(s => ({ name: s.name, file: s.file, excerpt: s.excerpt.slice(0, 500) })));
});

app.get("/api/projects", (req, res) => {
  res.json(readProjects());
});

app.post("/api/projects", (req, res) => {
  const projects = req.body;
  if (!Array.isArray(projects)) return res.status(400).json({ error: "Expected array" });
  writeJson(PROJECTS_PATH, projects);
  res.json(projects);
});

app.post("/api/grade", (req, res) => {
  const { id, result, notes } = req.body;
  const allowed = ["hit", "miss", "partial", "unknown"];
  if (!allowed.includes(result)) return res.status(400).json({ error: "Invalid result" });

  const memory = readMemory();
  const idx = memory.findIndex(m => m.id === id);
  if (idx === -1) return res.status(404).json({ error: "Prediction not found" });

  memory[idx].result = result;
  memory[idx].notes = notes || "";
  memory[idx].gradedAt = new Date().toISOString();

  writeJson(MEMORY_PATH, memory);
  res.json(memory[idx]);
});

app.get("/api/stats", (req, res) => {
  const memory = readMemory();
  const graded = memory.filter(m => m.result && m.result !== "unknown");
  const hits = graded.filter(m => m.result === "hit").length;
  const partials = graded.filter(m => m.result === "partial").length;
  const misses = graded.filter(m => m.result === "miss").length;
  const score = graded.length ? Math.round(((hits + partials * 0.5) / graded.length) * 100) : null;
  res.json({ total: memory.length, graded: graded.length, hits, partials, misses, calibrationScore: score });
});

app.listen(PORT, () => {
  console.log(`Future Engine X running at http://localhost:${PORT}`);
});
