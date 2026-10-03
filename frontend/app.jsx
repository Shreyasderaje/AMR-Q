/* AMR-Q platform dashboard — React (compiled in-browser by Babel standalone). */
const { useState, useEffect, useRef, useCallback } = React;

/* GitHub repository (source link in the navbar). */
const REPO_URL = "https://github.com/Shreyasderaje/AMR-Q";

const KNOWN_LABELS = {
  class_a: "Class A β-lactamase",
  class_b: "Metallo-β-lactamase",
  class_c: "Class C β-lactamase",
  class_d: "Class D β-lactamase",
  efflux: "Efflux substrate",
};

/* ------------------------------------------------------------------ */
/* Inline SVG icon set (1.5px stroke, currentColor)                     */
const Icon = ({ d, size = 15, vb = 24, children }) => (
  <svg width={size} height={size} viewBox={`0 0 ${vb} ${vb}`} fill="none"
       stroke="currentColor" strokeWidth="1.7" strokeLinecap="round"
       strokeLinejoin="round" aria-hidden="true">
    {d ? <path d={d} /> : children}
  </svg>
);
const Mark = ({ size = 26 }) => (
  <svg width={size} height={size} viewBox="0 0 32 32" aria-hidden="true">
    <rect width="32" height="32" rx="7" fill="none" stroke="rgba(255,255,255,0.14)" />
    <circle cx="16" cy="16" r="3" fill="#32D9BE" />
    <ellipse cx="16" cy="16" rx="11" ry="4.4" fill="none" stroke="#32D9BE" strokeWidth="1.3" transform="rotate(-24 16 16)" opacity="0.9" />
    <ellipse cx="16" cy="16" rx="11" ry="4.4" fill="none" stroke="#67E8F9" strokeWidth="1.3" transform="rotate(52 16 16)" opacity="0.5" />
  </svg>
);
const IPlay = () => <Icon d="M7 5.5v13l11-6.5z" />;
const ICsv = () => <Icon d="M12 3v12m0 0l-4.5-4.5M12 15l4.5-4.5M4 19h16" />;
const IX = () => <Icon d="M6 6l12 12M18 6L6 18" />;
const IExt = () => <Icon d="M14 5h5v5M19 5l-8 8M9 5H6a1 1 0 0 0-1 1v12a1 1 0 0 0 1 1h12a1 1 0 0 0 1-1v-3" />;
const ITarget = () => <Icon><circle cx="12" cy="12" r="8.5" /><circle cx="12" cy="12" r="4" /><circle cx="12" cy="12" r="0.5" fill="currentColor" /></Icon>;
const IGitHub = () => (
  <svg width="15" height="15" viewBox="0 0 16 16" fill="currentColor" aria-hidden="true">
    <path d="M8 0C3.58 0 0 3.58 0 8c0 3.54 2.29 6.53 5.47 7.59.4.07.55-.17.55-.38 0-.19-.01-.82-.01-1.49-2.01.37-2.53-.49-2.69-.94-.09-.23-.48-.94-.82-1.13-.28-.15-.68-.52-.01-.53.63-.01 1.08.58 1.23.82.72 1.21 1.87.87 2.33.66.07-.52.28-.87.51-1.07-1.78-.2-3.64-.89-3.64-3.95 0-.87.31-1.59.82-2.15-.08-.2-.36-1.02.08-2.12 0 0 .67-.21 2.2.82.64-.18 1.32-.27 2-.27s1.36.09 2 .27c1.53-1.04 2.2-.82 2.2-.82.44 1.1.16 1.92.08 2.12.51.56.82 1.27.82 2.15 0 3.07-1.87 3.75-3.65 3.95.29.25.54.73.54 1.48 0 1.07-.01 1.93-.01 2.2 0 .21.15.46.55.38A8.01 8.01 0 0 0 16 8c0-4.42-3.58-8-8-8z" />
  </svg>
);

/* ------------------------------------------------------------------ */
const svgUri = (svg) => "data:image/svg+xml;utf8," + encodeURIComponent(svg);

async function api(path, options) {
  const res = await fetch(path, options);
  if (!res.ok) {
    let msg = res.statusText;
    try { msg = (await res.json()).detail || msg; } catch (e) { /* noop */ }
    throw new Error(msg);
  }
  return res.json();
}

const fmt = (v, dp = 2) => (v === null || v === undefined || Number.isNaN(v)) ? "—" : (+v).toFixed(dp);

/* ------------------------------------------------------------------ */
function ConvergenceChart({ trace, exact, height = 148 }) {
  if (!trace || trace.length < 2) {
    return <div className="tiny" style={{ color: "var(--text-3)" }}>No convergence trace recorded.</div>;
  }
  const w = 620, h = height, pad = 12;
  const lo = Math.min(...trace, exact), hi = Math.max(...trace, exact);
  const span = hi - lo || 1;
  const x = (i) => pad + (w - 2 * pad) * (i / (trace.length - 1));
  const y = (v) => h - pad - (h - 2 * pad) * ((v - lo) / span);
  const pts = trace.map((v, i) => `${x(i).toFixed(1)},${y(v).toFixed(1)}`).join(" ");
  return (
    <div className="chart-box">
      <svg viewBox={`0 0 ${w} ${h}`} width="100%" role="img"
           aria-label="VQE energy per iteration versus exact diagonalization">
        <line x1={pad} x2={w - pad} y1={y(exact)} y2={y(exact)}
              stroke="#67E8F9" strokeDasharray="2 4" strokeWidth="1" opacity="0.7" />
        <text x={w - pad} y={y(exact) - 6} textAnchor="end" fontSize="10.5" fill="#67E8F9"
              fontFamily="JetBrains Mono, monospace">
          exact E&#8320; {exact.toFixed(4)} eV
        </text>
        <polyline points={pts} fill="none" stroke="#32D9BE" strokeWidth="1.5" />
        <text x={pad} y={h - 2} fontSize="9.5" fill="#626a72" fontFamily="JetBrains Mono, monospace">
          {trace.length} energy evaluations
        </text>
      </svg>
    </div>
  );
}

function FeatureBars({ features }) {
  const defs = [
    ["contact_frac", "Pocket contact fraction", (v) => (v * 100).toFixed(0) + "%", 1],
    ["hbond_contacts", "Hydrogen-bond contacts", (v) => v.toFixed(0), 8],
    ["hydrophobic_contacts", "Hydrophobic contacts", (v) => v.toFixed(0), 30],
    ["electrostatic_complementarity", "Charge complementarity", (v) => v.toFixed(2), 1],
    ["metal_proximity_score", "Metal proximity", (v) => v.toFixed(2), 1],
    ["clash", "Steric clashes", (v) => v.toFixed(0), 5],
  ];
  return (
    <div className="feat-bars">
      {defs.map(([key, label, f, max]) => {
        const v = Math.max(0, features[key] || 0);
        return (
          <div className="feat-row" key={key}>
            <span className="fl">{label}</span>
            <div className="ft"><div className="ff" style={{ width: Math.min(100, (v / max) * 100) + "%" }} /></div>
            <span className="fv">{f(v)}</span>
          </div>
        );
      })}
    </div>
  );
}

/* ------------------------------------------------------------------ */
function DetailSheet({ compound, onClose }) {
  const q = compound.quantum || {};
  const mp = q.model_params || {};
  useEffect(() => {
    const esc = (e) => e.key === "Escape" && onClose();
    window.addEventListener("keydown", esc);
    document.body.style.overflow = "hidden";
    return () => { window.removeEventListener("keydown", esc); document.body.style.overflow = ""; };
  }, [onClose]);
  return (
    <div className="backdrop" onClick={onClose}>
      <aside className="sheet" onClick={(e) => e.stopPropagation()}>
        <button className="sheet-close" onClick={onClose} aria-label="Close detail view"><IX /></button>
        <div className="eyebrow">Candidate {String(compound.rank).padStart(2, "0")}</div>
        <div className="sheet-head">
          <img src={svgUri(compound.svg)} alt={"Two-dimensional structure of " + compound.name} />
          <div>
            <h3>{compound.name}</h3>
            <div className="cls">{compound.drug_class}</div>
            <div className="score-display">
              <div className="big num">{(compound.ml_score * 100).toFixed(1)}<small>/ 100</small></div>
              <div className="tiny" style={{ color: "var(--text-3)" }}>resistance-breaking score</div>
            </div>
            <div style={{ display: "flex", gap: 5, flexWrap: "wrap", marginTop: 10 }}>
              {(compound.known_targets || []).map((k) => (
                <span className="tag known" key={k}>{KNOWN_LABELS[k] || k}</span>))}
              {compound.ligand.chelating_atoms > 0 &&
                <span className="tag metal">{compound.ligand.chelating_atoms} chelating atoms</span>}
            </div>
          </div>
        </div>
        <p className="mech">{compound.mechanism}</p>

        <h4>Quantum simulation — VQE convergence</h4>
        <div className="kv-grid">
          <div><div className="k">Optimizer</div><div className="v">{q.optimizer} · {q.iterations} evals</div></div>
          <div><div className="k">Converged</div><div className="v">{q.converged ? "yes" : "boundary"}</div></div>
          <div><div className="k">VQE energy</div><div className="v">{fmt(q.energy, 4)} eV</div></div>
          <div><div className="k">Exact E&#8320; (diag.)</div><div className="v">{fmt(q.exact_energy, 4)} eV</div></div>
          <div><div className="k">Error vs exact</div><div className="v">{q.error?.toExponential(1)}</div></div>
          <div><div className="k">Stabilization &#916;E</div><div className="v">{fmt(q.delta_e_ev, 3)} eV</div></div>
          <div><div className="k">Charge-transfer weight</div><div className="v">{(q.ct_weight * 100).toFixed(1)}%</div></div>
          <div><div className="k">Hamiltonian</div><div className="v">{q.n_pauli_terms} Pauli terms · 4q</div></div>
        </div>
        <div style={{ marginTop: 10 }}>
          <ConvergenceChart trace={q.trace} exact={q.exact_energy} />
        </div>

        <h4>Binding-model parameters</h4>
        <div className="kv-grid">
          <div><div className="k">&#949; ligand HOMO</div><div className="v">{mp.eps_LH} eV</div></div>
          <div><div className="k">&#949; ligand LUMO</div><div className="v">{mp.eps_LL} eV</div></div>
          <div><div className="k">&#949; pocket HOMO</div><div className="v">{mp.eps_PH} eV</div></div>
          <div><div className="k">&#949; pocket LUMO</div><div className="v">{mp.eps_PL} eV</div></div>
          <div><div className="k">Coupling t&#8321;</div><div className="v">{mp.t_ct} eV</div></div>
          <div><div className="k">Coupling t&#8322;</div><div className="v">{mp.t_ct2} eV</div></div>
          <div><div className="k">Contact V</div><div className="v">{mp.V_total} eV</div></div>
          <div><div className="k">Decoupled E</div><div className="v">{fmt(q.decoupled_energy, 3)} eV</div></div>
        </div>

        <h4>Ligand–pocket interaction</h4>
        <FeatureBars features={compound.feature_dict} />

        <h4>Score drivers — gradient attribution</h4>
        <div className="attr-list">
          {(compound.attribution || []).map((a, i) => (
            <div className="attr-row" key={i}>
              <span className="fn">{a.feature.replace(/_/g, " ")}</span>
              <span className={"fc " + (a.contribution >= 0 ? "pos" : "neg")}>
                {a.contribution >= 0 ? "+" : ""}{a.contribution.toFixed(3)}
              </span>
            </div>
          ))}
        </div>

        <h4>Molecule profile</h4>
        <div className="kv-grid">
          <div><div className="k">Molecular weight</div><div className="v">{fmt(compound.ligand.mw, 1)} Da</div></div>
          <div><div className="k">cLogP</div><div className="v">{fmt(compound.ligand.crippen_logp, 2)}</div></div>
          <div><div className="k">TPSA</div><div className="v">{fmt(compound.ligand.tpsa, 0)} &#197;&#178;</div></div>
          <div><div className="k">HBD / HBA</div><div className="v">{compound.ligand.hbd} / {compound.ligand.hba}</div></div>
          <div><div className="k">Rotatable bonds</div><div className="v">{compound.ligand.rotatable_bonds}</div></div>
          <div><div className="k">Formal charge</div><div className="v">{compound.ligand.formal_charge}</div></div>
        </div>
        <div style={{ marginTop: 12 }}>
          <div className="smiles-box">{compound.smiles}</div>
        </div>
      </aside>
    </div>
  );
}

/* ------------------------------------------------------------------ */
function App() {
  const [targets, setTargets] = useState([]);
  const [backends, setBackends] = useState([]);
  const [apiOnline, setApiOnline] = useState(true);
  const [selected, setSelected] = useState(null);
  const [mode, setMode] = useState("known");
  const [customPdbId, setCustomPdbId] = useState("");
  const [limit, setLimit] = useState(40);
  const [backend, setBackend] = useState("statevector");
  const [optimizer, setOptimizer] = useState("cobyla");
  const [decoys, setDecoys] = useState(true);
  const [job, setJob] = useState(null);
  const [jobId, setJobId] = useState(null);
  const [results, setResults] = useState(null);
  const [detail, setDetail] = useState(null);
  const [error, setError] = useState(null);
  const pollRef = useRef(null);

  useEffect(() => {
    api("/api/health").then(() => setApiOnline(true)).catch(() => setApiOnline(false));
    api("/api/targets").then((d) => {
      setTargets(d.targets);
      setSelected((s) => s || d.targets[0]?.key || null);
    }).catch((e) => setError("Failed to load targets: " + e.message));
    api("/api/backends").then((d) => setBackends(d.backends)).catch(() => {});
  }, []);

  const stopPoll = useCallback(() => {
    if (pollRef.current) { clearInterval(pollRef.current); pollRef.current = null; }
  }, []);
  useEffect(() => stopPoll, [stopPoll]);

  const startRun = useCallback(async () => {
    setError(null); setResults(null); setDetail(null); setJobId(null);
    setJob({ progress: 0, message: "Initializing screen" });
    try {
      const body = mode === "known"
        ? { target_key: selected, limit, backend, optimizer, include_decoys: decoys }
        : { pdb_id: customPdbId.trim().toUpperCase(), limit, backend, optimizer, include_decoys: decoys };
      const { job_id } = await api("/api/analyze", {
        method: "POST", headers: { "Content-Type": "application/json" },
        body: JSON.stringify(body),
      });
      setJobId(job_id);
      setJob({ progress: 0, message: "Queued" });
      pollRef.current = setInterval(async () => {
        try {
          const s = await api(`/api/jobs/${job_id}`);
          setJob({ progress: s.progress, message: s.message, stage: s.stage });
          if (s.status === "done") {
            stopPoll();
            const r = await api(`/api/jobs/${job_id}/results`);
            setResults(r); setJob(null);
          } else if (s.status === "error") {
            stopPoll(); setError(s.error || "The screening job failed."); setJob(null);
          }
        } catch (e) { stopPoll(); setError(e.message); setJob(null); }
      }, 1200);
    } catch (e) { setError(e.message); setJob(null); }
  }, [mode, selected, customPdbId, limit, backend, optimizer, decoys, stopPoll]);

  const running = !!job;
  const selectedTarget = targets.find((t) => t.key === selected);

  return (
    <div>
      {/* ---------------- nav ---------------- */}
      <nav className="nav">
        <div className="nav-inner">
          <a className="wordmark" href="#top">
            <Mark />
            <b>AMR-Q</b>
            <span>Quantum drug-discovery platform</span>
          </a>
          <div className="nav-links">
            <a href="#console">Console</a>
            <a href="#pipeline">Pipeline</a>
            <a href="#science">Science</a>
          </div>
          <div className="nav-right">
            <span className={"api-status" + (apiOnline ? "" : " down")}>
              <span className="dot" />API {apiOnline ? "online" : "offline"}
            </span>
            {REPO_URL &&
              <a className="gh-link" href={REPO_URL} target="_blank" rel="noreferrer">
                <IGitHub />Source
              </a>}
          </div>
        </div>
      </nav>

      <div className="app" id="top">
        {/* ---------------- hero ---------------- */}
        <header className="hero">
          <div className="eyebrow">Quantum-classical drug discovery · open source</div>
          <h1>Prioritize resistance-breaking antibiotics<br />
            with <em>quantum-computed</em> binding physics.</h1>
          <p className="lede">
            AMR-Q screens compound libraries against the proteins that make bacteria
            resistant — beta-lactamases and multidrug efflux pumps — by running a
            variational quantum eigensolver on a reduced binding model for every
            candidate, then fusing the result with classical interaction features
            in a machine-learning ranker.
          </p>
          <div className="stat-row">
            <div className="stat">
              <div className="value num">1.27M<small>deaths / yr</small></div>
              <div className="label">attributable to bacterial AMR in 2019 (GRAM, Lancet 2022)</div>
            </div>
            <div className="stat">
              <div className="value num">1.91M<small>by 2050</small></div>
              <div className="label">projected annual attributable deaths (GBD 2021 forecast)</div>
            </div>
            <div className="stat">
              <div className="value num">$2.6B<small>per drug</small></div>
              <div className="label">capitalized cost of bringing one new drug to market</div>
            </div>
            <div className="stat">
              <div className="value num">4<small>qubits</small></div>
              <div className="label">per VQE binding simulation — the same scale as published protein–ligand VQE work</div>
            </div>
          </div>
        </header>

        {/* ---------------- pipeline strip ---------------- */}
        <section className="pipeline-strip" id="pipeline">
          {[
            ["01", "Structure", "RCSB PDB fetch &amp; parse"],
            ["02", "Pocket", "catalytic / metal / co-crystal detection"],
            ["03", "Interaction", "pose features · RDKit + KD-tree contacts"],
            ["04", "VQE", "4-qubit two-site binding Hamiltonian"],
            ["05", "Ranking", "PyTorch scorer over 23 features"],
            ["06", "Candidates", "ranked, explainable, exportable"],
          ].map(([n, t, d]) => (
            <div className="pl-step" key={n}>
              <div className="n">{n}</div>
              <div className="t">{t}</div>
              <div className="d" dangerouslySetInnerHTML={{ __html: d }} />
            </div>
          ))}
        </section>

        {error && <div className="error-bar"><ITarget /> <span>{error}</span></div>}

        {/* ---------------- console ---------------- */}
        <main className="console" id="console">
          <section className="card card-pad">
            <h2>Screening console</h2>
            <p className="sub">Configure a target and launch the hybrid quantum–classical pipeline.</p>

            <div className="seg" role="tablist">
              <button className={mode === "known" ? "on" : ""} onClick={() => setMode("known")}>Curated resistance panel</button>
              <button className={mode === "custom" ? "on" : ""} onClick={() => setMode("custom")}>Custom PDB entry</button>
            </div>

            {mode === "known" ? (
              <div className="target-list">
                {targets.map((t) => (
                  <button key={t.key} className={"target-row" + (selected === t.key ? " sel" : "")}
                          onClick={() => setSelected(t.key)}>
                    <span className="radio" />
                    <span className="tt">
                      <b>{t.short}</b>
                      <span>{t.enzyme_class}</span>
                    </span>
                    {t.chelation_relevant && <span className="zn">Zn²⁺</span>}
                    <span className="pdb">{t.pdb_id}</span>
                  </button>
                ))}
              </div>
            ) : (
              <div className="field" style={{ marginTop: 4 }}>
                <label>PDB entry ID</label>
                <input type="text" value={customPdbId} maxLength={4} placeholder="e.g. 4HL2"
                       onChange={(e) => setCustomPdbId(e.target.value)} />
                <span className="hint">
                  The binding pocket is detected from the co-crystallized ligand.
                  Proteins without a ligand accept manual pocket residues through the REST API
                  (<span className="mono">pocket_residues</span>).
                </span>
              </div>
            )}

            {mode === "known" && selectedTarget && (
              <div className="target-detail">
                <b style={{ color: "var(--text)" }}>{selectedTarget.name}</b> — {selectedTarget.resistance_mechanism}
                <div style={{ marginTop: 6, color: "var(--text-3)" }}>{selectedTarget.clinical_note}</div>
              </div>
            )}

            <div style={{ height: 22 }} />
            <div className="field">
              <label>Library size <b>{limit} compounds</b></label>
              <input type="range" min="10" max="120" step="5" value={limit}
                     onChange={(e) => setLimit(+e.target.value)} />
            </div>
            <div className="field">
              <label>Quantum backend</label>
              <select value={backend} onChange={(e) => setBackend(e.target.value)}>
                {backends.map((b) => (
                  <option key={b.key} value={b.key} disabled={!b.available}>
                    {b.label}{b.available ? "" : " — unavailable"}
                  </option>
                ))}
              </select>
            </div>
            <div className="field">
              <label>VQE optimizer</label>
              <select value={optimizer} onChange={(e) => setOptimizer(e.target.value)}>
                <option value="cobyla">COBYLA — gradient-free, deterministic trace</option>
                <option value="spsa">SPSA — simultaneous perturbation, hardware-native</option>
              </select>
            </div>
            <label className="check-row">
              <input type="checkbox" checked={decoys} onChange={(e) => setDecoys(e.target.checked)} />
              Include pharmacologically unrelated decoy controls
            </label>

            <button className="primary-btn" disabled={running || (mode === "known" && !selected)}
                    onClick={startRun}>
              {running ? <><span className="running-dot" />Screening in progress</> : <><IPlay />Run quantum screen</>}
            </button>
            {backends.some((b) => b.key === "ibm" && !b.available) && (
              <p className="ibm-hint">
                To execute on IBM Quantum hardware, install <code>qiskit-ibm-runtime</code> and set the
                <code>AMRQ_IBM_TOKEN</code> environment variable with a free account token.
              </p>
            )}
          </section>

          <section className="card card-pad">
            <h2>Ranked candidates</h2>
            <p className="sub">Sorted by the fused quantum–classical resistance-breaking score.</p>

            {running && (
              <div className="progress-block">
                <div className="progress-head">
                  <span className="stage">{job.stage === "quantum" ? "Quantum screening" : job.stage === "ranking" ? "ML ranking" : job.stage === "pocket" ? "Pocket detection" : "Preparing"}</span>
                  <span className="pct num">{Math.round(job.progress || 0)}%</span>
                </div>
                <div className="progress-track">
                  <div className="progress-fill" style={{ width: (job.progress || 0) + "%" }} />
                </div>
                <div className="progress-msg"><span className="running-dot" />{job.message}</div>
              </div>
            )}

            {!running && !results && (
              <div className="empty-state">
                <div className="glyph"><ITarget size={34} /></div>
                <h3>No screen yet</h3>
                <p>Select a resistance protein and run the screen. Each compound is placed into the
                   binding pocket, converted into a 4-qubit two-site Hamiltonian, and solved with VQE
                   before scoring.</p>
                <div className="empty-steps">
                  {[
                    ["01", "Fetch & parse", "The protein structure is retrieved from RCSB and the resistance pocket is defined."],
                    ["02", "Quantum loop", "A VQE loop optimizes the binding Hamiltonian for every compound."],
                    ["03", "Fusion scoring", "Quantum and classical features are combined by the trained ranker."],
                    ["04", "Explainable output", "Every row opens into a full quantum and interaction report."],
                  ].map(([n, t, d]) => (
                    <div className="empty-step" key={n}>
                      <div className="n">{n}</div>
                      <div className="t">{t}</div>
                      <div className="d">{d}</div>
                    </div>
                  ))}
                </div>
              </div>
            )}

            {results && !running && (
              <>
                <div className="meta-grid">
                  <div><div className="k">Structure</div><div className="v">{results.summary.structure}</div></div>
                  <div><div className="k">Pocket</div><div className="v">
                    {results.pocket.n_residues} residues <small>· {results.pocket.method}</small></div></div>
                  <div><div className="k">Screened</div><div className="v">
                    {results.summary.n_compounds_ranked} <small>in {results.summary.elapsed_sec}s</small></div></div>
                  <div><div className="k">VQE mean |&#916;| exact</div><div className="v num">
                    {results.summary.vqe.mean_error_vs_exact.toExponential(1)} <small>eV</small></div></div>
                  <div><div className="k">Backend</div><div className="v">{results.summary.backend_used}</div></div>
                  <div><div className="k">Ranker</div><div className="v">{results.summary.ranker.backend}</div></div>
                </div>
                <div className="residue-row">
                  <span className="lbl">Pocket</span>
                  {results.pocket.residues.slice(0, 14).map((r, i) => <span className="res-chip" key={i}>{r}</span>)}
                  {results.pocket.residues.length > 14 &&
                    <span className="res-chip more">+{results.pocket.residues.length - 14}</span>}
                  {results.pocket.metals?.map((m) => <span className="res-chip metal" key={m}>{m}</span>)}
                </div>
                <table className="results">
                  <thead>
                    <tr>
                      <th style={{ width: 34 }}>#</th>
                      <th style={{ width: 54 }}>Structure</th>
                      <th>Compound</th>
                      <th className="r">Score</th>
                      <th className="r">&#916;E quantum</th>
                      <th className="r">Contacts</th>
                      <th style={{ width: 90 }}></th>
                    </tr>
                  </thead>
                  <tbody>
                    {results.compounds.map((c) => {
                      const isDecoy = c.drug_class === "decoy";
                      const known = c.known_targets.includes(results.target.resistance_class)
                                    && results.target.resistance_class !== "unknown";
                      return (
                        <tr key={c.name} className={isDecoy ? "decoy-row" : ""} onClick={() => setDetail(c)}>
                          <td className="num muted tiny">{String(c.rank).padStart(2, "0")}</td>
                          <td><img className="c-thumb" src={svgUri(c.svg)} alt={c.name} loading="lazy" /></td>
                          <td className="c-name">
                            <b>{c.name}</b>
                            <span>{c.drug_class}</span>
                          </td>
                          <td className="r">
                            <div className="score-cell">
                              <div className="bar"><div style={{ width: (c.ml_score * 100) + "%" }} /></div>
                              <span className="val">{(c.ml_score * 100).toFixed(0)}</span>
                            </div>
                          </td>
                          <td className="r num">{fmt(c.quantum.delta_e_ev, 2)}<span style={{ color: "var(--text-3)" }}> eV</span></td>
                          <td className="r num">{fmt(c.feature_dict.contact_frac, 2)}</td>
                          <td className="r">
                            {known && <span className="tag known">Known</span>}
                            {isDecoy && <span className="tag decoy">Decoy</span>}
                            {c.ligand.chelating_atoms > 0 && results.target.chelation_relevant &&
                              <span className="tag metal">Chel</span>}
                          </td>
                        </tr>
                      );
                    })}
                  </tbody>
                </table>
                <div className="table-foot">
                  {jobId
                    ? <a className="ghost-btn" href={`/api/jobs/${jobId}/results.csv`}><ICsv />Export CSV</a>
                    : <span />}
                  <span className="tiny" style={{ color: "var(--text-3)" }}>
                    Click any row for the quantum report · scores fuse 23 classical and quantum features
                  </span>
                </div>
              </>
            )}
          </section>
        </main>

        {/* ---------------- science note ---------------- */}
        <section className="card card-pad" id="science" style={{ marginTop: 20 }}>
          <h2>How the quantum layer works</h2>
          <p className="sub" style={{ maxWidth: 760 }}>
            Full drug–protein electronic structure is far beyond any simulator or quantum device
            (one qubit per spin-orbital implies thousands of qubits). AMR-Q follows the field's
            standard reduced-model approach: the binding interface is condensed into a two-site,
            4-orbital Hamiltonian — ligand donor/acceptor coupled to pocket donor/acceptor — and VQE
            computes its ground state on the qiskit statevector simulator with a particle-conserving
            excitation ansatz. The stabilization energy relative to the decoupled reference, plus the
            ground state's charge-transfer character, feed the ranking model alongside classical features.
          </p>
          <p className="sub" style={{ maxWidth: 760 }}>
            This is an honest research prototype: the quantum term is a relative scoring signal, not a
            validated free-energy predictor. The methodology, parameter mapping, and limitations are
            documented in <span className="mono">docs/SCIENCE.md</span> and <span className="mono">docs/RESEARCH.md</span>.
          </p>
        </section>

        <DetailSheetMount compound={detail} onClose={() => setDetail(null)} />

        {/* ---------------- footer ---------------- */}
        <footer className="footer">
          <div className="footer-grid">
            <div>
              <h5>AMR-Q</h5>
              <p>
                An open-source research prototype for quantum-assisted prioritization of
                resistance-breaking antibiotic candidates. Built for education and early-stage
                research — not a validated drug-discovery engine, and not medical advice.
              </p>
            </div>
            <div>
              <h5>Data sources</h5>
              <ul>
                <li>RCSB Protein Data Bank</li>
                <li>PubChem · ChEMBL</li>
                <li>GRAM project burden estimates</li>
              </ul>
            </div>
            <div>
              <h5>Stack</h5>
              <ul>
                <li>Qiskit — VQE &amp; statevector simulation</li>
                <li>RDKit · BioPython — chemistry &amp; structures</li>
                <li>PyTorch — ranking model</li>
                <li>FastAPI · React — platform</li>
              </ul>
            </div>
          </div>
          <div className="fine">
            <span>AMR-Q · MIT License · quantum simulator: {apiOnline ? "connected" : "offline"}</span>
            <span>1.27M attributable deaths/yr (Lancet 2022) · 1.91M projected by 2050 (GBD 2021)</span>
          </div>
        </footer>
      </div>
    </div>
  );
}

function DetailSheetMount(props) {
  return props.compound ? <DetailSheet {...props} /> : null;
}

ReactDOM.createRoot(document.getElementById("root")).render(<App />);
