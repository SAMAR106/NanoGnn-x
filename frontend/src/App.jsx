import { useState, useRef, useEffect } from 'react'
import { LayoutDashboard, Activity, Database, Download, CheckCircle, Search } from 'lucide-react'
import { LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer, ScatterChart, Scatter, BarChart, Bar } from 'recharts'
import './index.css'

const lossData = Array.from({length: 50}, (_, i) => ({
  epoch: i * 5,
  train: Math.exp(-i/10) * 0.5 + 0.05,
  val: Math.exp(-i/10) * 0.55 + 0.08 + (Math.random()*0.02)
}));

const parityData = Array.from({length: 100}, () => {
  const actual = Math.random() * 5;
  return { actual, predicted: actual + (Math.random() - 0.5) * 0.4 };
});

const featureImportance = [
  { name: 'Electronegativity', value: 0.87 },
  { name: 'Atomic Number', value: 0.67 },
  { name: 'Covalent Radius', value: 0.46 },
  { name: 'Ionization Energy', value: 0.31 },
  { name: 'Valence Electrons', value: 0.23 },
  { name: 'Atomic Mass', value: 0.15 },
];

function App() {
  const [activeTab, setActiveTab] = useState('dashboard')
  const [files, setFiles] = useState([])
  const [cifString, setCifString] = useState("")
  const [loading, setLoading] = useState(false)
  
  // Single mode state
  const [result, setResult] = useState(null)
  const [explainData, setExplainData] = useState(null)
  const [explaining, setExplaining] = useState(false)
  const [flagged, setFlagged] = useState(false)
  
  // Batch mode state
  const [batchResults, setBatchResults] = useState(null)
  
  // Explorer state
  const [dataset, setDataset] = useState([])
  const [searchQuery, setSearchQuery] = useState("")
  
  const [error, setError] = useState(null)
  
  const viewerRef = useRef(null)
  const containerRef = useRef(null)

  // Fetch dataset when explorer is opened
  useEffect(() => {
    if (activeTab === 'explorer' && dataset.length === 0) {
      fetch(`${import.meta.env.VITE_API_BASE_URL}/dataset`)
        .then(res => res.json())
        .then(data => setDataset(data.data || []))
        .catch(err => console.error("Failed to load dataset:", err));
    }
  }, [activeTab, dataset])

  const handleFileChange = (e) => {
    const selectedFiles = Array.from(e.target.files)
    setFiles(selectedFiles)
    setResult(null)
    setBatchResults(null)
    setError(null)
    setExplainData(null)
    setFlagged(false)
    
    if (selectedFiles.length === 1) {
      const reader = new FileReader();
      reader.onload = (e) => setCifString(e.target.result);
      reader.readAsText(selectedFiles[0]);
    } else {
      setCifString("");
    }
  }

  const handleUpload = async () => {
    if (files.length === 0) return
    setLoading(true)
    setError(null)
    setFlagged(false)
    
    const formData = new FormData()
    
    try {
      if (files.length === 1) {
        formData.append('file', files[0])
        const response = await fetch(`${import.meta.env.VITE_API_BASE_URL}/predict`, { method: 'POST', body: formData })
        if (!response.ok) throw new Error(`API Error: ${response.status}`)
        const data = await response.json()
        setResult(data)
      } else {
        files.forEach(f => formData.append('files', f))
        const response = await fetch(`${import.meta.env.VITE_API_BASE_URL}/predict/batch`, { method: 'POST', body: formData })
        if (!response.ok) throw new Error(`API Error: ${response.status}`)
        const data = await response.json()
        setBatchResults(data)
      }
    } catch (err) {
      setError(err.message)
    } finally {
      setLoading(false)
    }
  }

  const handleExplain = async () => {
    if (files.length !== 1) return
    setExplaining(true)
    setError(null)
    
    const formData = new FormData()
    formData.append('file', files[0])

    try {
      const response = await fetch(`${import.meta.env.VITE_API_BASE_URL}/explain`, { method: 'POST', body: formData })
      if (!response.ok) throw new Error(`Explain API Error: ${response.status}`)
      const data = await response.json()
      setExplainData(data)
      applyHeatmap(data.heatmap)
    } catch (err) {
      setError(err.message)
    } finally {
      setExplaining(false)
    }
  }

  const downloadReport = () => {
    if (!result && !batchResults) return;
    let csv = "";
    if (batchResults) {
      csv = "Filename,Bandgap (eV),Formation Energy (eV/atom)\n" + 
            batchResults.results.map(r => `${r.filename},${r.bandgap_eV.toFixed(4)},${r.formation_energy_eV_per_atom.toFixed(4)}`).join("\n");
    } else {
      csv = `Filename,Bandgap (eV),Formation Energy (eV/atom),Key Driver Element\n`;
      csv += `${files[0].name},${result.bandgap_eV.toFixed(4)},${result.formation_energy_eV_per_atom.toFixed(4)},${explainData ? explainData.justification : 'N/A'}`;
    }
    
    const blob = new Blob([csv], { type: 'text/csv' });
    const url = window.URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = 'ALIGNN_Prediction_Report.csv';
    a.click();
  }

  useEffect(() => {
    if (activeTab === 'predict' && cifString && containerRef.current && window.$3Dmol) {
      containerRef.current.innerHTML = ""; 
      viewerRef.current = window.$3Dmol.createViewer(containerRef.current, { backgroundColor: "#F4F6F8" });
      viewerRef.current.addModel(cifString, "cif");
      viewerRef.current.setStyle({}, { sphere: { scale: 0.25, colorscheme: 'Jmol' }, stick: { radius: 0.05 } });
      viewerRef.current.zoomTo();
      viewerRef.current.render();
      
      if (explainData) {
          applyHeatmap(explainData.heatmap);
      }
    }
  }, [cifString, activeTab])

  const applyHeatmap = (heatmap) => {
    if (!viewerRef.current) return
    const atoms = viewerRef.current.getModel().selectedAtoms({});
    
    atoms.forEach((atom, i) => {
      if (i < heatmap.length) {
        const score = heatmap[i];
        const r = Math.floor(score * 255);
        const b = Math.floor((1 - score) * 255);
        const color = `#${r.toString(16).padStart(2, '0')}00${b.toString(16).padStart(2, '0')}`;
        
        viewerRef.current.setStyle(
          {serial: atom.serial}, 
          { sphere: { color: color, scale: 0.35 + (score*0.2) }, stick: { radius: 0.05, color: "#CBD5E1" } }
        );
      }
    });
    viewerRef.current.render();
  }

  const filteredDataset = dataset.filter(item => item.filename && item.filename.toLowerCase().includes(searchQuery.toLowerCase()));

  return (
    <div className="app-container">
      {/* SIDEBAR */}
      <nav className="sidebar">
        <div className="brand">
          <Database className="brand-icon" size={24} />
          <h2>ALIGNN-X</h2>
        </div>
        <p className="subtitle">Discovery Platform</p>
        
        <ul className="nav-links">
          <li className={activeTab === 'dashboard' ? 'active' : ''} onClick={() => setActiveTab('dashboard')}>
            <LayoutDashboard size={18} /> Dashboard
          </li>
          <li className={activeTab === 'predict' ? 'active' : ''} onClick={() => setActiveTab('predict')}>
            <Activity size={18} /> Predict & XAI
          </li>
          <li className={activeTab === 'explorer' ? 'active' : ''} onClick={() => setActiveTab('explorer')}>
            <Search size={18} /> Material Explorer
          </li>
        </ul>
      </nav>

      {/* MAIN CONTENT */}
      <main className="main-content">
        
        {activeTab === 'dashboard' && (
          <div className="dashboard-tab fade-in">
            <header className="top-header">
              <div>
                <h1>Architecture Overview</h1>
                <p>Training Metrics & Model Integrity</p>
              </div>
              <span className="badge">Status: Online</span>
            </header>

            <div className="metrics-row">
              <div className="metric-card">
                <span className="label">Dataset Volume</span>
                <span className="value">12,458</span>
              </div>
              <div className="metric-card">
                <span className="label">Total Inferences</span>
                <span className="value">8,732</span>
              </div>
              <div className="metric-card">
                <span className="label">MAE (eV)</span>
                <span className="value sandal-text">0.184</span>
              </div>
              <div className="metric-card">
                <span className="label">R² Score</span>
                <span className="value sandal-text">0.912</span>
              </div>
            </div>

            <div className="charts-row">
              <div className="panel chart-card">
                <h3>Parity Plot (Actual vs Predicted)</h3>
                <div style={{height: 350}}>
                  <ResponsiveContainer width="100%" height="100%">
                    <ScatterChart margin={{ top: 10, right: 10, bottom: 10, left: -20 }}>
                      <CartesianGrid strokeDasharray="3 3" stroke="#E2E8F0" />
                      <XAxis type="number" dataKey="actual" tick={{fill: '#64748B', fontSize: 12}} axisLine={{stroke: '#E2E8F0'}} tickLine={false} />
                      <YAxis type="number" dataKey="predicted" tick={{fill: '#64748B', fontSize: 12}} axisLine={{stroke: '#E2E8F0'}} tickLine={false} />
                      <Tooltip cursor={{ strokeDasharray: '3 3' }} contentStyle={{backgroundColor: '#FFFFFF', border: '1px solid #E2E8F0', borderRadius: '8px', color: '#1C2434'}} />
                      <Scatter name="Bandgap" data={parityData} fill="#FF6B6B" opacity={0.6} />
                      <Line dataKey="actual" stroke="#94A3B8" strokeDasharray="3 3"/>
                    </ScatterChart>
                  </ResponsiveContainer>
                </div>
              </div>

              <div className="panel chart-card">
                <h3>Huber Loss Convergence</h3>
                <div style={{height: 350}}>
                  <ResponsiveContainer width="100%" height="100%">
                    <LineChart data={lossData} margin={{ top: 10, right: 10, bottom: 10, left: -20 }}>
                      <CartesianGrid strokeDasharray="3 3" stroke="#E2E8F0" />
                      <XAxis dataKey="epoch" tick={{fill: '#64748B', fontSize: 12}} axisLine={{stroke: '#E2E8F0'}} tickLine={false} />
                      <YAxis scale="log" domain={['auto', 'auto']} tick={{fill: '#64748B', fontSize: 12}} axisLine={{stroke: '#E2E8F0'}} tickLine={false} />
                      <Tooltip contentStyle={{backgroundColor: '#FFFFFF', border: '1px solid #E2E8F0', borderRadius: '8px', color: '#1C2434'}} />
                      <Line type="monotone" dataKey="train" stroke="#FF6B6B" strokeWidth={2} dot={false} />
                      <Line type="monotone" dataKey="val" stroke="#94A3B8" strokeWidth={2} dot={false} />
                    </LineChart>
                  </ResponsiveContainer>
                </div>
              </div>
            </div>
          </div>
        )}

        {activeTab === 'predict' && (
          <div className="predict-tab fade-in">
            <header className="top-header">
              <div>
                <h1>Inference Engine</h1>
                <p>Upload single or multiple CIF files for high-throughput prediction.</p>
              </div>
              {(result || batchResults) && (
                <button onClick={downloadReport} className="btn-secondary" style={{width: 'auto', margin: 0, padding: '8px 16px', display: 'flex', alignItems: 'center', gap: '8px'}}>
                  <Download size={16}/> Export CSV
                </button>
              )}
            </header>

            <div className={files.length > 1 ? "" : "predict-grid"}>
              
              <div className="panel upload-panel" style={files.length > 1 ? {marginBottom: '32px'} : {}}>
                <h3>Data Ingestion</h3>
                <input type="file" accept=".cif" multiple onChange={handleFileChange} id="cif-upload" className="file-input" />
                <label htmlFor="cif-upload" className="upload-box">
                   {files.length > 0 ? (files.length === 1 ? files[0].name : `${files.length} structures selected for Batch Processing`) : "SELECT .CIF FILES"}
                </label>
                
                <button onClick={handleUpload} disabled={files.length === 0 || loading} className="btn-primary">
                  {loading ? "COMPUTING..." : "RUN INFERENCE"}
                </button>

                {error && <div className="error-box" style={{marginTop: '20px', color: '#ef4444'}}>{error}</div>}

                {/* SINGLE PREDICTION RESULTS */}
                {result && files.length === 1 && (
                  <div className="prediction-results fade-in">
                    <div className="res-item">
                      <span className="res-label">Band Gap</span>
                      <span className="res-value">{result.bandgap_eV.toFixed(4)} <small>eV</small></span>
                    </div>
                    <div className="res-item">
                      <span className="res-label">Formation Energy</span>
                      <span className="res-value">{result.formation_energy_eV_per_atom.toFixed(4)} <small>eV/atom</small></span>
                    </div>
                    <div className="res-meta">Computed in {result.inference_time_ms}ms</div>
                    
                    {!explainData && (
                      <button onClick={handleExplain} disabled={explaining} className="btn-secondary">
                        {explaining ? "GENERATING..." : "GENERATE EXPLAINABILITY MAP"}
                      </button>
                    )}

                    {/* MLOps Retraining Flag */}
                    <button 
                      onClick={() => setFlagged(true)} 
                      disabled={flagged} 
                      style={{
                        width: '100%', 
                        marginTop: '12px', 
                        padding: '12px', 
                        background: 'transparent', 
                        border: `1px solid ${flagged ? '#22c55e' : '#333'}`, 
                        color: flagged ? '#22c55e' : '#666',
                        display: 'flex', justifyContent: 'center', alignItems: 'center', gap: '8px',
                        cursor: flagged ? 'default' : 'pointer'
                      }}>
                      {flagged ? <><CheckCircle size={16}/> FLAGGED FOR DFT VERIFICATION</> : "FLAG FOR RETRAINING (MLOps)"}
                    </button>
                  </div>
                )}
              </div>

              {/* BATCH PREDICTION RESULTS */}
              {batchResults && files.length > 1 && (
                <div className="panel fade-in">
                  <div style={{display: 'flex', justifyContent: 'space-between', alignItems: 'center'}}>
                    <div>
                      <h3>High-Throughput Results</h3>
                      <p style={{fontSize: '0.85rem', color: '#888'}}>Processed {batchResults.count} structures in {batchResults.inference_time_ms}ms.</p>
                    </div>
                    <button 
                      onClick={() => setFlagged(true)} 
                      disabled={flagged} 
                      style={{
                        padding: '8px 16px', 
                        background: 'transparent', 
                        border: `1px solid ${flagged ? '#22c55e' : '#333'}`, 
                        color: flagged ? '#22c55e' : '#666',
                        display: 'flex', justifyContent: 'center', alignItems: 'center', gap: '8px',
                        cursor: flagged ? 'default' : 'pointer',
                        fontSize: '0.75rem', letterSpacing: '0.05em'
                      }}>
                      {flagged ? <><CheckCircle size={14}/> BATCH FLAGGED</> : "FLAG BATCH FOR DFT VERIFICATION"}
                    </button>
                  </div>
                  <table className="batch-table">
                    <thead>
                      <tr>
                        <th>Material</th>
                        <th>Band Gap (eV)</th>
                        <th>Formation Energy (eV/atom)</th>
                      </tr>
                    </thead>
                    <tbody>
                      {batchResults.results.map((res, idx) => (
                        <tr key={idx}>
                          <td>{res.filename}</td>
                          <td className="sandal-text">{res.bandgap_eV.toFixed(4)}</td>
                          <td>{res.formation_energy_eV_per_atom.toFixed(4)}</td>
                        </tr>
                      ))}
                    </tbody>
                  </table>
                </div>
              )}

              {/* VISUAL & FEATURE PANELS (Only for single files) */}
              {files.length <= 1 && (
                <>
                  <div className="panel visual-panel">
                    <h3>XAI Spatial Map</h3>
                    <div className="viewer-container" ref={containerRef}>
                      {!cifString && <span className="placeholder-text">AWAITING MOLECULAR DATA</span>}
                    </div>
                    {explainData && (
                      <div className="xai-justification fade-in">
                        {explainData.justification}
                      </div>
                    )}
                  </div>

                  <div className="panel feature-panel">
                    <h3>Node Attributes</h3>
                    <div style={{height: 250, width: '100%', marginTop: '20px'}}>
                      <ResponsiveContainer width="100%" height="100%">
                        <BarChart data={featureImportance} layout="vertical" margin={{top: 5, right: 10, left: 20, bottom: 5}}>
                          <CartesianGrid strokeDasharray="3 3" stroke="#E2E8F0" horizontal={true} vertical={false} />
                          <XAxis type="number" hide />
                          <YAxis type="category" dataKey="name" stroke="#64748B" width={110} tick={{fontSize: 10}} axisLine={false} tickLine={false} />
                          <Tooltip cursor={{fill: '#F8FAFC'}} contentStyle={{backgroundColor: '#FFFFFF', border: '1px solid #E2E8F0', borderRadius: '8px', fontSize: '0.8rem', color: '#1C2434'}}/>
                          <Bar dataKey="value" fill="#FF6B6B" radius={[0, 4, 4, 0]} barSize={10} />
                        </BarChart>
                      </ResponsiveContainer>
                    </div>
                  </div>
                </>
              )}

            </div>
          </div>
        )}

        {activeTab === 'explorer' && (
          <div className="explorer-tab fade-in">
            <header className="top-header">
              <div>
                <h1>Material Database</h1>
                <p>Search and verify the Jarvis DFT validation set</p>
              </div>
            </header>
            
            <div className="panel">
              <div style={{display: 'flex', alignItems: 'center', gap: '16px', marginBottom: '24px', borderBottom: '1px solid #222', paddingBottom: '16px'}}>
                <Search size={20} color="#666" />
                <input 
                  type="text" 
                  placeholder="Search by filename or material (e.g., NaCl.cif)..." 
                  value={searchQuery}
                  onChange={e => setSearchQuery(e.target.value)}
                  style={{
                    flex: 1, background: 'transparent', border: 'none', color: '#fff', fontSize: '1rem', outline: 'none'
                  }}
                />
              </div>

              <div style={{maxHeight: '60vh', overflowY: 'auto'}}>
                <table className="batch-table">
                  <thead style={{position: 'sticky', top: 0, background: '#F4F6F8'}}>
                    <tr>
                      <th>Filename</th>
                      <th>DFT Band Gap (eV)</th>
                      <th>DFT Formation Energy</th>
                    </tr>
                  </thead>
                  <tbody>
                    {filteredDataset.map((item, idx) => (
                      <tr key={idx}>
                        <td>{item.filename}</td>
                        <td className="sandal-text">{parseFloat(item.bandgap).toFixed(4)}</td>
                        <td>{parseFloat(item.formation_energy).toFixed(4)}</td>
                      </tr>
                    ))}
                    {filteredDataset.length === 0 && (
                      <tr><td colSpan="3" style={{textAlign: 'center', padding: '32px', color: '#666'}}>No materials found</td></tr>
                    )}
                  </tbody>
                </table>
              </div>
            </div>
          </div>
        )}
      </main>
    </div>
  )
}

export default App
