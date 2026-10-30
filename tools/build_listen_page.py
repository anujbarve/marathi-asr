import base64
import json
from pathlib import Path

out = {}
for variant in ['v1', 'v2-lowram']:
    d = Path.home() / '.cache' / 'maya_fast' / 'listen' / variant
    key = json.loads((d / 'key.json').read_text(encoding='utf-8'))
    out[variant] = {}
    for k, item in key.items():
        a_b64 = base64.b64encode((d / f'pair_{k}_A.wav').read_bytes()).decode('ascii')
        b_b64 = base64.b64encode((d / f'pair_{k}_B.wav').read_bytes()).decode('ascii')
        out[variant][k] = {
            'id': item['id'],
            'text': item['text'],
            'candidate': item['candidate'],
            'audioA': 'data:audio/wav;base64,' + a_b64,
            'audioB': 'data:audio/wav;base64,' + b_b64
        }

html_content = f"""<!DOCTYPE html>
<html lang="mr">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Maya-Fast A/B Listening Test & Scorer</title>
<style>
  :root {{
    --bg: #0f172a;
    --card: #1e293b;
    --card-border: #334155;
    --primary: #38bdf8;
    --text: #f8fafc;
    --text-muted: #94a3b8;
    --accent: #a855f7;
    --success: #22c55e;
    --danger: #ef4444;
  }}
  * {{ box-sizing: border-box; margin: 0; padding: 0; }}
  body {{
    background: var(--bg);
    color: var(--text);
    font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Helvetica, Arial, sans-serif;
    padding: 24px;
    max-width: 900px;
    margin: 0 auto;
    line-height: 1.5;
  }}
  header {{
    margin-bottom: 24px;
    padding-bottom: 16px;
    border-bottom: 1px solid var(--card-border);
  }}
  h1 {{ font-size: 1.8rem; font-weight: 700; color: var(--primary); margin-bottom: 6px; }}
  p.subtitle {{ color: var(--text-muted); font-size: 0.95rem; }}
  .controls-bar {{
    display: flex;
    justify-content: space-between;
    align-items: center;
    background: var(--card);
    padding: 12px 18px;
    border-radius: 10px;
    border: 1px solid var(--card-border);
    margin-bottom: 24px;
    flex-wrap: wrap;
    gap: 12px;
  }}
  .variant-selector {{
    display: flex;
    align-items: center;
    gap: 10px;
  }}
  select, button {{
    background: #0f172a;
    color: var(--text);
    border: 1px solid var(--card-border);
    padding: 8px 14px;
    border-radius: 6px;
    font-size: 0.95rem;
    cursor: pointer;
    outline: none;
  }}
  select:focus, button:focus {{
    border-color: var(--primary);
  }}
  button.primary-btn {{
    background: var(--primary);
    color: #0f172a;
    font-weight: 600;
    border: none;
    transition: opacity 0.2s;
  }}
  button.primary-btn:hover {{ opacity: 0.9; }}
  .progress-badge {{
    font-size: 0.9rem;
    color: var(--primary);
    font-weight: 600;
  }}
  .pair-card {{
    background: var(--card);
    border: 1px solid var(--card-border);
    border-radius: 12px;
    padding: 20px;
    margin-bottom: 20px;
    transition: border-color 0.2s;
  }}
  .pair-card.completed {{
    border-color: #059669;
  }}
  .pair-header {{
    display: flex;
    justify-content: space-between;
    margin-bottom: 12px;
  }}
  .pair-id {{
    font-weight: 700;
    color: var(--accent);
    font-size: 1.05rem;
  }}
  .pair-cat {{
    background: #334155;
    padding: 2px 8px;
    border-radius: 4px;
    font-size: 0.75rem;
    text-transform: uppercase;
    letter-spacing: 0.5px;
  }}
  .sentence-text {{
    font-size: 1.25rem;
    margin-bottom: 18px;
    color: #f1f5f9;
    font-weight: 500;
    background: rgba(15, 23, 42, 0.4);
    padding: 12px 14px;
    border-radius: 8px;
    border-left: 3px solid var(--primary);
  }}
  .audio-grid {{
    display: grid;
    grid-template-columns: 1fr 1fr;
    gap: 16px;
    margin-bottom: 18px;
  }}
  @media (max-width: 640px) {{
    .audio-grid {{ grid-template-columns: 1fr; }}
  }}
  .audio-item {{
    background: rgba(15, 23, 42, 0.6);
    padding: 12px;
    border-radius: 8px;
    display: flex;
    flex-direction: column;
    gap: 8px;
  }}
  .audio-label {{
    font-weight: 600;
    font-size: 0.9rem;
    color: var(--text-muted);
  }}
  audio {{
    width: 100%;
    height: 36px;
    outline: none;
  }}
  .vote-options {{
    display: flex;
    gap: 12px;
  }}
  .vote-btn {{
    flex: 1;
    padding: 10px;
    border: 1px solid var(--card-border);
    border-radius: 8px;
    background: #0f172a;
    color: var(--text);
    font-size: 0.95rem;
    font-weight: 600;
    cursor: pointer;
    transition: all 0.2s;
  }}
  .vote-btn:hover {{
    border-color: var(--primary);
  }}
  .vote-btn.selected {{
    background: var(--primary);
    color: #0f172a;
    border-color: var(--primary);
  }}
  .vote-btn.tie-btn.selected {{
    background: #94a3b8;
    color: #0f172a;
    border-color: #94a3b8;
  }}
  .results-section {{
    background: var(--card);
    border: 1px solid var(--card-border);
    border-radius: 12px;
    padding: 24px;
    margin-top: 30px;
    display: none;
  }}
  .results-section.visible {{ display: block; }}
  .score-card {{
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(160px, 1fr));
    gap: 14px;
    margin: 16px 0;
  }}
  .metric-box {{
    background: #0f172a;
    padding: 16px;
    border-radius: 8px;
    text-align: center;
    border: 1px solid var(--card-border);
  }}
  .metric-value {{
    font-size: 1.8rem;
    font-weight: 700;
    margin-top: 4px;
  }}
  .metric-label {{
    font-size: 0.8rem;
    color: var(--text-muted);
    text-transform: uppercase;
  }}
  .gate-badge {{
    display: inline-block;
    padding: 6px 14px;
    border-radius: 9999px;
    font-weight: 700;
    font-size: 0.9rem;
    margin-top: 10px;
  }}
  .gate-pass {{ background: rgba(34, 197, 94, 0.2); color: var(--success); border: 1px solid var(--success); }}
  .gate-fail {{ background: rgba(239, 68, 68, 0.2); color: var(--danger); border: 1px solid var(--danger); }}
  .breakdown-table {{
    width: 100%;
    margin-top: 16px;
    border-collapse: collapse;
    font-size: 0.9rem;
  }}
  .breakdown-table th, .breakdown-table td {{
    padding: 10px;
    text-align: left;
    border-bottom: 1px solid var(--card-border);
  }}
  .breakdown-table th {{ color: var(--text-muted); }}
</style>
</head>
<body>

<header>
  <h1>🎧 Maya-Fast Blind A/B Listening Test</h1>
  <p class="subtitle">Blind evaluation of Maya-Fast ONNX candidate vs. Upstream PyTorch bf16 Reference. Loudness is normalized.</p>
</header>

<div class="controls-bar">
  <div class="variant-selector">
    <label for="variantSelect"><strong>Asset Set:</strong></label>
    <select id="variantSelect" onchange="switchVariant()">
      <option value="v1">v1 (wo8 backbone + fp16 depth, ~2.2 RTF)</option>
      <option value="v2-lowram">v2-lowram (wo8 backbone + q8a1_kg8 depth, ~3.8 GB RAM)</option>
    </select>
  </div>
  <div id="progressText" class="progress-badge">0 / 10 Rated</div>
  <div>
    <button class="primary-btn" onclick="calculateScore()">Reveal & Score Results</button>
  </div>
</div>

<div id="pairsContainer"></div>

<div id="resultsSection" class="results-section">
  <h2>📊 Test Results & Gate Check</h2>
  <p id="gateStatusNotice" style="margin-top: 4px; color: var(--text-muted);"></p>
  
  <div class="score-card">
    <div class="metric-box">
      <div class="metric-label">Candidate Favored</div>
      <div id="candScore" class="metric-value" style="color: var(--primary);">0</div>
    </div>
    <div class="metric-box">
      <div class="metric-label">Tied</div>
      <div id="tieScore" class="metric-value" style="color: #94a3b8;">0</div>
    </div>
    <div class="metric-box">
      <div class="metric-label">Reference Favored</div>
      <div id="refScore" class="metric-value" style="color: var(--accent);">0</div>
    </div>
    <div class="metric-box">
      <div class="metric-label">Combined Rate (Gate &ge; 45%)</div>
      <div id="rateScore" class="metric-value">0%</div>
    </div>
  </div>

  <div id="gateBadgeContainer" style="text-align: center; margin-bottom: 16px;"></div>

  <h3>Unblinded Breakdown:</h3>
  <table class="breakdown-table">
    <thead>
      <tr>
        <th>Pair</th>
        <th>Text</th>
        <th>Candidate Was</th>
        <th>Your Choice</th>
        <th>Outcome</th>
      </tr>
    </thead>
    <tbody id="breakdownBody"></tbody>
  </table>
  <div style="margin-top: 18px; display: flex; gap: 10px;">
    <button onclick="downloadAnswers()">Download answers.json</button>
    <button onclick="copyAnswers()">Copy JSON to Clipboard</button>
  </div>
</div>

<script>
const DATA = {json.dumps(out, ensure_ascii=False)};

let currentVariant = 'v1';
let userVotes = {{ 'v1': {{}}, 'v2-lowram': {{}} }};

function renderPairs() {{
  const container = document.getElementById('pairsContainer');
  container.innerHTML = '';
  const pairs = DATA[currentVariant];
  const votes = userVotes[currentVariant];

  for (const [pid, item] of Object.entries(pairs)) {{
    const card = document.createElement('div');
    card.className = 'pair-card' + (votes[pid] ? ' completed' : '');
    card.id = 'card-' + pid;

    const vote = votes[pid] || null;

    card.innerHTML = `
      <div class="pair-header">
        <span class="pair-id">Pair #${{pid}} (${{item.id}})</span>
        <span class="pair-cat">Marathi</span>
      </div>
      <div class="sentence-text">${{item.text}}</div>
      <div class="audio-grid">
        <div class="audio-item">
          <span class="audio-label">Option A</span>
          <audio controls preload="auto" src="${{item.audioA}}"></audio>
        </div>
        <div class="audio-item">
          <span class="audio-label">Option B</span>
          <audio controls preload="auto" src="${{item.audioB}}"></audio>
        </div>
      </div>
      <div class="vote-options">
        <button class="vote-btn ${{vote === 'A' ? 'selected' : ''}}" onclick="castVote('${{pid}}', 'A')">Prefer A</button>
        <button class="vote-btn tie-btn ${{vote === 'tie' ? 'selected' : ''}}" onclick="castVote('${{pid}}', 'tie')">Tie / Equal</button>
        <button class="vote-btn ${{vote === 'B' ? 'selected' : ''}}" onclick="castVote('${{pid}}', 'B')">Prefer B</button>
      </div>
    `;
    container.appendChild(card);
  }}
  updateProgress();
}}

function castVote(pid, choice) {{
  userVotes[currentVariant][pid] = choice;
  renderPairs();
}}

function updateProgress() {{
  const votes = userVotes[currentVariant];
  const total = Object.keys(DATA[currentVariant]).length;
  const count = Object.keys(votes).length;
  document.getElementById('progressText').innerText = `${{count}} / ${{total}} Rated`;
}}

function switchVariant() {{
  currentVariant = document.getElementById('variantSelect').value;
  document.getElementById('resultsSection').classList.remove('visible');
  renderPairs();
}}

function calculateScore() {{
  const pairs = DATA[currentVariant];
  const votes = userVotes[currentVariant];
  const total = Object.keys(pairs).length;
  const votedCount = Object.keys(votes).length;

  if (votedCount < total) {{
    if (!confirm(`You have rated ${{votedCount}} of ${{total}} pairs. Reveal anyway?`)) {{
      return;
    }}
  }}

  let candPref = 0;
  let tie = 0;
  let refPref = 0;
  const breakdownRows = [];

  for (const [pid, item] of Object.entries(pairs)) {{
    const userChoice = votes[pid];
    const candSide = item.candidate;
    let outcome = 'Unanswered';
    let outcomeColor = 'var(--text-muted)';

    if (userChoice === 'tie') {{
      tie++;
      outcome = 'Tie';
      outcomeColor = '#94a3b8';
    }} else if (userChoice === candSide) {{
      candPref++;
      outcome = 'Candidate Preferred';
      outcomeColor = 'var(--success)';
    }} else if (userChoice) {{
      refPref++;
      outcome = 'Reference Preferred';
      outcomeColor = 'var(--accent)';
    }}

    breakdownRows.push(`
      <tr>
        <td><strong>#${{pid}}</strong></td>
        <td>${{item.text}}</td>
        <td><strong>${{candSide}}</strong></td>
        <td><strong>${{userChoice || 'None'}}</strong></td>
        <td style="color: ${{outcomeColor}}; font-weight: 600;">${{outcome}}</td>
      </tr>
    `);
  }}

  const ratedTotal = candPref + tie + refPref;
  const rate = ratedTotal > 0 ? (candPref + tie) / ratedTotal : 0;
  const ratePercent = (rate * 100).toFixed(1) + '%';
  const passed = rate >= 0.45;

  document.getElementById('candScore').innerText = candPref;
  document.getElementById('tieScore').innerText = tie;
  document.getElementById('refScore').innerText = refPref;
  document.getElementById('rateScore').innerText = ratePercent;
  document.getElementById('rateScore').style.color = passed ? 'var(--success)' : 'var(--danger)';

  const badgeHtml = passed
    ? `<span class="gate-badge gate-pass">✓ PASS: Gate Met (&ge; 45% Candidate Preferred / Tied)</span>`
    : `<span class="gate-badge gate-fail">✗ FAIL: Below 45% Threshold</span>`;
  document.getElementById('gateBadgeContainer').innerHTML = badgeHtml;
  document.getElementById('breakdownBody').innerHTML = breakdownRows.join('');
  document.getElementById('resultsSection').classList.add('visible');
  document.getElementById('resultsSection').scrollIntoView({{ behavior: 'smooth' }});
}}

function downloadAnswers() {{
  const dataStr = "data:text/json;charset=utf-8," + encodeURIComponent(JSON.stringify(userVotes[currentVariant], null, 2));
  const a = document.createElement('a');
  a.setAttribute("href", dataStr);
  a.setAttribute("download", `answers_${{currentVariant}}.json`);
  document.body.appendChild(a);
  a.click();
  a.remove();
}}

function copyAnswers() {{
  navigator.clipboard.writeText(JSON.stringify(userVotes[currentVariant], null, 2))
    .then(() => alert('Answers copied to clipboard!'));
}}

renderPairs();
</script>
</body>
</html>"""

out_file = Path('/Users/anujbarve/Documents/marathi-helper/listening_test.html')
out_file.write_text(html_content, encoding='utf-8')
print("Successfully generated:", out_file)
