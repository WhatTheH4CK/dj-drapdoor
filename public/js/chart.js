let snapshots = [];

slider.oninput = () => {
  intervalDisp.textContent = slider.value;
  if (running) startCapture();
};
const maxTimeSec = Infinity;          // or just remove it
function startCapture () {
  clearInterval(captureTimer);
  if (!snapshots.length) captureSnapshot();      // keep existing data
  captureTimer = setInterval(
      captureSnapshot,
      parseInt(slider.value, 10) * 1000          // slider in seconds
  );
}

function captureSnapshot() {
  const total = latestPoints.length || 1;
  let likes = 0, dis = 0;
  latestPoints.forEach(p => {
    const pt = mapPoint(p);
    areas.forEach(a => {
      if (pointInPoly(pt, a.points)) {
        if (a.type === 'like') likes++;
        if (a.type === 'dislike') dis++;
      }
    });
  });
  snapshots.push({
    time: Math.floor(elapsed / 1000),
    likePct: Math.round(likes / total * 100),
    dislikePct: Math.round(dis / total * 100)
  });
  renderChart();
}

function renderChart() {
  const chart = document.getElementById('chart');
  chart.innerHTML = '';
  snapshots.forEach(s => {
    const w = document.createElement('div');
    w.style.flex = '1';
    w.style.display = 'flex';
    w.style.flexDirection = 'column';
    w.style.alignItems = 'center';
    const bar = document.createElement('div');
    bar.style.display = 'flex';
    bar.style.flexDirection = 'column-reverse';
    bar.style.width = '20px';
    bar.style.height = '100%';
    const g = document.createElement('div');
    g.style.height = s.likePct + '%';
    g.style.width = '100%';
    g.style.background = 'green';
    const r = document.createElement('div');
    r.style.height = s.dislikePct + '%';
    r.style.width = '100%';
    r.style.background = 'red';
    bar.appendChild(g);
    bar.appendChild(r);
    const lbl = document.createElement('div');
    lbl.textContent = 'sec' + s.time;
    lbl.style.fontSize = '10px';
    w.appendChild(bar);
    w.appendChild(lbl);
    chart.appendChild(w);
  });
}