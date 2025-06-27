let areas = [], current = null, tempPts = [], latestPoints = [];
let previewPt = null;

drawBtn.onclick = () => {
  current = { points: [], type: null, history: [] };
  tempPts = [];
  previewPt = null;
  drawBtn.disabled = true;
  canvas.style.pointerEvents = 'auto';
};

function getCanvasPos(e) {
  const r  = canvas.getBoundingClientRect();
  const sx = canvas.width  / r.width;   // per-axis scale factors
  const sy = canvas.height / r.height;
  return {
    x: (e.clientX - r.left) * sx,
    y: (e.clientY - r.top)  * sy
  };
}
video.onload = resizeCanvas;
window.onresize = resizeCanvas;

canvas.onclick = e => {
  if (!current) return;
  const { x, y } = getCanvasPos(e);
  tempPts.push({ x, y });

  if (tempPts.length === 4) {          // rectangle finished
    current.points = tempPts.slice();
    areas.push(current);
    addAreaUI(current);

    current   = null;
    tempPts   = [];
    previewPt = null;
    drawBtn.disabled = false;
    canvas.style.pointerEvents = 'none';
  }
  render();
};

canvas.onmousemove = e => {
  if (!current) return;
  previewPt = getCanvasPos(e);
  render();
};




function render() {
  ctx.clearRect(0, 0, canvas.width, canvas.height);
  areas.forEach(a => {
    ctx.beginPath();
    a.points.forEach((p, i) => i ? ctx.lineTo(p.x, p.y) : ctx.moveTo(p.x, p.y));
    ctx.closePath();
    ctx.strokeStyle = a.type === 'dislike' ? 'red' : a.type === 'like' ? 'green' : 'blue';
    ctx.lineWidth = 2;
    ctx.stroke();
  });
  if (tempPts.length) {
    ctx.beginPath();
    tempPts.forEach((p, i) => i ? ctx.lineTo(p.x, p.y) : ctx.moveTo(p.x, p.y));
    if (previewPt) ctx.lineTo(previewPt.x, previewPt.y);
    ctx.setLineDash([10, 10]);
    ctx.lineWidth  = 3;          // thicker = easier to see
    ctx.strokeStyle = 'orange';
    ctx.stroke();
    ctx.setLineDash([]);
  }
  latestPoints.forEach(p => {
    const { x, y } = mapPoint(p);
    let col = 'blue';
    for (const a of areas) {
      if (pointInPoly({ x, y }, a.points)) {
        col = a.type === 'like' ? 'green' : a.type === 'dislike' ? 'red' : 'blue';
        break;
      }
    }
    ctx.beginPath();
    ctx.arc(x, y, 4, 0, 2 * Math.PI);
    ctx.fillStyle = col;
    ctx.fill();
  });
}

function pointInPoly(pt, poly) {
  let inside = false;
  for (let i = 0, j = poly.length - 1; i < poly.length; j = i++) {
    const xi = poly[i].x, yi = poly[i].y,
          xj = poly[j].x, yj = poly[j].y;
    const intersect = ((yi > pt.y) != (yj > pt.y))
      && (pt.x < (xj - xi) * (pt.y - yi) / (yj - yi) + xi);
    if (intersect) inside = !inside;
  }
  return inside;
}
function resizeCanvas() {
  // Wait until the backend has sent frame size once
  if (!frameWidth || !frameHeight) return;

  const prevW = canvas.width  || 1;
  const prevH = canvas.height || 1;

  const cw    = video.clientWidth;
  const ratio = frameHeight / frameWidth;
  const ch    = cw * ratio;

  // Rescale all saved geometry when the canvas changes size
  if (prevW !== cw || prevH !== ch) {
    const sx = cw / prevW;
    const sy = ch / prevH;

    areas.forEach(a =>
      a.points.forEach(p => { p.x *= sx; p.y *= sy; })
    );
    tempPts.forEach(p => { p.x *= sx; p.y *= sy; });
    if (previewPt) { previewPt.x *= sx; previewPt.y *= sy; }
  }

  video.parentElement.style.height = ch + 'px';
  canvas.width  = cw;
  canvas.height = ch;
  render();
}

function mapPoint(p) {
  return {
    x: p[0] * (canvas.width  / frameWidth),
    y: p[1] * (canvas.height / frameHeight)
  };
}
