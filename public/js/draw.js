let areas = [], current = null, tempPts = [], latestPoints = [];
let previewPt = null;

drawBtn.onclick = () => {
  current = { points: [], type: null, history: [] };
  tempPts = [];
  previewPt = null;
  drawBtn.disabled = true;
  canvas.style.pointerEvents = 'auto';
};
function resizeCanvas(){
  const cw    = video.clientWidth,
        ratio = frameHeight / frameWidth,
        ch    = cw * ratio;
  video.parentElement.style.height = ch + 'px';
  canvas.width  = cw;
  canvas.height = ch;
  render();
}

video.onload = resizeCanvas;
window.onresize = resizeCanvas;

canvas.onclick = e => {
  if (!current) return;
  const r = canvas.getBoundingClientRect(),
        x = e.clientX - r.left,
        y = e.clientY - r.top;
  tempPts.push({ x, y });
  if (tempPts.length === 4) {
    current.points = tempPts.slice();
    areas.push(current);
    addAreaUI(current);
    current = null;
    tempPts = [];
    previewPt = null;
    drawBtn.disabled = false;
    canvas.style.pointerEvents = 'none';
  }
  render();
};

canvas.onmousemove = e => {
  if (!current) return;
  const r = canvas.getBoundingClientRect(),
        x = e.clientX - r.left,
        y = e.clientY - r.top;
  previewPt = { x, y };
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
    ctx.setLineDash([5, 5]);
    ctx.strokeStyle = 'gray';
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
function mapPoint(p){
  // p = [x_px, y_px] in the resized-frame coordinate space (Wn×Hn)
  const s = canvas.width / frameWidth;
  return { x: p[0] * s, y: p[1] * s };
}
