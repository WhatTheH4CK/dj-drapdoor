let critStart = null, countdownInterval = null;

function processAndRender() {
  const total = latestPoints.length || 1;
  areas.forEach(a => {
    const cnt = latestPoints.filter(p => pointInPoly(mapPoint(p), a.points)).length;
    const pct = Math.round(cnt / total * 100);
    a.div.querySelector('.percent').textContent = pct + '%';
    if (running) {
      a.history = a.history || [];
      a.history.push(pct);
      const avg = Math.round(a.history.reduce((u, v) => u + v, 0) / a.history.length);
      a.div.querySelector('.sessionPercent').textContent = avg + '%';
    }
  });

  const disCount = latestPoints.filter(p =>
    areas.some(a => a.type === 'dislike' && pointInPoly(mapPoint(p), a.points))
  ).length;
  const dislikePct = Math.round(disCount / total * 100);
  const minPct = +minPctInput.value;
  const neededMs = +critDurIn.value * 1000;

  if (running && dislikePct >= minPct) {
    if (!critStart) critStart = Date.now();
    if (Date.now() - critStart >= neededMs && !countdownInterval) {
      let rem = +cdSecInput.value;
      alarm.style.display = 'block';
      alarmTm.textContent = `${String(Math.floor(rem/60)).padStart(2,'0')}:${String(rem%60).padStart(2,'0')}`;
      countdownInterval = setInterval(() => {
        rem--;
        if (rem <= 0) {
          clearInterval(countdownInterval);
          alarm.style.display = 'none';
          gotcha.style.display = 'block';
          running = false;
          clearInterval(timerInt);
          clearInterval(captureTimer);
        } else {
          alarmTm.textContent = `${String(Math.floor(rem/60)).padStart(2,'0')}:${String(rem%60).padStart(2,'0')}`;
        }
      }, 1000);
    }
  } else {
    critStart = null;
    clearInterval(countdownInterval);
    countdownInterval = null;
    alarm.style.display = 'none';
    gotcha.style.display = 'none';
  }

  render();
}