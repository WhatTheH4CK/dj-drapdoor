let critStart = null,
  countdownInterval = null;
function processAndRender() {
  /* ── update area percentages (unchanged) ───────────── */
  const total = latestPoints.length || 1;

  areas.forEach((a) => {
    const cnt = latestPoints.filter((p) => pointInPoly(mapPoint(p), a.points)).length;
    const pct = Math.round((cnt / total) * 100);
    a.div.querySelector(".percent").textContent = pct + "%";

    if (running) {
      a.history = a.history || [];
      a.history.push(pct);
      const avg = Math.round(a.history.reduce((u, v) => u + v, 0) / a.history.length);
      a.div.querySelector(".sessionPercent").textContent = avg + "%" + ' ('+cnt+')';
    }
  });

  /* ── critical-mass check: dislikes vs. likes ───────── */
  const disCount = latestPoints.filter((p) =>
    areas.some((a) => a.type === "dislike" && pointInPoly(mapPoint(p), a.points))
  ).length;

  const likeCount = latestPoints.filter((p) =>
    areas.some((a) => a.type === "like" && pointInPoly(mapPoint(p), a.points))
  ).length;

  /* % of dislikes COMPARED TO likes (not overall) */
  const dislikePct = likeCount ? (disCount / likeCount) * 100 : 0;
  const minPct     = +minPctInput.value;          // critical-mass input (%)
  const neededMs   = +critDurIn.value * 1000;     // sustain duration (ms)

  if (running && dislikePct >= minPct) {
    if (!critStart) critStart = Date.now();

    if (Date.now() - critStart >= neededMs && !countdownInterval) {
      let rem = +cdSecInput.value;
      alarm.style.display = "block";
      alarmTm.textContent = `${String(Math.floor(rem / 60)).padStart(2, "0")}:${String(rem % 60).padStart(2, "0")}`;

      countdownInterval = setInterval(() => {
        rem--;
        if (rem <= 0) {
          clearInterval(countdownInterval);
          alarm.style.display = "none";
          gotcha.style.display = "block";
          running = false;
          clearInterval(timerInt);
          clearInterval(captureTimer);
        } else {
          alarmTm.textContent = `${String(Math.floor(rem / 60)).padStart(2, "0")}:${String(rem % 60).padStart(2, "0")}`;
        }
      }, 1000);
    }
  } else {
    critStart = null;
    alarm.style.display = "none";
    if (countdownInterval) {
      clearInterval(countdownInterval);
      countdownInterval = null;
      gotcha.style.display = "none";
    }
  }

  render();
}
