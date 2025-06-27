let running = false, startTime = 0, elapsed = 0, timerInt, captureTimer;

function formatTime(ms) {
  const t = ms / 1000;
  return String(Math.floor(t / 60)).padStart(2, '0') + ':' + (t % 60).toFixed(2).padStart(5, '0');
}

startBtn.onclick = () => {
  if (!running) {
    running = true;
    startTime = Date.now() - elapsed;
    timerInt = setInterval(() => {
      elapsed = Date.now() - startTime;
      timerSpan.textContent = formatTime(elapsed);
    }, 50);
    startCapture();
  }
};

pauseBtn.onclick = () => {
  if (running) {
    running = false;
    clearInterval(timerInt);
    clearInterval(captureTimer);
    clearInterval(countdownInterval);
    timerSpan.textContent = formatTime(elapsed);
    critStart = null;
  }
};

resetBtn.onclick = () => {
  running = false;
  clearInterval(timerInt);
  clearInterval(captureTimer);
  clearInterval(countdownInterval);
  elapsed = 0;
  timerSpan.textContent = formatTime(0);
  snapshots = [];
  renderChart();
  areas.forEach(a => {
    a.history = [];
    a.div.querySelector('.sessionPercent').textContent = '0%';
  });
  critStart = null;
  alarm.style.display = 'none';
  gotcha.style.display = 'none';
};