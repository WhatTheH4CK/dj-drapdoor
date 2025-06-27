let frameWidth = 0, frameHeight = 0;

const es = new EventSource('http://localhost:8000/sse/count');
es.onmessage = e => {
  const d = JSON.parse(e.data);
  latestPoints = d.points;
  frameWidth   = d.width;
  frameHeight  = d.height;
  processAndRender();
};
