let frameWidth = 0,
  frameHeight = 0;

const es = new EventSource("http://localhost:8000/sse/count");
es.onmessage = (e) => {
  const d = JSON.parse(e.data);
  latestPoints = d.points;
  if (document.querySelector("#pplcounter") && d?.points?.length) {
    document.querySelector("#pplcounter").innerHTML = d.points.length;
  }
  if (frameWidth !== d.width || frameHeight !== d.height) {
    frameWidth = d.width;
    frameHeight = d.height;
    resizeCanvas(); // keep overlay in sync
  }

  processAndRender();
};
