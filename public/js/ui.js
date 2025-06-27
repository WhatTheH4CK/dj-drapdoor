const video = document.getElementById("video"),
  canvas = document.getElementById("overlay"),
  ctx = canvas.getContext("2d"),
  drawBtn = document.getElementById("drawBtn"),
  list = document.getElementById("areas"),
  startBtn = document.getElementById("startBtn"),
  pauseBtn = document.getElementById("pauseBtn"),
  resetBtn = document.getElementById("resetBtn"),
  timerSpan = document.getElementById("timer"),
  slider = document.getElementById("intervalSlider"),
  intervalDisp = document.getElementById("intervalDisplay"),
  minPctInput = document.getElementById("minPct"),
  critDurIn = document.getElementById("critDur"),
  cdSecInput = document.getElementById("countdownSec"),
  alarm = document.getElementById("alarm"),
  alarmTm = document.getElementById("alarmTimer"),
  gotcha = document.getElementById("gotcha");

const fsBtn = document.getElementById("fsBtn");
fsBtn.onclick = () => {
  const container = document.getElementById("video-container");
  if (document.fullscreenElement) {
    document.exitFullscreen();
  } else {
    container.requestFullscreen();
  }
};
function addAreaUI(a) {
  const div = document.createElement("div");
  div.className = "area-item";
  div.innerHTML = `
    <div class="d-flex flex-row justify-content-center ">
      <button class="likeBtn">Like</button>
      <button class="dislikeBtn">Dislike</button>
    </div>
    <div class="w-100 d-flex flex-row justify-content-around">
      <div class="d-flex flex-row m-4">
        <span>Currently: </span> <span class="percent">0%</span>
      </div>

      <div class="d-flex flex-row m-4">
        <span>Over time: </span> <span class="sessionPercent">0%</span>
      </div>
    </div>
    <button class="removeBtn w-100">Remove</button>
  `;
  const likeBtn = div.querySelector(".likeBtn"),
    dislikeBtn = div.querySelector(".dislikeBtn"),
    removeBtn = div.querySelector(".removeBtn");

  likeBtn.onclick = () => {
    a.type = "like";
    likeBtn.classList.add("selected");
    dislikeBtn.classList.remove("selected");
    render();
  };
  dislikeBtn.onclick = () => {
    a.type = "dislike";
    dislikeBtn.classList.add("selected");
    likeBtn.classList.remove("selected");
    render();
  };
  removeBtn.onclick = () => {
    areas = areas.filter((x) => x !== a);
    list.removeChild(div);
    render();
  };

  a.div = div;
  list.appendChild(div);
}
