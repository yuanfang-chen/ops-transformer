const STORE_KEY = "npu_occupancy_data_v1";

const machineForm = document.getElementById("machine-form");
const machineList = document.getElementById("machine-list");
const machineSelect = document.getElementById("machine-select");
const viewDate = document.getElementById("view-date");
const timeline = document.getElementById("timeline");
const bookingForm = document.getElementById("booking-form");
const bookingHint = document.getElementById("booking-hint");
const startHourSelect = document.getElementById("start-hour");
const endHourSelect = document.getElementById("end-hour");

let state = loadState();

function todayISO() {
  const d = new Date();
  const y = d.getFullYear();
  const m = String(d.getMonth() + 1).padStart(2, "0");
  const day = String(d.getDate()).padStart(2, "0");
  return `${y}-${m}-${day}`;
}

function loadState() {
  const raw = localStorage.getItem(STORE_KEY);
  if (raw) return JSON.parse(raw);
  return { machines: [], bookings: [] };
}

function saveState() {
  localStorage.setItem(STORE_KEY, JSON.stringify(state));
}

function uid(prefix) {
  return `${prefix}_${Math.random().toString(36).slice(2, 9)}`;
}

function toHH(hour) {
  return `${String(hour).padStart(2, "0")}:00`;
}

function initHourOptions() {
  for (let h = 0; h <= 23; h++) {
    const o = document.createElement("option");
    o.value = String(h);
    o.textContent = toHH(h);
    startHourSelect.appendChild(o);
  }
  for (let h = 1; h <= 24; h++) {
    const o = document.createElement("option");
    o.value = String(h);
    o.textContent = h === 24 ? "24:00" : toHH(h);
    endHourSelect.appendChild(o);
  }
  startHourSelect.value = "9";
  endHourSelect.value = "11";
}

function machineById(id) {
  return state.machines.find((m) => m.id === id);
}

function bookingsOf(machineId, date) {
  return state.bookings.filter((b) => b.machineId === machineId && b.date === date);
}

function overlaps(aStart, aEnd, bStart, bEnd) {
  return Math.max(aStart, bStart) < Math.min(aEnd, bEnd);
}

function renderMachines() {
  machineList.innerHTML = "";
  machineSelect.innerHTML = "";

  state.machines.forEach((m) => {
    const li = document.createElement("li");
    li.textContent = `${m.name} | ${m.model} | ${m.location}${m.remark ? ` | ${m.remark}` : ""}`;
    machineList.appendChild(li);

    const opt = document.createElement("option");
    opt.value = m.id;
    opt.textContent = `${m.name} (${m.model})`;
    machineSelect.appendChild(opt);
  });

  if (state.machines.length === 0) {
    const empty = document.createElement("li");
    empty.textContent = "暂无机器，请先录入机器信息。";
    machineList.appendChild(empty);
  } else if (!machineById(machineSelect.value)) {
    machineSelect.value = state.machines[0].id;
  }
}

function renderTimeline() {
  timeline.innerHTML = "";
  const machineId = machineSelect.value;
  const date = viewDate.value;
  if (!machineId) {
    timeline.textContent = "请先录入并选择机器。";
    return;
  }

  const bs = bookingsOf(machineId, date);
  for (let h = 0; h < 24; h++) {
    const cell = document.createElement("div");
    cell.className = "slot free";

    const hit = bs.find((b) => h >= b.startHour && h < b.endHour);
    if (hit) cell.className = "slot busy";

    const hourDiv = document.createElement("div");
    hourDiv.className = "hour";
    hourDiv.textContent = toHH(h);
    cell.appendChild(hourDiv);

    const owner = document.createElement("div");
    owner.className = "owner";
    owner.textContent = hit ? `${hit.userName}` : "空闲";
    owner.title = hit ? `${hit.userName} | ${hit.contact} | ${hit.purpose}` : "空闲";
    cell.appendChild(owner);

    timeline.appendChild(cell);
  }
}

machineForm.addEventListener("submit", (e) => {
  e.preventDefault();
  const fd = new FormData(machineForm);
  const name = String(fd.get("name") || "").trim();
  const model = String(fd.get("model") || "").trim();
  const location = String(fd.get("location") || "").trim();
  const remark = String(fd.get("remark") || "").trim();

  if (!name || !model || !location) return;

  state.machines.push({ id: uid("machine"), name, model, location, remark });
  saveState();
  machineForm.reset();
  renderMachines();
  renderTimeline();
});

bookingForm.addEventListener("submit", (e) => {
  e.preventDefault();
  bookingHint.textContent = "";

  const machineId = machineSelect.value;
  if (!machineId) {
    bookingHint.textContent = "请先录入并选择机器。";
    return;
  }

  const fd = new FormData(bookingForm);
  const userName = String(fd.get("userName") || "").trim();
  const contact = String(fd.get("contact") || "").trim();
  const purpose = String(fd.get("purpose") || "").trim();
  const startHour = Number(fd.get("startHour"));
  const endHour = Number(fd.get("endHour"));
  const date = viewDate.value;

  if (!(startHour >= 0 && endHour <= 24 && startHour < endHour)) {
    bookingHint.textContent = "时间段不合法，请检查开始/结束时间。";
    return;
  }

  const bs = bookingsOf(machineId, date);
  const conflict = bs.find((b) => overlaps(startHour, endHour, b.startHour, b.endHour));
  if (conflict) {
    bookingHint.textContent = `冲突：${toHH(conflict.startHour)}-${toHH(conflict.endHour)} 已被 ${conflict.userName} 占用。`;
    return;
  }

  state.bookings.push({
    id: uid("booking"),
    machineId,
    date,
    startHour,
    endHour,
    userName,
    contact,
    purpose
  });

  saveState();
  bookingForm.reset();
  startHourSelect.value = "9";
  endHourSelect.value = "11";
  bookingHint.textContent = `提交成功：${date} ${toHH(startHour)}-${toHH(endHour)} 已占用。`;
  renderTimeline();
});

machineSelect.addEventListener("change", renderTimeline);
viewDate.addEventListener("change", renderTimeline);

function boot() {
  initHourOptions();
  viewDate.value = todayISO();
  renderMachines();
  renderTimeline();
}

boot();
