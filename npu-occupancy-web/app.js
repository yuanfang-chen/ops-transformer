const machineForm = document.getElementById("machine-form");
const machineList = document.getElementById("machine-list");
const machineSelect = document.getElementById("machine-select");
const openMachineModalBtn = document.getElementById("open-machine-modal");
const closeMachineModalBtn = document.getElementById("close-machine-modal");
const machineModal = document.getElementById("machine-modal");
const viewDate = document.getElementById("view-date");
const timeline = document.getElementById("timeline");
const bookingForm = document.getElementById("booking-form");
const bookingMachineSelect = document.getElementById("booking-machine-select");
const bookingHint = document.getElementById("booking-hint");
const startHourSelect = document.getElementById("start-hour");
const endHourSelect = document.getElementById("end-hour");
const currentUserLabel = document.getElementById("current-user-label");
const switchUserBtn = document.getElementById("switch-user-btn");
const userModal = document.getElementById("user-modal");
const userForm = document.getElementById("user-form");
const userNameInput = document.getElementById("user-name-input");
const backendStatusLabel = document.getElementById("backend-status-label");

let state = { machines: [], bookings: [] };
let currentUser = "";
let sb = null;
const tooltipEl = createTooltip();
let hideTooltipTimer = null;

function todayISO() {
  const d = new Date();
  const y = d.getFullYear();
  const m = String(d.getMonth() + 1).padStart(2, "0");
  const day = String(d.getDate()).padStart(2, "0");
  return `${y}-${m}-${day}`;
}

function saveCurrentUser(name) {
  currentUser = name.trim();
  currentUserLabel.textContent = `当前用户：${currentUser || "-"}`;
}

function isAdminUser() {
  return currentUser === "xukenan";
}

function uid(prefix) {
  return `${prefix}_${Math.random().toString(36).slice(2, 9)}`;
}

function toHH(hour) {
  return `${String(hour).padStart(2, "0")}:00`;
}

function initSupabase() {
  const cfg = window.NPU_APP_CONFIG || {};
  if (!cfg.supabaseUrl || !cfg.supabaseAnonKey) {
    backendStatusLabel.textContent = "后端状态：未配置 Supabase（请填 config.js）";
    return false;
  }
  sb = window.supabase.createClient(cfg.supabaseUrl, cfg.supabaseAnonKey);
  backendStatusLabel.textContent = "后端状态：Supabase 已连接";
  return true;
}

async function refreshStateFromCloud() {
  const [{ data: machines, error: mErr }, { data: bookings, error: bErr }] = await Promise.all([
    sb.from("machines").select("*").order("created_at", { ascending: true }),
    sb.from("bookings").select("*").order("created_at", { ascending: true })
  ]);
  if (mErr || bErr) {
    throw new Error(`加载失败: ${mErr?.message || ""} ${bErr?.message || ""}`.trim());
  }

  state.machines = (machines || []).map((m) => ({
    id: m.id,
    name: m.name,
    model: m.model,
    location: m.location,
    remark: m.remark || ""
  }));

  state.bookings = (bookings || []).map((b) => ({
    id: b.id,
    machineId: b.machine_id,
    date: b.date,
    startHour: b.start_hour,
    endHour: b.end_hour,
    userName: b.user_name,
    purpose: b.purpose
  }));
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

function setMachineModal(open) {
  machineModal.classList.toggle("hidden", !open);
}

function setUserModal(open) {
  userModal.classList.toggle("hidden", !open);
  if (open) {
    userNameInput.value = currentUser || "";
    setTimeout(() => userNameInput.focus(), 0);
  }
}

function ensureUserReady() {
  if (!currentUser) {
    setUserModal(true);
  }
}

function renderMachines() {
  machineList.innerHTML = "";
  machineSelect.innerHTML = "";
  bookingMachineSelect.innerHTML = "";

  state.machines.forEach((m) => {
    const li = document.createElement("li");
    li.textContent = `${m.name} | ${m.model} | ${m.location}${m.remark ? ` | ${m.remark}` : ""}`;
    machineList.appendChild(li);

    const opt = document.createElement("option");
    opt.value = m.id;
    opt.textContent = `${m.name} (${m.model})`;
    machineSelect.appendChild(opt);

    const bookingOpt = document.createElement("option");
    bookingOpt.value = m.id;
    bookingOpt.textContent = `${m.name} (${m.model})`;
    bookingMachineSelect.appendChild(bookingOpt);
  });

  if (state.machines.length === 0) {
    const empty = document.createElement("li");
    empty.textContent = "暂无机器，请先录入机器信息。";
    machineList.appendChild(empty);
  } else if (!machineById(machineSelect.value)) {
    machineSelect.value = state.machines[0].id;
    bookingMachineSelect.value = state.machines[0].id;
  } else if (!machineById(bookingMachineSelect.value)) {
    bookingMachineSelect.value = state.machines[0].id;
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
    owner.removeAttribute("title");
    attachTooltip(cell, hit, h);
    cell.appendChild(owner);

    timeline.appendChild(cell);
  }
}

function createTooltip() {
  const el = document.createElement("div");
  el.className = "nice-tooltip hidden";
  document.body.appendChild(el);
  el.addEventListener("mouseenter", () => clearHideTooltipTimer());
  el.addEventListener("mouseleave", () => hideTooltip());
  el.addEventListener("click", (e) => {
    const btn = e.target.closest("[data-cancel-booking-id]");
    if (!btn) return;
    const bookingId = btn.getAttribute("data-cancel-booking-id");
    const cancelHour = Number(btn.getAttribute("data-cancel-hour"));
    cancelSingleHourBooking(bookingId, cancelHour);
  });
  return el;
}

function clearHideTooltipTimer() {
  if (hideTooltipTimer) {
    clearTimeout(hideTooltipTimer);
    hideTooltipTimer = null;
  }
}

function hideTooltip(delayMs = 0) {
  clearHideTooltipTimer();
  hideTooltipTimer = setTimeout(() => {
    tooltipEl.classList.add("hidden");
  }, delayMs);
}

function placeTooltip(mouseX, mouseY) {
  const offset = 14;
  let left = mouseX + offset;
  let top = mouseY + offset;
  const rect = tooltipEl.getBoundingClientRect();

  if (left + rect.width > window.innerWidth - 8) {
    left = mouseX - rect.width - offset;
  }
  if (top + rect.height > window.innerHeight - 8) {
    top = mouseY - rect.height - offset;
  }

  tooltipEl.style.left = `${Math.max(8, left)}px`;
  tooltipEl.style.top = `${Math.max(8, top)}px`;
}

function setTooltipContent(hit, hour) {
  if (!hit) {
    tooltipEl.textContent = `状态：空闲\n时间段：${toHH(hour)}-${toHH(hour + 1)}`;
    return;
  }
  const canCancel = currentUser && (hit.userName === currentUser || isAdminUser());
  const cancelBtnText = isAdminUser() && hit.userName !== currentUser ? "管理员取消该小时占用" : "取消我的该小时占用";
  tooltipEl.innerHTML = `
    <div>占用人：${hit.userName}</div>
    <div>占用目的：${hit.purpose}</div>
    <div>结束时间：${toHH(hit.endHour)}</div>
    ${canCancel ? `<div class="tooltip-actions"><button class="tooltip-btn" data-cancel-booking-id="${hit.id}" data-cancel-hour="${hour}" type="button">${cancelBtnText}</button></div>` : ""}
  `;
}

function attachTooltip(targetEl, hit, hour) {
  targetEl.addEventListener("mouseenter", (e) => {
    clearHideTooltipTimer();
    setTooltipContent(hit, hour);
    tooltipEl.classList.remove("hidden");
    placeTooltip(e.clientX, e.clientY);
  });
  targetEl.addEventListener("mousemove", (e) => {
    placeTooltip(e.clientX, e.clientY);
  });
  targetEl.addEventListener("mouseleave", () => {
    hideTooltip(120);
  });
}

async function cancelSingleHourBooking(bookingId, cancelHour) {
  const target = state.bookings.find((b) => b.id === bookingId);
  if (!target) return;
  const canCancel = target.userName === currentUser || isAdminUser();
  if (!canCancel) return;
  if (!(cancelHour >= target.startHour && cancelHour < target.endHour)) return;

  const inserts = [];
  if (target.startHour < cancelHour) {
    inserts.push({
      id: uid("booking"),
      machine_id: target.machineId,
      date: target.date,
      start_hour: target.startHour,
      end_hour: cancelHour,
      user_name: target.userName,
      purpose: target.purpose
    });
  }
  if (cancelHour + 1 < target.endHour) {
    inserts.push({
      id: uid("booking"),
      machine_id: target.machineId,
      date: target.date,
      start_hour: cancelHour + 1,
      end_hour: target.endHour,
      user_name: target.userName,
      purpose: target.purpose
    });
  }

  const { error: delErr } = await sb.from("bookings").delete().eq("id", bookingId);
  if (delErr) {
    bookingHint.textContent = `取消失败：${delErr.message}`;
    return;
  }
  if (inserts.length > 0) {
    const { error: insErr } = await sb.from("bookings").insert(inserts);
    if (insErr) {
      bookingHint.textContent = `取消后重建区间失败：${insErr.message}`;
      return;
    }
  }
  await refreshStateFromCloud();
  hideTooltip();
  bookingHint.textContent = `已取消 ${toHH(cancelHour)}-${toHH(cancelHour + 1)} 的占用。`;
  renderTimeline();
}

machineForm.addEventListener("submit", async (e) => {
  e.preventDefault();
  const fd = new FormData(machineForm);
  const name = String(fd.get("name") || "").trim();
  const model = String(fd.get("model") || "").trim();
  const location = String(fd.get("location") || "").trim();
  const remark = String(fd.get("remark") || "").trim();

  if (!name || !model || !location) return;

  const { error } = await sb.from("machines").insert({
    id: uid("machine"),
    name,
    model,
    location,
    remark
  });
  if (error) {
    bookingHint.textContent = `新增机器失败：${error.message}`;
    return;
  }
  await refreshStateFromCloud();
  machineForm.reset();
  renderMachines();
  renderTimeline();
});

bookingForm.addEventListener("submit", async (e) => {
  e.preventDefault();
  bookingHint.textContent = "";

  if (!currentUser) {
    bookingHint.textContent = "请先输入用户名。";
    setUserModal(true);
    return;
  }

  const machineId = bookingMachineSelect.value;
  if (!machineId) {
    bookingHint.textContent = "请先录入并选择机器。";
    return;
  }

  const fd = new FormData(bookingForm);
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

  const { error } = await sb.from("bookings").insert({
    id: uid("booking"),
    machine_id: machineId,
    date,
    start_hour: startHour,
    end_hour: endHour,
    user_name: currentUser,
    purpose
  });
  if (error) {
    bookingHint.textContent = `提交失败：${error.message}`;
    return;
  }
  await refreshStateFromCloud();
  bookingForm.reset();
  startHourSelect.value = "9";
  endHourSelect.value = "11";
  bookingHint.textContent = `提交成功：${date} ${toHH(startHour)}-${toHH(endHour)} 已占用。`;
  renderTimeline();
});

userForm.addEventListener("submit", (e) => {
  e.preventDefault();
  const fd = new FormData(userForm);
  const name = String(fd.get("userName") || "").trim();
  if (!name) return;
  saveCurrentUser(name);
  setUserModal(false);
});

switchUserBtn.addEventListener("click", () => {
  setUserModal(true);
});

machineSelect.addEventListener("change", renderTimeline);
viewDate.addEventListener("change", renderTimeline);

openMachineModalBtn.addEventListener("click", () => setMachineModal(true));
closeMachineModalBtn.addEventListener("click", () => setMachineModal(false));
machineModal.addEventListener("click", (e) => {
  if (e.target === machineModal) {
    setMachineModal(false);
  }
});

function boot() {
  initHourOptions();
  viewDate.value = todayISO();
  setMachineModal(false);
  currentUserLabel.textContent = `当前用户：${currentUser || "-"}`;
  ensureUserReady();
}

(async function start() {
  boot();
  if (!initSupabase()) return;
  try {
    await refreshStateFromCloud();
    renderMachines();
    renderTimeline();
  } catch (err) {
    backendStatusLabel.textContent = "后端状态：连接失败";
    bookingHint.textContent = err.message || "云端数据加载失败";
  }
})();
