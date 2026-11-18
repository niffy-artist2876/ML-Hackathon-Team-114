const apiInput = document.querySelector("#api-url");
const tripSelect = document.querySelector("#trip-select");
const button = document.querySelector("#load-button");
const status = document.querySelector("#status");
const result = document.querySelector("#result");
const summary = document.querySelector("#summary");
const route = document.querySelector("#route");

async function loadTrips() {
  const response = await fetch(`${apiInput.value.replace(/\/$/, "")}/trips`);
  if (!response.ok) throw new Error(await response.text());
  const data = await response.json();
  tripSelect.replaceChildren(...data.trip_ids.map((id) => new Option(id, id)));
  status.textContent = `Loaded ${data.trip_ids.length} trips.`;
}

async function predict() {
  const api = apiInput.value.replace(/\/$/, "");
  const trip = tripSelect.value;
  status.textContent = "Requesting prediction…";
  const response = await fetch(`${api}/predict/${trip}`);
  if (!response.ok) throw new Error(await response.text());
  const data = await response.json();
  const metrics = data.metrics;
  summary.textContent = `${data.trip_id}: ${metrics.points} observations; HMM continuity ${metrics.hmm_route_continuity.toFixed(3)}; ${metrics.latency_ms} ms.`;
  route.textContent = `Predicted arc IDs: ${data.predicted_arcs.join(", ")}`;
  result.hidden = false;
  status.textContent = "Prediction received.";
}

button.addEventListener("click", () => predict().catch((error) => { status.textContent = `Error: ${error.message}`; }));
loadTrips().catch((error) => { status.textContent = `Error: ${error.message}`; });
