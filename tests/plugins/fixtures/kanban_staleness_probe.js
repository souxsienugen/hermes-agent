// Behavioral probe for the dashboard's running-card staleness clock: extracts
// the real staleness helpers from the shipped dashboard bundle (no build step —
// the bundle IS the source) and drives them with the card payloads the backend
// actually emits.
//
// The bug this pins: the board coloured a RUNNING card from
// `age.started_age_seconds`, which the backend derives from `tasks.started_at` —
// the FIRST-EVER start. That column is never reset on re-dispatch (the claim
// path writes COALESCE(started_at, ?)), so a re-dispatched card kept ageing
// forever and stayed RED/AMBER while the live worker was healthy and
// heartbeating. The correct clock is the CURRENT run's start, which the
// backend already exposes as `current_run_started_at`.
//
// The engine's own rule (hermes_cli/kanban_db_dispatch.detect_stale_running)
// is "run age over the threshold AND no heartbeat progress" — a card whose
// worker is still reporting is not a stall. This probe pins that same
// two-condition rule so the board and the dispatcher agree.
//
// Run via: node kanban_staleness_probe.js <path-to-bundle>
const fs = require("fs");

const bundlePath = process.argv[2];
const src = fs.readFileSync(bundlePath, "utf8");

// Pull the real functions out of the bundle verbatim (brace-matched, not
// regex-matched) and evaluate them in isolation. The three helpers are
// optional: a bundle that predates the fix does not have them yet, so fall
// back to their pre-fix semantics (first-ever age, no heartbeat gate) and
// let the behavioural checks below decide pass/fail.
function extractFunction(name, required) {
  // Anchor the name: a bare prefix match would happily extract a
  // same-prefix sibling (`stalenessClassOf`) defined earlier in the bundle.
  const match = new RegExp("\\bfunction\\s+" + name + "\\s*\\(").exec(src);
  if (!match) {
    if (required) { console.error(name + " not found in bundle"); process.exit(1); }
    return null;
  }
  const start = match.index;
  const bodyStart = src.indexOf("{", start);
  let depth = 0, end = bodyStart;
  for (; end < src.length; end++) {
    if (src[end] === "{") depth++;
    else if (src[end] === "}") { depth--; if (depth === 0) break; }
  }
  return src.slice(start, end + 1);
}

// STALENESS is a `const` object literal, not a function — grab it by
// bracket-matching the literal. The heartbeat gap is optional: on a bundle
// that predates the fix it does not exist yet, so fall back to the engine's
// value and let the behavioural checks below decide pass/fail.
//
// A const's value is not always an object literal (the heartbeat gap is a
// plain number), so branch on the value's first character: `{` means an
// object literal (match to its closing brace), anything else means a simple
// expression (match to the terminating `;`). Bracket-matching a simple
// expression would run past it into unrelated later code.
function extractConst(name, required) {
  // Anchored for the same reason as extractFunction: a bare prefix match
  // could pick up a same-prefix sibling constant.
  const match = new RegExp("\\bconst\\s+" + name + "\\s*=").exec(src);
  if (!match) {
    if (required) { console.error(name + " not found in bundle"); process.exit(1); }
    return null;
  }
  const start = match.index;
  const valueStart = src.indexOf("=", start) + 1;
  const rest = src.slice(valueStart);
  const firstChar = (rest.match(/[^\s]/) || [""])[0];
  if (firstChar === "{") {
    const objStart = valueStart + rest.indexOf("{");
    let depth = 0, end = objStart;
    for (; end < src.length; end++) {
      if (src[end] === "{") depth++;
      else if (src[end] === "}") { depth--; if (depth === 0) break; }
    }
    return src.slice(start, end + 2);
  }
  let depth = 0, end = valueStart;
  for (; end < src.length; end++) {
    const c = src[end];
    if (c === "(" || c === "[" || c === "{") depth++;
    else if (c === ")" || c === "]" || c === "}") depth--;
    else if (c === ";" && depth === 0) break;
  }
  return src.slice(start, end + 1);
}

const heartbeatGapSrc = extractConst("STALE_HEARTBEAT_GAP_SECONDS", false);

// Evaluate a `const X = <expr>;` snippet and return X's value (null when the
// snippet cannot be evaluated). Used to report the constant the bundle
// DECLARES, so the Python suite can compare it against the engine's own.
function evalConst(srcText, name) {
  try {
    return new Function(srcText + "\nreturn " + name + ";")();
  } catch (e) {
    return null;
  }
}

// `--print-constants` reports the bundle's declared heartbeat gap as JSON and
// exits. The Python suite uses it to pin the bundle's constant against the
// engine's `_STALE_HEARTBEAT_GAP_SECONDS` — the two live in different files
// and nothing else ties them together.
if (process.argv[3] === "--print-constants") {
  const declared = heartbeatGapSrc
    ? evalConst(heartbeatGapSrc, "STALE_HEARTBEAT_GAP_SECONDS")
    : null;
  process.stdout.write(JSON.stringify({ stale_heartbeat_gap_seconds: declared }) + "\n");
  process.exit(0);
}

// Pre-fix fallbacks for a bundle that predates the fix: the age came from the
// first-ever start and no heartbeat gate existed. They are only reached when
// the bundle lacks the helpers, i.e. exactly when the bug is still present.
const preFixPayloadNow = "function payloadNowSeconds(task) { return null; }";
const preFixCurrentRunAge = "function currentRunAgeSeconds(task, now) { return task.age.started_age_seconds; }";
const preFixWorkerReporting = "function workerReporting(task, now) { return false; }";

const preamble = [
  extractConst("STALENESS", true),
  heartbeatGapSrc || "const STALE_HEARTBEAT_GAP_SECONDS = 60 * 60;",
  extractFunction("payloadNowSeconds", false) || preFixPayloadNow,
  extractFunction("currentRunAgeSeconds", false) || preFixCurrentRunAge,
  extractFunction("workerReporting", false) || preFixWorkerReporting,
  extractFunction("stalenessClass", true),
].join("\n\n");

const factory = new Function(preamble + "\nreturn { STALENESS, STALE_HEARTBEAT_GAP_SECONDS, stalenessClass };");
const { STALENESS, STALE_HEARTBEAT_GAP_SECONDS, stalenessClass } = factory();

const NOW = 1000000;
const RED = "hermes-kanban-card--stale-red";
const AMBER = "hermes-kanban-card--stale-amber";

let failures = 0;
function check(label, actual, expected) {
  if (actual === expected) return;
  failures++;
  console.error("FAIL: " + label + " -> expected " + JSON.stringify(expected) + ", got " + JSON.stringify(actual));
}

// --- The heartbeat gap is a number shared with the engine -----------------
// The stale-heartbeat fixture below is built FROM this constant, so without
// this assertion a changed bundle value would silently rescale every fixture
// and the probe would still PASS — the probe could not detect drift in the
// very constant it derives its own expectations from.
//
// The engine's copy of the same number lives in
// hermes_cli/kanban_db_dispatch.py (_STALE_HEARTBEAT_GAP_SECONDS). The Python
// suite pins the bundle's value against the engine's, so this literal and the
// engine's value must agree; changing either one alone fails a test.
check("heartbeat gap constant matches the engine's 3600s", STALE_HEARTBEAT_GAP_SECONDS, 3600);

// Build a card shaped exactly like the backend's _task_dict output.
// startedAt is the FIRST-EVER start; runStart is the CURRENT run's start
// (null when the card has no live run).
function runningCard({ startedAt, runStart, lastHeartbeat, now }) {
  const serverNow = now === undefined ? NOW : now;
  return {
    id: "t_probe",
    status: "running",
    started_at: startedAt,
    created_at: startedAt - 600,
    current_run_started_at: runStart,
    last_heartbeat_at: lastHeartbeat === undefined ? null : lastHeartbeat,
    age: {
      created_age_seconds: serverNow - (startedAt - 600),
      started_age_seconds: serverNow - startedAt,
      time_to_complete_seconds: null,
    },
  };
}

// --- The reported bug -----------------------------------------------------
// A card re-dispatched after a long history: first-ever start is 2560 minutes
// ago, the CURRENT run started 111 minutes ago, and the worker heartbeats
// every 60s. It must NOT be red — the current run is the live attempt.
const healthy = runningCard({
  startedAt: NOW - 2560 * 60,
  runStart: NOW - 111 * 60,
  lastHeartbeat: NOW - 60,
});
check("re-dispatched healthy card is not red", stalenessClass(healthy), AMBER);
check("re-dispatched healthy card is not red (red class)",
  stalenessClass(healthy) === RED, false);

// --- AC2: a genuinely stalled card still goes red -------------------------
// Current run over 60 minutes with NO heartbeat: that IS a stall.
const stalled = runningCard({
  startedAt: NOW - 2560 * 60,
  runStart: NOW - 61 * 60,
  lastHeartbeat: null,
});
check("stalled card (no heartbeat) is red", stalenessClass(stalled), RED);

// A current run over 60 minutes whose last heartbeat is itself older than the
// gap is stalled too.
const stalledOld = runningCard({
  startedAt: NOW - 2560 * 60,
  runStart: NOW - 61 * 60,
  lastHeartbeat: NOW - (STALE_HEARTBEAT_GAP_SECONDS + 30),
});
check("stalled card (stale heartbeat) is red", stalenessClass(stalledOld), RED);

// A current run over 60 minutes that heartbeats right now is alive: amber.
const aliveLong = runningCard({
  startedAt: NOW - 2560 * 60,
  runStart: NOW - 61 * 60,
  lastHeartbeat: NOW - 5,
});
check("long-but-alive card is amber, not red", stalenessClass(aliveLong), AMBER);

// --- Thresholds unchanged (AC3) -------------------------------------------
check("running red threshold constant", STALENESS.running.red, 3600);
check("running amber threshold constant", STALENESS.running.amber, 600);
check("todo amber threshold constant", STALENESS.todo.amber, 604800);
check("todo red threshold constant", STALENESS.todo.red, 2592000);
check("ready amber threshold constant", STALENESS.ready.amber, 3600);
check("ready red threshold constant", STALENESS.ready.red, 86400);
check("blocked amber threshold constant", STALENESS.blocked.amber, 3600);
check("blocked red threshold constant", STALENESS.blocked.red, 86400);

// A healthy run of exactly the red threshold is amber (heartbeat is live).
const atRed = runningCard({
  startedAt: NOW - 2560 * 60,
  runStart: NOW - 3600,
  lastHeartbeat: NOW - 5,
});
check("healthy run at exactly the red threshold is amber", stalenessClass(atRed), AMBER);

// A healthy run past the red threshold is amber (heartbeat live).
const pastRed = runningCard({
  startedAt: NOW - 2560 * 60,
  runStart: NOW - 7200,
  lastHeartbeat: NOW - 5,
});
check("healthy run past the red threshold is amber", stalenessClass(pastRed), AMBER);

// A healthy run between amber and red is amber.
const between = runningCard({
  startedAt: NOW - 2560 * 60,
  runStart: NOW - 1800,
  lastHeartbeat: NOW - 5,
});
check("healthy run between amber and red is amber", stalenessClass(between), AMBER);

// A healthy run under the amber threshold is not stale at all.
const fresh = runningCard({
  startedAt: NOW - 2560 * 60,
  runStart: NOW - 120,
  lastHeartbeat: NOW - 5,
});
check("healthy fresh run is not stale", stalenessClass(fresh), "");

// --- Fallback: no current run uses the first-ever age ---------------------
// A running card with no live run (current_run_started_at null) falls back to
// started_age_seconds, exactly as before the fix.
const noRun = runningCard({
  startedAt: NOW - 7200,
  runStart: null,
  lastHeartbeat: NOW - 5,
});
check("running card with no current run falls back to first-ever age",
  stalenessClass(noRun), AMBER);

const noRunStalled = runningCard({
  startedAt: NOW - 7200,
  runStart: null,
  lastHeartbeat: null,
});
check("running card with no current run and no heartbeat is red",
  stalenessClass(noRunStalled), RED);

// A card whose age metrics carry no usable clock cannot be aged at all, so it
// is not painted stale — the run clock must not be faked from a missing
// timestamp.
const noClock = runningCard({
  startedAt: NOW - 7200,
  runStart: NOW - 7200,
  lastHeartbeat: null,
});
noClock.age = { created_age_seconds: null, started_age_seconds: null, time_to_complete_seconds: null };
noClock.started_at = null;
noClock.created_at = null;
check("card with no usable clock is not stale", stalenessClass(noClock), "");

// --- Non-running columns keep their existing behaviour ---------------------
function plainCard(status, createdAgeSeconds) {
  return {
    id: "t_probe_" + status,
    status: status,
    created_at: NOW - createdAgeSeconds,
    started_at: NOW - createdAgeSeconds,
    current_run_started_at: null,
    last_heartbeat_at: null,
    age: {
      created_age_seconds: createdAgeSeconds,
      started_age_seconds: createdAgeSeconds,
      time_to_complete_seconds: null,
    },
  };
}
check("ready card over 24h is red", stalenessClass(plainCard("ready", 25 * 3600)), RED);
check("ready card over 1h is amber", stalenessClass(plainCard("ready", 2 * 3600)), AMBER);
check("ready card under 1h is not stale", stalenessClass(plainCard("ready", 60)), "");
check("todo card over 30d is red", stalenessClass(plainCard("todo", 31 * 24 * 3600)), RED);
check("todo card over 7d is amber", stalenessClass(plainCard("todo", 8 * 24 * 3600)), AMBER);
check("blocked card over 24h is red", stalenessClass(plainCard("blocked", 25 * 3600)), RED);
check("blocked card under 1h is not stale", stalenessClass(plainCard("blocked", 60)), "");
check("done card is never stale", stalenessClass(plainCard("done", 100 * 24 * 3600)), "");
check("card with no age payload is not stale", stalenessClass({ id: "x", status: "running" }), "");

if (failures) {
  console.error(failures + " check(s) failed");
  process.exit(1);
}
process.stdout.write("PASS\n");
