/* ================= GLOBAL ================= */
let acfState = null;
let pacfState = null;
let globalD = 0; // ← DIISI DARI ADF/KPSS (0 / 1)

/* =========================================================
   ACF & PACF PLOT (FINAL – FIXED)
   - ACF  : CI per-lag (statsmodels style)
   - PACF : CI flat
========================================================= */

let acfChart = null;
let pacfChart = null;
let valuedef__ = null;

/* =========================
   ENTRY POINT
========================= */
function runACFPACF(series, valuedef) {
    valuedef__ = valuedef

    if (!Array.isArray(series) || series.length < 10) return;

    acfState = null;
    pacfState = null;

    loadACF(series);
    loadPACF(series);
}

/* =========================
   LOAD ACF (AJAX)
========================= */
function loadACF(series) {
    $.ajax({
        url: url_api+"/acf-plot",
        method: "POST",
        contentType: "application/json",
        data: JSON.stringify({ data: series }),

        success: (res) => {
            if (res.error) return;

            acfState = res;
            drawACFChart(res.acf, res.ci_lower, res.ci_upper);
            tryAnalyzeModel();
        }
    });
}

/* =========================
   LOAD PACF (AJAX)
========================= */
function loadPACF(series) {
    $.ajax({
        url: url_api+"/pacf-plot",
        method: "POST",
        contentType: "application/json",
        data: JSON.stringify({ data: series }),

        success: (res) => {
            if (res.error) return;

            pacfState = res;
            drawPACFChart(res.pacf, res.confidence_interval);
            tryAnalyzeModel();
        }
    });
}

/* =========================
   DRAW ACF (CI PER-LAG)
========================= */
function drawACFChart(values, ciLower, ciUpper) {

    if (acfChart) {
        acfChart.destroy();
        acfChart = null;
    }

    const ctx = document.getElementById("acfChart").getContext("2d");
    const labels = values.map((_, i) => i);

    acfChart = new Chart(ctx, {
        data: {
            labels,
            datasets: [

                {
                    type: "line",
                    data: ciUpper,
                    borderColor: "rgba(0,0,0,0)",
                    backgroundColor: "rgba(30,136,229,0.20)",
                    fill: "+1",
                    pointRadius: 0
                },
                {
                    type: "line",
                    data: ciLower,
                    borderColor: "rgba(0,0,0,0)",
                    backgroundColor: "rgba(30,136,229,0.20)",
                    fill: false,
                    pointRadius: 0
                },
                {
                    type: "bar",
                    data: values,
                    backgroundColor: values.map((v, i) =>
                        (v > ciUpper[i] || v < ciLower[i]) ? "#55a3e8ff" : "#a1dbfbff"
                    ),
                    barThickness: 14
                },
                {
                    type: "line",
                    data: labels.map(() => 0),
                    borderColor: "#000",
                    borderWidth: 1,
                    pointRadius: 0
                }
            ]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            plugins: {
                legend: { display: false },
                title: {
                    display: true,
                    text: "Autocorrelation Function (ACF)"
                }
            },
            scales: {
                y: { min: -1, max: 1 },
                x: { title: { display: true, text: "Lag" } }
            }
        }
    });
}

/* =========================
   DRAW PACF (CI FLAT)
========================= */
function drawPACFChart(values, conf) {

    if (pacfChart) {
        pacfChart.destroy();
        pacfChart = null;
    }

    const ctx = document.getElementById("pacfChart").getContext("2d");
    const labels = values.map((_, i) => i);

    pacfChart = new Chart(ctx, {
        type: "bar",
        data: {
            labels,
            datasets: [
                {
                    data: values,
                    backgroundColor: values.map(v =>
                        Math.abs(v) > conf.upper ? "#55a3e8ff" : "#a1dbfbff"
                    ),
                    barThickness: 18
                },
                {
                    type: "line",
                    data: labels.map(() => conf.upper),
                    borderColor: "#595656ff",
                    borderDash: [5, 5],
                    pointRadius: 0
                },
                {
                    type: "line",
                    data: labels.map(() => conf.lower),
                    borderColor: "#595656ff",
                    borderDash: [5, 5],
                    pointRadius: 0
                }
            ]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            plugins: {
                legend: { display: false },
                title: {
                    display: true,
                    text: "Partial Autocorrelation Function (PACF)"
                }
            },
            scales: {
                y: {
                    min: -1,
                    max: 1,
                    title: { display: true, text: "Correlation" }
                },
                x: { title: { display: true, text: "Lag" } }
            }
        }
    });
}

/* =========================================================
   AUTO MODEL DETECTION (SARIMA)
========================================================= */

function isSignificant(val, lower, upper) {
    return val < lower || val > upper;
}

function detectQ(acf, ciLower, ciUpper) {
    for (let i = 1; i <= 3; i++) {
        if (acf[i] < ciLower[i] || acf[i] > ciUpper[i]) return 2;
    }
    return 0;
}

function detectP(pacf, conf) {
    for (let i = 1; i <= 3; i++) {
        if (Math.abs(pacf[i]) > conf.upper) return 2;
    }
    return 0;
}

function detectQSeasonal(acf, ciLower, ciUpper) {
    let Q = 0;
    if (acf[12] < ciLower[12] || acf[12] > ciUpper[12]) Q++;
    if (acf[24] < ciLower[24] || acf[24] > ciUpper[24]) Q++;
    return Q;
}

function detectD(acf, ciLower, ciUpper) {
    return isSignificant(acf[12], ciLower[12], ciUpper[12]) ? 1 : 0;
}

function detectPSeasonal(pacf, conf) {
    let P = 0;
    if (Math.abs(pacf[12]) > conf.upper) P++;
    if (Math.abs(pacf[24]) > conf.upper) P++;
    return P;
}

/* =========================
   MAIN ORCHESTRATOR
========================= */
function tryAnalyzeModel() {

    if (!acfState || !pacfState) return;

    const p = detectP(pacfState.pacf, pacfState.confidence_interval);
    const q = detectQ(acfState.acf, acfState.ci_lower, acfState.ci_upper);
    const P = detectPSeasonal(pacfState.pacf, pacfState.confidence_interval);
    const D = detectD(acfState.acf, acfState.ci_lower, acfState.ci_upper);
    const Q = detectQSeasonal(acfState.acf, acfState.ci_lower, acfState.ci_upper);

    renderModelCard({
        p,
        d: globalD,
        q,
        P,
        D,
        Q
    });
}

function renderModelCard({ p, d, q, P, D, Q }) {

    $("#modelCard").html(`


    <div class="row m-3">
        <div class="col-3">
            <div class="card shadow-sm">
                <div class="card-body">
                    <h6>Parameter</h6>
                    <table class="table table-sm">
                    <tr>
                        <td>p (AR) MAX </td>
                        <td>${p}</td>
                    </tr>
                    <tr>
                        <td>d (Differencing)</td>
                        <td>${valuedef__}</td>
                    </tr>
                    <tr>
                        <td>q (MA) MAX </td>
                        <td>${q}</td>
                    </tr>
                    


                    </table>                    
                </div>
            </div>
        </div>
    </div>


    
  `);
}

//<tr><td>MAX P (Seasonal AR)</td><td>${P}</td></tr>
//                    <tr><td>D (Seasonal Diff)</td><td>${D}</td></tr>
//                    <tr><td>MAX Q (Seasonal MA)</td><td>${Q}</td></tr>                    