/* ================= GLOBAL ================= */
let stationarySeries = null;
let diffChart = null;
let valuedef = 0;

/* ================= ENTRY ================= */
function runStationarityTest(aggregatedData) {
    

    if (!Array.isArray(aggregatedData)) return;

    const series = aggregatedData
        .map(d => Number(d.avg))
        .filter(v => !isNaN(v));

    if (series.length < 10) {
        $("#adfLevel, #kpssLevel").html(`
          <div class="alert alert-warning">
            Data hasil agregasi kurang dari 10 observasi
          </div>
        `);
        return;
    }

    stationarySeries = null;
    $("#adfDiff, #kpssDiff").html("");

    runADF(series, "level");
}

/* ================= ADF ================= */
function runADF(series, stage) {

    const target = stage === "level" ? "#adfLevel" : "#adfDiff";

    $(target).html(`<div class="alert alert-info">Menghitung ADF...</div>`);

    $.ajax({
        url: url_api+"/adf-test",
        method: "POST",
        contentType: "application/json",
        data: JSON.stringify({ data: series }),
        success: function (res) {
            renderADF(res, stage);
            if (!res.error) {
                runKPSS(series, stage, res.stationary);
            }
        }
    });
}

/* ================= KPSS ================= */
function runKPSS(series, stage, adfStationary) {

    const target = stage === "level" ? "#kpssLevel" : "#kpssDiff";

    $(target).html(`<div class="alert alert-info">Menghitung KPSS...</div>`);

    $.ajax({
        url: url_api+"/kpss-test",
        method: "POST",
        contentType: "application/json",
        data: JSON.stringify({ data: series }),
        success: function (res) {

            renderKPSS(res, stage);
            if (res.error) return;

            /* ========= LEVEL ========= */
            if (stage === "level") {

                // ✅ LOLOS ADF & KPSS → LANGSUNG ACF & PACF
                if (adfStationary && res.stationary) {

                    hideDiffChart();       // ⬅️ CUMA INI

                    stationarySeries = series;

                    // 🔥 PANGGIL FILE ACF & PACF
                    runACFPACF(series, 0);

                    return; // STOP
                }

                // ❌ TIDAK LOLOS → DIFFERENCING
                showDiffChart();       
                runDifferencing(series);
                //runACFPACF(series);
            }

            /* ========= DIFF ========= */
            if (stage === "diff") {

                // ✅ LOLOS SETELAH DIFFERENCING → ACF & PACF
                if (adfStationary && res.stationary) {

                    stationarySeries = series;

                    // 🔥 PANGGIL FILE ACF & PACF
                    runACFPACF(series, 1);
                }
            }
        }
    });
}

/* ================= DIFFERENCING ================= */
function runDifferencing(series) {

    $.ajax({
        url: url_api+"/difference",
        method: "POST",
        contentType: "application/json",
        data: JSON.stringify({ data: series }),
        success: function (res) {
            valuedef = 1;

            if (res.error) {
                $("#adfDiff").html(`<div class="alert alert-danger">${res.error}</div>`);
                return;
            }

            drawDiffChart(res.data);
            runADF(res.data, "diff");
        }
    });
}

/* ================= RENDER ADF ================= */
function renderADF(res, stage) {

    const target = stage === "level" ? "#adfLevel" : "#adfDiff";

    if (res.error) {
        $(target).html(`<div class="alert alert-danger">${res.error}</div>`);
        return;
    }

    renderTestCard(
        target,
        "ADF",
        stage,
        res.adf_statistic,
        res.p_value,
        res.stationary
    );
}

/* ================= RENDER KPSS ================= */
function renderKPSS(res, stage) {

    const target = stage === "level" ? "#kpssLevel" : "#kpssDiff";

    if (res.error) {
        $(target).html(`<div class="alert alert-danger">${res.error}</div>`);
        return;
    }

    renderTestCard(
        target,
        "KPSS",
        stage,
        res.kpss_statistic,
        res.p_value,
        res.stationary
    );
}

/* ================= CARD ================= */
function renderTestCard(target, title, stage, stat, pval, stationary) {

    const cls = stationary ? "success" : "danger";
    const txt = stationary ? "STASIONER" : "TIDAK STASIONER";
    const suffix = stage === "diff" ? " (Setelah Differencing)" : "";

    $(target).html(`
        <div class="card shadow-sm border-0 ">
          <div class="card-body">
            <h6>${title}${suffix}</h6>
            <table class="table table-sm mb-0">
              <tr><td>Statistic</td><td>${stat.toFixed(4)}</td></tr>
              <tr><td>p-value</td><td>${pval.toFixed(4)}</td></tr>
              <tr>
                <td>Keputusan</td>
                <td class="fw-bold text-${cls}">${txt}</td>
              </tr>
            </table>
          </div>
        </div>
    `);
}

/* ================= GRAFIK DIFFERENCING ================= */
function drawDiffChart(diffSeries) {

    const $canvas = $("#diffChart");
    if (!$canvas.length) return;

    const ctx = $canvas[0].getContext("2d");
    if (diffChart) diffChart.destroy();

    diffChart = new Chart(ctx, {
        type: "line",
        data: {
            labels: diffSeries.map((_, i) => `t-${i + 1}`),
            datasets: [{
                label: "Differenced Series (d = 1)",
                data: diffSeries,
                borderWidth: 2,
                tension: 0.3,
                fill: false
            }]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            scales: {
                x: { grid: { display: false } },
                y: { title: { display: true, text: "Δ Nilai" } }
            }
        }
    });
}


function showDiffChart() {
    $("#diffInfo").html("");
    $("#diffChartContainer").show();
}

function hideDiffChart() {
    $("#diffChartContainer").hide();
    $("#diffInfo").html(`
        <span class="text-secodary mb-0">
            Data sudah stasioner, differencing tidak diperlukan
        </span>
    `);
}
