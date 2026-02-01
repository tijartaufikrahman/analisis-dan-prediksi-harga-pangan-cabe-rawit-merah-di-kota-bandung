/* =========================================================
   SARIMA FORECAST - GLOBAL AGGREGATED DATA (FINAL FIX)
   ========================================================= */

/* ================= BULAN MAP ================= */
const MONTH_MAP = {
    "Januari": { short: "Jan", idx: 0 },
    "Februari": { short: "Feb", idx: 1 },
    "Maret": { short: "Mar", idx: 2 },
    "April": { short: "Apr", idx: 3 },
    "Mei": { short: "Mei", idx: 4 },
    "Juni": { short: "Jun", idx: 5 },
    "Juli": { short: "Jul", idx: 6 },
    "Agustus": { short: "Agu", idx: 7 },
    "September": { short: "Sep", idx: 8 },
    "Oktober": { short: "Okt", idx: 9 },
    "November": { short: "Nov", idx: 10 },
    "Desember": { short: "Des", idx: 11 }
};

/* ================= MAIN ================= */
async function runSarimaForecastFromGlobal() {

    if (!GLOBAL_AGGREGATED_DATA || !GLOBAL_AGGREGATED_DATA.length) {
        alert("Data agregasi belum tersedia");
        return;
    }

    showSarimaSpinner();

    try {
        /* ================= DATA ================= */
        const actualValues = GLOBAL_AGGREGATED_DATA.map(d => d.avg);
        const actualPeriods = GLOBAL_AGGREGATED_DATA.map(d => d.periode);
        const nForecast = +$('#n_forecast').val();

        const payload = {
            series: actualValues,

            p: +$('#best_p').val(),
            d: +$('#best_d').val(),
            q: +$('#best_q').val(),

            P: +$('#best_P').val(),
            D: +$('#best_D').val(),
            Q: +$('#best_Q').val(),
            s: +$('#best_s').val(),

            n_forecast: nForecast
        };

        /* ================= REQUEST ================= */
        const res = await fetch(url_api+"/forecast/sarima", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify(payload)
        });

        const json = await res.json();

        $('#forecastInfo').text(
            JSON.stringify(json.diagnostic, null, 2)
        );

        /* ================= LABEL ================= */
        const allPeriods = buildMonthYearLabels(
            actualPeriods,
            nForecast
        );

        /* ================= RENDER ================= */
        renderForecastChart(
            actualValues,
            json.forecast,
            allPeriods
        );

        renderForecastTable(
            allPeriods.slice(-nForecast),
            json.forecast
        );

        //renderResidualChart(json.residual);

    } catch (e) {
        console.error(e);
        $('#forecastTableContainer').html(`
            <div class="alert alert-danger">
                Gagal memproses forecast SARIMA
            </div>
        `);
    } finally {
        hideSarimaSpinner();
    }
}

/* =========================================================
   BUILD LABEL: Jan 2023 → Feb 2023 → dst
   ========================================================= */
function buildMonthYearLabels(actualPeriods, nForecast) {

    let labels = actualPeriods.map(p => {
        const [bulan, tahun] = p.split(" ");
        return `${MONTH_MAP[bulan].short} ${tahun}`;
    });

    const [lastMonth, lastYear] =
        actualPeriods[actualPeriods.length - 1].split(" ");

    let monthIdx = MONTH_MAP[lastMonth].idx;
    let year = parseInt(lastYear);

    for (let i = 0; i < nForecast; i++) {
        monthIdx++;
        if (monthIdx > 11) {
            monthIdx = 0;
            year++;
        }

        const monthShort = Object.values(MONTH_MAP)
            .find(m => m.idx === monthIdx).short;

        labels.push(`${monthShort} ${year}`);
    }

    return labels;
}

/* =========================================================
   CHART: ACTUAL + FORECAST (SMOOTH)
   ========================================================= */
function renderForecastChart(actual, forecast, labels) {

    const ctx = document.getElementById("forecastChart");
    if (!ctx) return;

    if (ctx._chart) ctx._chart.destroy();

    ctx._chart = new Chart(ctx, {
        type: "line",
        data: {
            labels,
            datasets: [
                {
                    label: "Aktual",
                    data: actual,
                    borderWidth: 2,
                    tension: 0.4,
                    pointRadius: 3
                },
                {
                    label: "Forecast",
                    data: Array(actual.length).fill(null).concat(forecast),
                    borderDash: [6, 6],
                    borderWidth: 2,
                    tension: 0.4,
                    pointRadius: 3
                }
            ]
        },
        options: {
            responsive: true,
            interaction: {
                mode: "index",
                intersect: false
            },
            plugins: {
                legend: { position: "top" }
            }
        }
    });
}

/* =========================================================
   TABLE FORECAST (ID KHUSUS)
   ========================================================= */
function renderForecastTable(periods, forecast) {

    let html = `
        <div class="card mt-4">
            <div class="card-header fw-bold">
                Tabel Hasil Prediksi Harga
            </div>
            <div class="card-body p-0">
                <table class="table table-bordered table-sm mb-0">
                    <thead class="table-light">
                        <tr>
                            <th style="width:60px">No</th>
                            <th>Periode</th>
                            <th class="text-end">Harga Prediksi</th>
                        </tr>
                    </thead>
                    <tbody>
    `;

    forecast.forEach((v, i) => {
        html += `
            <tr>
                <td class="text-center">${i + 1}</td>
                <td>${periods[i]}</td>
                <td class="text-end">
                    ${Math.round(v).toLocaleString('id-ID')}
                </td>
            </tr>
        `;
    });

    html += `
                    </tbody>
                </table>
            </div>
        </div>
    `;

    $('#forecastTableContainer').html(html);
}

/* =========================================================
   RESIDUAL CHART
   ========================================================= */
function renderResidualChart(resid) {

    const ctx = document.getElementById("residualChart");
    if (!ctx) return;

    if (ctx._chart) ctx._chart.destroy();

    ctx._chart = new Chart(ctx, {
        type: "line",
        data: {
            labels: resid.map((_, i) => i + 1),
            datasets: [{
                label: "Residual",
                data: resid,
                tension: 0.3,
                pointRadius: 2
            }]
        }
    });
}

/* ================= SPINNER ================= */
function showSarimaSpinner() {
    $('#sarimaSpinner').removeClass('d-none');
}

function hideSarimaSpinner() {
    $('#sarimaSpinner').addClass('d-none');
}
