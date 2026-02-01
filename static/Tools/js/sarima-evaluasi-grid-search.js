/* =========================================================
   SARIMA UI (OUT-OF-SAMPLE)
   - Source data: GLOBAL_TRAIN_DATA, GLOBAL_TEST_DATA
   - Grid parameter dari INPUT (jQuery)
   - Request via AJAX jQuery
   - Spinner ON/OFF
   ========================================================= */

async function runSarimaFromGlobalUI() {

    if (!GLOBAL_TRAIN_DATA.length || !GLOBAL_TEST_DATA.length) {
        alert("Data train / test belum tersedia");
        return;
    }

    const container = document.getElementById("sarimaResult");
    container.innerHTML = "";

    showSarimaSpinner();

    try {
        /* ================= DATA ================= */
        const train = GLOBAL_TRAIN_DATA.map(d => d.avg);
        const actual = GLOBAL_TEST_DATA.map(d => d.avg);
        const periods = GLOBAL_TEST_DATA.map(d => d.periode);

        /* ================= GRID PARAM (INPUT) ================= */
        const max_p = parseInt($('#max_p').val()) || 0;
        const max_d = parseInt($('#max_d').val()) || 0;
        const max_q = parseInt($('#max_q').val()) || 0;

        const max_P = parseInt($('#max_P').val()) || 0;
        const max_D = parseInt($('#max_D').val()) || 0;
        const max_Q = parseInt($('#max_Q').val()) || 0;

        const s = parseInt($('#s').val()) || 1;

        /* ================= REQUEST ================= */


        const data = await $.ajax({
            url: url_api+"/sarima/grid-search",
            method: "POST",
            contentType: "application/json",
            data: JSON.stringify({
                train,
                actual,
                max_p,
                max_d,
                max_q,
                max_P,
                max_D,
                max_Q,
                s
            })
        });

        if (!data.success) {
            container.innerHTML = `
              <div class="alert alert-danger">${data.message}</div>
            `;
            return;
        }

        if (data.models && data.models.length > 0) {
            const best = data.models[0]; // 🔥 sudah diurutkan → best model

            // order = [p, d, q]
            $('#best_p').val(best.order[0]);
            $('#best_d').val(best.order[1]);
            $('#best_q').val(best.order[2]);

            // seasonal = [P, D, Q, s]
            $('#best_P').val(best.seasonal[0]);
            $('#best_D').val(best.seasonal[1]);
            $('#best_Q').val(best.seasonal[2]);
            $('#best_s').val(best.seasonal[3]);
        }

        /* ================= SUMMARY TABLE ================= */
        let summary = `
          <div class="card mb-4">
            <div class="card-header fw-bold">
              Ringkasan Model SARIMA (Out-of-Sample)
            </div>
            <div class="card-body p-0">
              <table class="table table-bordered table-sm mb-0">
                <thead class="table-light">
                  <tr>
                    <th>No</th>
                    <th>Order (p,d,q)</th>
                    <th>Seasonal (P,D,Q,s)</th>
                    <th>MAPE</th>
                    <th>RMSE</th>
                    <th>AIC</th>
                    <th>Ljung-Box p</th>
                  </tr>
                </thead>
                <tbody>
        `;

        data.models.forEach((m, i) => {
            summary += `
              <tr class="${i === 0 ? 'table-success fw-bold' : ''}">
                <td>${i + 1}</td>
                <td>${m.order.join(",")}</td>
                <td>${m.seasonal.join(",")}</td>
                <td>${m.MAPE}</td>
                <td>${m.RMSE}</td>
                <td>${m.AIC}</td>
                <td>${m.white_noise}</td>
              </tr>
            `;
        });

        summary += `</tbody></table></div></div>`;
        container.innerHTML = summary;

        /* ================= DETAIL PER MODEL ================= */
        data.models.forEach((m, index) => {

            const card = document.createElement("div");
            card.className = "card mb-4";

            const rows = actual.map((a, i) => `
                <tr>
                  <td>${periods[i]}</td>
                  <td>${a.toLocaleString()}</td>
                  <td>${Math.round(m.forecast[i]).toLocaleString()}</td>
                </tr>
            `).join("");

            card.innerHTML = `
              <div class="card-header fw-bold">
                Model #${index + 1} |
                SARIMA(${m.order.join(",")}) (${m.seasonal.join(",")})
                | AIC: ${m.AIC} | MAPE: ${m.MAPE}%
              </div>

              <div class="card-body">
                <div class="row">
                  <div class="col-md-7">
                    <canvas id="chart-${index}"></canvas>
                  </div>
                  <div class="col-md-5">
                    <table class="table table-sm table-bordered">
                      <thead class="table-light">
                        <tr>
                          <th>Periode</th>
                          <th>Aktual</th>
                          <th>Prediksi</th>
                        </tr>
                      </thead>
                      <tbody>${rows}</tbody>
                    </table>
                  </div>
                </div>
              </div>
            `;

            container.appendChild(card);

            new Chart(document.getElementById(`chart-${index}`), {
                type: "line",
                data: {
                    labels: periods,
                    datasets: [
                        {
                            label: "Aktual (OOS)",
                            data: actual,
                            borderDash: [5, 5],
                            borderWidth: 2
                        },
                        {
                            label: "Prediksi",
                            data: m.forecast.slice(0, actual.length),
                            borderWidth: 2
                        }
                    ]
                },
                options: {
                    responsive: true,
                    plugins: {
                        legend: { position: "top" }
                    }
                }
            });
        });

    } catch (err) {
        console.error(err);
        container.innerHTML = `
          <div class="alert alert-danger">
            Terjadi kesalahan saat memproses SARIMA
          </div>
        `;
    } finally {
        hideSarimaSpinner();
    }
}

/* ================= SPINNER ================= */
function showSarimaSpinner() {
    $('#sarimaSpinner').removeClass('d-none');
}

function hideSarimaSpinner() {
    $('#sarimaSpinner').addClass('d-none');
}
