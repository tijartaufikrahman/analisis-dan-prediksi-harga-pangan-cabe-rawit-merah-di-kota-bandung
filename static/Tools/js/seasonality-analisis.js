/* ======================================================
   SEASONALITY CHART (jQuery Version)
   MODE: daily | weekly | monthly
   DATA: GLOBAL_FILTERED_DATA_PREDIKSI
====================================================== */

let chartSeasonPrediksi = null;

/* ================= NORMALISASI TANGGAL ================= */
function normalisasiTanggalSeason(tanggal, mode) {

    const d = new Date(tanggal);

    if (mode === 'daily') {
        return new Date(d.getFullYear(), d.getMonth(), d.getDate(), 12);
    }

    if (mode === 'weekly') {
        const day = d.getDay() || 7;
        d.setDate(d.getDate() - day + 1);
        return new Date(d.getFullYear(), d.getMonth(), d.getDate(), 12);
    }

    if (mode === 'monthly') {
        return new Date(d.getFullYear(), d.getMonth(), 1, 12);
    }

    return d;
}

/* ================= RENDER SEASONALITY ================= */
function renderSeasonChartPrediksi(data, modeWaktu) {

    const $canvas = $('#seasonChartPrediksi');
    const $info = $('#insightBoxSeasonPrediksi');

    if (!$canvas.length || !$info.length) return;

    if (chartSeasonPrediksi) {
        chartSeasonPrediksi.destroy();
        chartSeasonPrediksi = null;
    }

    const namaBulan = [
        'Januari', 'Februari', 'Maret', 'April', 'Mei', 'Juni',
        'Juli', 'Agustus', 'September', 'Oktober', 'November', 'Desember'
    ];

    /* ================= 1️⃣ AGREGASI PER PERIODE ================= */
    const bucketPeriode = {};

    data.forEach(d => {

        if (!(d.tanggal_obj instanceof Date) || isNaN(d.tanggal_obj)) return;
        if (isNaN(d.harga)) return;

        const t = normalisasiTanggalSeason(d.tanggal_obj, modeWaktu);
        const key = t.getTime();

        if (!bucketPeriode[key]) {
            bucketPeriode[key] = { sum: 0, cnt: 0, date: t };
        }

        bucketPeriode[key].sum += d.harga;
        bucketPeriode[key].cnt++;
    });

    const periodeValues = Object.values(bucketPeriode)
        .map(b => ({
            date: b.date,
            avg: b.sum / b.cnt
        }));

    if (periodeValues.length < 3) {
        $info.text('Data tidak mencukupi untuk analisis musiman.');
        return;
    }

    /* ================= 2️⃣ KELOMPOK BULAN ================= */
    const bucketBulanan = Array.from({ length: 12 }, () => ({ sum: 0, cnt: 0 }));

    periodeValues.forEach(p => {
        const m = p.date.getMonth();
        bucketBulanan[m].sum += p.avg;
        bucketBulanan[m].cnt++;
    });

    const monthlyAvg = bucketBulanan.map(b =>
        b.cnt ? b.sum / b.cnt : null
    );

    const validMonths = monthlyAvg
        .map((v, i) => ({ bulan: namaBulan[i], value: v }))
        .filter(d => d.value !== null);

    if (validMonths.length < 3) {
        $info.text('Data bulanan tidak cukup untuk analisis musiman.');
        return;
    }

    /* ================= 3️⃣ CV (COEFFICIENT OF VARIATION) ================= */
    const values = periodeValues.map(p => p.avg);
    const mean = values.reduce((a, b) => a + b, 0) / values.length;

    const variance =
        values.reduce((s, v) => s + Math.pow(v - mean, 2), 0) / values.length;

    const stdDev = Math.sqrt(variance);
    const cv = (stdDev / mean) * 100;

    const kategori = getSeasonCategory(cv);
    const textClass = getSeasonTextClass(cv);

    /* ================= 4️⃣ BULAN MAHAL & MURAH ================= */
    const indexBulanan = validMonths.map(d => ({
        ...d,
        index: d.value / mean
    }));

    const bulanMahal = indexBulanan.filter(d => d.index > 1.1);
    const bulanMurah = indexBulanan.filter(d => d.index < 0.95);

    /* ================= 5️⃣ RENDER CHART ================= */
    chartSeasonPrediksi = new Chart($canvas[0], {
        type: 'bar',
        data: {
            labels: validMonths.map(d => d.bulan),
            datasets: [{
                label: `Rata-rata Harga Bulanan (${modeWaktu})`,
                data: validMonths.map(d => Math.round(d.value)),
                backgroundColor: '#cfeefe'
            }]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            devicePixelRatio: window.devicePixelRatio || 1,
            scales: {
                y: {
                    ticks: {
                        callback: v => 'Rp ' + Number(v).toLocaleString('id-ID')
                    }
                }
            }
        }

    });

    /* ================= 6️⃣ INSIGHT ================= */
    let narasiUtama = '';
    let narasiFaktor = '';

    // =============================
    // INSIGHT MUSIMAN (FINAL FIX)
    // =============================

    // =====================
    // 1️⃣ STABIL
    // =====================
    if (cv < 10) {

        narasiUtama = `
        Terdapat fluktuasi harga antarbulan,
        namun tidak membentuk pola musiman yang konsisten.
    `;

        narasiFaktor = `
        Faktor waktu
        <b class="text-muted fw-bold">
            tidak memengaruhi harga secara signifikan
        </b>.
    `;

    }

    // =====================
    // 2️⃣ MODERAT (RINGAN)
    // =====================
    else if (cv < 20 && (bulanMahal.length + bulanMurah.length) >= 2) {

        narasiUtama = `
        Terdapat variasi harga antarbulan,
        namun perbedaannya masih relatif terbatas.
    `;

        narasiFaktor = `
        Faktor waktu
        <b class="text-info fw-bold">
            mulai berpengaruh
        </b>
        terhadap pergerakan harga.
    `;

    }

    // =====================
    // 3️⃣ MODERAT TAPI TIDAK KONSISTEN
    // =====================
    else if (cv < 20) {

        narasiUtama = `
        Terdapat fluktuasi harga antarbulan,
        namun tidak membentuk pola musiman yang konsisten.
    `;

        narasiFaktor = `
        Faktor waktu
        <b class="text-muted fw-bold">
            tidak memengaruhi harga secara signifikan
        </b>.
    `;

    }

    // =====================
    // 4️⃣ KUAT / SANGAT BERPENGARUH
    // =====================
    else {

        narasiUtama = `
        Harga menunjukkan variasi antarbulan,
        dengan kecenderungan lebih tinggi pada
        <b>${bulanMahal.map(b => b.bulan).join(', ')}</b>
        dan lebih rendah pada
        <b>${bulanMurah.map(b => b.bulan).join(', ')}</b>.
    `;

        narasiFaktor = `
        Faktor waktu
        <b class="text-danger fw-bold">
            sangat berpengaruh
        </b>
        terhadap pergerakan harga.
    `;
    }

    // =====================
    // OUTPUT KE UI
    // =====================
    $info.html(`
    ${narasiUtama}
    <br><br>
    ${narasiFaktor}
`);


    renderSeasonTablePrediksi(validMonths, mean);
}

/* ================= TABEL MUSIMAN ================= */
function renderSeasonTablePrediksi(validMonths, mean) {

    const $tbody = $('#seasonTableBodyPrediksi');
    if (!$tbody.length) return;

    $tbody.empty();

    validMonths.forEach(d => {

        const index = d.value / mean;

        let cls = 'text-muted';
        if (index > 1.1) cls = 'text-danger fw-semibold';
        else if (index < 0.95) cls = 'text-success fw-semibold';

        $tbody.append(`
            <tr>
                <td>${d.bulan}</td>
                <td>Rp ${Math.round(d.value).toLocaleString('id-ID')}</td>
                <td class="${cls}">${index.toFixed(2)}</td>
            </tr>
        `);
    });
}

/* ================= UTIL MUSIMAN ================= */
function getSeasonCategory(cv) {
    if (cv < 10) return 'Stabil';
    if (cv < 20) return 'Moderat';
    if (cv < 30) return 'Musiman Kuat';
    return 'Musiman Sangat Kuat';
}

function getSeasonTextClass(cv) {
    if (cv < 10) return 'text-success fw-bold';
    if (cv < 20) return 'text-info fw-bold';
    if (cv < 30) return 'text-warning fw-bold';
    return 'text-danger fw-bold';
}
