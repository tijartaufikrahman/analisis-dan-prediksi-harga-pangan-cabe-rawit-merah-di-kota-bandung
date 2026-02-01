/* =====================================================
   TREND GRAFIK PREDIKSI (jQuery Version)
   MODE: daily | weekly | monthly
===================================================== */

let chartTrendPrediksi = null;

/* ================= NORMALISASI TANGGAL ================= */
function normalisasiTanggalPrediksi(tanggal, mode) {

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

/* ================= RENDER TREND CHART ================= */
function renderTrendChartAnalisis(data, modeWaktu) {

    const $canvas = $('#trendChartPrediksi');
    if (!$canvas.length) return;

    if (chartTrendPrediksi) {
        chartTrendPrediksi.destroy();
        chartTrendPrediksi = null;
    }

    /* ================= GROUPING DATA ================= */
    const bucket = {};

    data.forEach(d => {

        if (!(d.tanggal_obj instanceof Date) || isNaN(d.tanggal_obj)) return;
        if (isNaN(d.harga)) return;

        const t = normalisasiTanggalPrediksi(d.tanggal_obj, modeWaktu);
        const key = t.getTime();

        if (!bucket[key]) {
            bucket[key] = {
                tanggal: t,
                total: 0,
                count: 0
            };
        }

        bucket[key].total += d.harga;
        bucket[key].count++;
    });

    const dataGrafik = Object.values(bucket)
        .map(x => ({
            x: x.tanggal,
            y: Math.round(x.total / x.count)
        }))
        .sort((a, b) => a.x - b.x);

    /* 🔥 PENTING: KIRIM DATA MENTAH JUGA */
    renderInsightPrediksi(
        dataGrafik,
        GLOBAL_FILTERED_DATA,
        modeWaktu
    );

    renderTabelTrendPrediksi(dataGrafik, modeWaktu);

    /* ================= CHART.JS ================= */
    chartTrendPrediksi = new Chart($canvas[0], {
        type: 'line',
        data: {
            datasets: [{
                label: `Tren Harga (${modeWaktu})`,
                data: dataGrafik,
                borderColor: '#3aa0ff',
                tension: 0.3,
                pointRadius: 0
            }]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            interaction: {
                mode: 'index',
                intersect: false
            },
            plugins: {
                tooltip: {
                    callbacks: {
                        title: function (items) {
                            const t = new Date(items[0].parsed.x);

                            if (modeWaktu === 'weekly') {
                                return `Minggu ${getWeekNumber(t)} ${t.getFullYear()}`;
                            }

                            if (modeWaktu === 'monthly') {
                                return t.toLocaleDateString('id-ID', {
                                    month: 'long',
                                    year: 'numeric'
                                });
                            }

                            return t.toLocaleDateString('id-ID', {
                                day: '2-digit',
                                month: 'long',
                                year: 'numeric'
                            });
                        },
                        label: c => `Rp ${c.parsed.y.toLocaleString('id-ID')}`
                    }
                }
            },
            scales: {
                x: {
                    type: 'time',
                    time: { unit: modeWaktu === 'daily' ? 'day' : 'month' }
                },
                y: {
                    ticks: {
                        callback: v => 'Rp ' + v.toLocaleString('id-ID')
                    }
                }
            }
        }
    });
}

/* ================= INSIGHT (DENGAN PASAR & KOTA) ================= */
function renderInsightPrediksi(dataGrafik, rows, modeWaktu) {

    const $el = $('#infoVolatilitasGlobal');

    if (!dataGrafik || dataGrafik.length < 2) {
        $el.text('Data tidak mencukupi untuk analisis tren');
        return;
    }

    /* ================= TREN GLOBAL ================= */
    const awal = dataGrafik[0].y;
    const akhir = dataGrafik[dataGrafik.length - 1].y;
    const perubahan = ((akhir - awal) / awal) * 100;

    let narasi = 'Harga relatif stabil';
    if (perubahan > 1)
        narasi = `Harga cenderung meningkat sekitar ${perubahan.toFixed(1)}%`;
    else if (perubahan < -1)
        narasi = `Harga cenderung menurun sekitar ${Math.abs(perubahan).toFixed(1)}%`;

    /* ================= RATA-RATA ================= */
    const rata =
        dataGrafik.reduce((s, d) => s + d.y, 0) / dataGrafik.length;

    /* ================= MIN & MAX ================= */
    const minPoint = dataGrafik.reduce((a, b) => b.y < a.y ? b : a);
    const maxPoint = dataGrafik.reduce((a, b) => b.y > a.y ? b : a);

    /* ================= NORMALISASI (SAMA DENGAN GRAFIK) ================= */
    const norm = t => normalisasiTanggalPrediksi(t, modeWaktu).getTime();

    /* ================= CARI ROW REPRESENTATIF ================= */
    function cariRow(point) {

        const key = norm(point.x);

        const kandidat = rows.filter(r =>
            r.tanggal_obj &&
            norm(r.tanggal_obj) === key
        );

        if (!kandidat.length) return null;

        return kandidat.reduce((a, b) =>
            Math.abs(b.harga - point.y) <
                Math.abs(a.harga - point.y)
                ? b : a
        );
    }

    const rowMin = cariRow(minPoint);
    const rowMax = cariRow(maxPoint);

    /* ================= FORMAT ================= */
    const fmtTanggal = d =>
        new Date(d).toLocaleDateString('id-ID', {
            day: '2-digit',
            month: 'long',
            year: 'numeric'
        });

    const fmtLokasi = r =>
        r ? `${r.pasar ?? '-'}, ${r.kota ?? '-'}` : '-';

    /* ================= OUTPUT ================= */
    $el.html(`
        <b>Periode:</b> ${modeWaktu}<br>
        ${narasi}<br><br>

        <b>Rata-rata Harga:</b><br>
        Rp ${Math.round(rata).toLocaleString('id-ID')}<br><br>

        <b>Harga Terendah:</b><br>
        Rp ${minPoint.y.toLocaleString('id-ID')}<br>
        <span class="dash-small text-muted">
            ${fmtTanggal(minPoint.x)} — ${fmtLokasi(rowMin)}
        </span><br><br>

        <b>Harga Tertinggi:</b><br>
        Rp ${maxPoint.y.toLocaleString('id-ID')}<br>
        <span class="dash-small text-muted">
            ${fmtTanggal(maxPoint.x)} — ${fmtLokasi(rowMax)}
        </span>
    `);
}

/* ================= TABEL TREND ================= */
function renderTabelTrendPrediksi(dataGrafik, modeWaktu) {

    const $tbody = $('#tabelHargaBody');
    const $jumlah = $('#jumlahObservasi');

    if (!$tbody.length) return;

    $tbody.empty();
    $jumlah.text(dataGrafik.length);

    const formatPeriode = d => {
        const t = new Date(d);

        if (modeWaktu === 'monthly')
            return t.toLocaleDateString('id-ID', { month: 'long', year: 'numeric' });

        if (modeWaktu === 'weekly')
            return `Minggu ${getWeekNumber(t)} ${t.getFullYear()}`;

        return t.toLocaleDateString('id-ID', {
            day: '2-digit',
            month: 'long',
            year: 'numeric'
        });
    };

    dataGrafik.forEach((d, i) => {

        let arah = 'Tetap';
        let cls = 'text-muted';
        let ikon = '<i class="bi bi-dash"></i>';

        if (i > 0) {
            const prev = dataGrafik[i - 1].y;

            if (d.y > prev) {
                arah = 'Naik';
                cls = 'text-success';
                ikon = '<i class="bi bi-arrow-up"></i>';
            }
            else if (d.y < prev) {
                arah = 'Turun';
                cls = 'text-danger';
                ikon = '<i class="bi bi-arrow-down"></i>';
            }
        }

        $tbody.append(`
            <tr>
                <td class="text-center">${i + 1}</td>
                <td>${formatPeriode(d.x)}</td>
                <td>Rp ${d.y.toLocaleString('id-ID')}</td>
                <td class="${cls}">${arah} ${ikon}</td>
            </tr>
        `);
    });
}
