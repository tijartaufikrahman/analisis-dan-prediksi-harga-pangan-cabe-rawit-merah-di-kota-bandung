/* ================= GLOBAL ================= */
let GLOBAL_RAW_DATA_PREDIKSI = [];
let GLOBAL_FILTERED_DATA_PREDIKSI = [];


let MAP_KOTA_PASAR_PREDIKSI = {};
let MAP_PASAR_KOTA_PREDIKSI = {};

let GLOBAL_MIN_DATE = null;
let GLOBAL_MAX_DATE = null;

let GLOBAL_AGGREGATED_DATA = [];
let GLOBAL_TRAIN_DATA = [];
let GLOBAL_TEST_DATA = [];

console.log("runStationarityTest(aggregated);");

/* ================= SPINNER ================= */
function showSourceSpinner() { $('#loadingSource').removeClass('d-none'); }
function hideSourceSpinner() { $('#loadingSource').addClass('d-none'); }
function showFilterSpinner() { $('#loadingFilter').removeClass('d-none'); }
function hideFilterSpinner() { $('#loadingFilter').addClass('d-none'); }

/* ================= APPLY SOURCE ================= */
$('#btnApplySourcePrediksi').on('click', function () {

    let file = $('#sumberDataPrediksi').val();
    if (!file) return alert('Pilih file terlebih dahulu');

    showSourceSpinner();

    $.get('/Home/ReadFile', { fileName: file }, function (lines) {

        GLOBAL_RAW_DATA_PREDIKSI = parseCSV(lines);
        GLOBAL_FILTERED_DATA_PREDIKSI = [...GLOBAL_RAW_DATA_PREDIKSI];
       

        buildRelations();

        fillSelect('#selCommodityPrediksi', GLOBAL_RAW_DATA_PREDIKSI.map(x => x.komoditas).filter(Boolean));
        fillSelect('#selCityPrediksiPrediksi', GLOBAL_RAW_DATA_PREDIKSI.map(x => x.kota).filter(Boolean));
        fillSelect('#selMarketPrediksi', GLOBAL_RAW_DATA_PREDIKSI.map(x => x.pasar).filter(Boolean));

        renderPreviewTable(GLOBAL_RAW_DATA_PREDIKSI);
        renderAgregasiTable([]);

        hideSourceSpinner();

        // pindah file
        btnApplyFilterPrediksi();
    }).fail(function () {
        hideSourceSpinner();
    });
});

/* ================= RELATION & DATE ================= */
function buildRelations() {

    MAP_KOTA_PASAR_PREDIKSI = {};
    MAP_PASAR_KOTA_PREDIKSI = {};
    let dates = [];

    GLOBAL_RAW_DATA_PREDIKSI.forEach(d => {

        if (!d.kota || !d.pasar) return;

        MAP_KOTA_PASAR_PREDIKSI[d.kota] ??= new Set();
        MAP_KOTA_PASAR_PREDIKSI[d.kota].add(d.pasar);

        MAP_PASAR_KOTA_PREDIKSI[d.pasar] ??= new Set();
        MAP_PASAR_KOTA_PREDIKSI[d.pasar].add(d.kota);

        if (d.tanggal_obj instanceof Date && !isNaN(d.tanggal_obj)) {
            dates.push(d.tanggal_obj);
        }
    });

    if (dates.length) {
        GLOBAL_MIN_DATE = new Date(Math.min(...dates));
        GLOBAL_MAX_DATE = new Date(Math.max(...dates));

        // 🔥 FIX DATE INPUT (ANTI UTC BUG)
        $('#dateStartPrediksi').val(formatDate(GLOBAL_MIN_DATE));
        $('#dateEndPrediksi').val(formatDate(GLOBAL_MAX_DATE));
    }
}

/* ================= DYNAMIC FILTER ================= */
$('#selCityPrediksiPrediksi').on('change', function () {

    let kota = $(this).val();
    let currentPasar = $('#selMarketPrediksi').val();

    let pasarList = kota && MAP_KOTA_PASAR_PREDIKSI[kota]
        ? [...MAP_KOTA_PASAR_PREDIKSI[kota]]
        : [...new Set(GLOBAL_RAW_DATA_PREDIKSI.map(x => x.pasar).filter(Boolean))];

    $('#selMarketPrediksi').html('<option value="">Semua</option>');
    pasarList.forEach(p => $('#selMarketPrediksi').append(`<option value="${p}">${p}</option>`));

    if (currentPasar && pasarList.includes(currentPasar)) {
        $('#selMarketPrediksi').val(currentPasar);
    }
});

/* ================= APPLY FILTER ================= */

function btnApplyFilterPrediksi() {
    showFilterSpinner();

    // 🔥 PAKSA BROWSER RENDER SPINNER
    setTimeout(() => {

        let komoditas = $('#selCommodityPrediksi').val();
        let kota = $('#selCityPrediksiPrediksi').val();
        let pasar = $('#selMarketPrediksi').val();
        let periode = $('#selPeriodePrediksi').val(); // daily | weekly | monthly

        let start = $('#dateStartPrediksi').val()
            ? new Date($('#dateStartPrediksi').val() + 'T00:00:00')
            : GLOBAL_MIN_DATE;

        let end = $('#dateEndPrediksi').val()
            ? new Date($('#dateEndPrediksi').val() + 'T23:59:59')
            : GLOBAL_MAX_DATE;

        GLOBAL_FILTERED_DATA_PREDIKSI = GLOBAL_RAW_DATA_PREDIKSI.filter(d =>
            (!komoditas || d.komoditas === komoditas) &&
            (!kota || d.kota === kota) &&
            (!pasar || d.pasar === pasar) &&
            d.tanggal_obj &&
            d.tanggal_obj >= start &&
            d.tanggal_obj <= end
        );

        renderPreviewTable(GLOBAL_FILTERED_DATA_PREDIKSI);

        // //
        let aggregated = aggregateByPeriod(GLOBAL_FILTERED_DATA_PREDIKSI, periode);

        // simpan full aggregated
        GLOBAL_AGGREGATED_DATA = aggregated;

        // ambil input test size
        let testSize = parseInt($('#inputTestSizePrediksi').val()) || 0;

        // potong train & test
        let split = splitTrainTestFromAggregated(aggregated, testSize);

        // simpan global
        GLOBAL_TRAIN_DATA = split.train;
        GLOBAL_TEST_DATA = split.test;

        // render tabel pakai FULL aggregated (bukan train/test)
        renderAgregasiTable(aggregated);

        // contoh pemanggilan API / uji stasioner
        runStationarityTest(GLOBAL_TRAIN_DATA);

        console.log('FULL', GLOBAL_AGGREGATED_DATA);
        console.log('TRAIN', GLOBAL_TRAIN_DATA);
        console.log('TEST', GLOBAL_TEST_DATA);

        // //

        
        

        hideFilterSpinner();

        $('.d-display-prediksi').show();



    }, 0); 
}

    


/* ================= AGREGASI ================= */
function aggregateByPeriod(data, period) {

    let map = {};

    data.forEach(d => {

        if (!(d.tanggal_obj instanceof Date) || isNaN(d.tanggal_obj)) return;
        if (isNaN(d.harga)) return;

        let dt = d.tanggal_obj;
        let key, label, sortDate;

        if (period === 'daily') {
            key = formatDate(dt); // 🔥 AMAN TANPA UTC
            label = dt.toLocaleDateString('id-ID');
            sortDate = new Date(dt.getFullYear(), dt.getMonth(), dt.getDate(), 12);
        }
        else if (period === 'weekly') {
            let temp = new Date(dt);
            temp.setDate(temp.getDate() + 3 - ((temp.getDay() + 6) % 7));
            let week1 = new Date(temp.getFullYear(), 0, 4);
            let week = 1 + Math.round(
                ((temp - week1) / 86400000 - 3 + ((week1.getDay() + 6) % 7)) / 7
            );

            key = `${temp.getFullYear()}-W${week}`;
            label = `Minggu ke-${week} ${temp.getFullYear()}`;
            sortDate = temp;
        }
        else if (period === 'monthly') {
            let y = dt.getFullYear();
            let m = dt.getMonth();
            key = `${y}-${m}`;
            label = new Date(y, m, 1).toLocaleString('id-ID', {
                month: 'long',
                year: 'numeric'
            });
            sortDate = new Date(y, m, 1);
        }
        else return;

        if (!map[key]) map[key] = { label, total: 0, count: 0, sortDate };
        map[key].total += d.harga;
        map[key].count++;
    });

    return Object.values(map)
        .sort((a, b) => a.sortDate - b.sortDate)
        .map(x => ({
            periode: x.label,
            avg: Math.round(x.total / x.count)
        }));
}

/* ================= TABLE AGREGASI ================= */
function renderAgregasiTable(data) {

    $('#previewCountPrediksi').text(data.length);
    let tbody = $('#tblAgregasiBodyPrediksi');
    tbody.empty();

    if (!data || !data.length) {
        tbody.append(`
            <tr>
                <td colspan="4" class="text-center text-muted">
                    Belum ada data
                </td>
            </tr>
        `);
        return;
    }

    let prev = null;

    data.forEach((d, i) => {
        let arah = '-';
        if (prev !== null) {
            if (d.avg > prev) arah = 'Naik';
            else if (d.avg < prev) arah = 'Turun';
            else arah = 'Tetap';
        }

        tbody.append(`
            <tr>
                <td class="text-center" >${i + 1}</td>
                <td class="text-center">${d.periode}</td>
                <td class="text-center">${d.avg.toLocaleString('id-ID', { maximumFractionDigits: 0 })}</td>
                <td class="text-center">${arah}</td>
            </tr>
        `);


        prev = d.avg;
    });
}

/* ================= PREVIEW TABLE ================= */
function renderPreviewTable(data) {

    let tbody = $('#previewTableBodyPrediksi');
    if (!tbody.length) return;

    tbody.empty();

    if (!data || !data.length) {
        tbody.append(`
            <tr>
                <td colspan="7" class="text-center text-muted">
                    Data tidak ditemukan
                </td>
            </tr>
        `);
        return;
    }

    data.forEach((d, i) => {
        tbody.append(`
            <tr>
                <td>${i + 1}</td>
                <td>${d.komoditas ?? '-'}</td>
                <td>${d.tanggal_raw ?? '-'}</td>
                <td>${d.provinsi ?? '-'}</td>
                <td>${d.kota ?? '-'}</td>
                <td>${d.pasar ?? '-'}</td>
                <td class="text-end">${isNaN(d.harga) ? '-' : d.harga}</td>
            </tr>
        `);
    });
}

/* ================= CSV PARSER ================= */
function parseCSV(lines) {

    let data = [];

    for (let i = 1; i < lines.length; i++) {

        let row = lines[i];
        if (!row || !row.trim()) continue;

        let c = splitLine(row);
        if (c.length < 7) continue;

        let tgl = c[2]?.trim();

        data.push({
            komoditas: c[1]?.trim(),
            tanggal_raw: tgl,
            tanggal_obj: parseTanggalCSV(tgl),
            provinsi: c[3]?.trim(),
            kota: c[4]?.trim(),
            pasar: c[5]?.trim(),
            harga: parseHarga(c[6])
        });
    }

    return data;
}

/* ================= UTIL ================= */
function parseTanggalCSV(raw) {
    if (!raw) return null;
    if (raw.includes('/')) {
        let [d, m, y] = raw.split('/');
        // 🔥 SET JAM TENGAH HARI BIAR ANTI UTC SHIFT
        return new Date(y, m - 1, d, 12, 0, 0);
    }
    return new Date(raw);
}

function parseHarga(raw) {
    if (raw === undefined || raw === null || raw.toString().trim() === '') return NaN;
    return Number(
        raw.toString()
            .replace(/Rp/gi, '')
            .replace(/\./g, '')
            .replace(/,/g, '')
            .replace(/\s+/g, '')
    );
}

function splitLine(l) {
    if (l.includes('\t')) return l.split('\t');
    if (l.includes(';')) return l.split(';');
    return l.split(',');
}

function fillSelect(sel, arr) {
    let unique = [...new Set(arr)];
    $(sel).html('<option value="">Semua</option>');
    unique.forEach(v => $(sel).append(`<option value="${v}">${v}</option>`));
}

/* ================= DATE FORMAT (ANTI UTC BUG) ================= */
function formatDate(d) {
    if (!(d instanceof Date) || isNaN(d)) return '';
    const y = d.getFullYear();
    const m = String(d.getMonth() + 1).padStart(2, '0');
    const day = String(d.getDate()).padStart(2, '0');
    return `${y}-${m}-${day}`;
}

function splitTrainTestFromAggregated(aggregated, testSize) {

    if (!Array.isArray(aggregated) || aggregated.length === 0) {
        return { train: [], test: [] };
    }

    if (testSize <= 0) {
        return {
            train: [...aggregated],
            test: []
        };
    }

    if (testSize >= aggregated.length) {
        return {
            train: [],
            test: [...aggregated]
        };
    }

    let train = aggregated.slice(0, aggregated.length - testSize);
    let test = aggregated.slice(aggregated.length - testSize);

    return { train, test };
}
