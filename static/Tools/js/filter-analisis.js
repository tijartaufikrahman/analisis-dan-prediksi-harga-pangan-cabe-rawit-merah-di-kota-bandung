/* ================= GLOBAL ================= */
let GLOBAL_RAW_DATA = [];
let GLOBAL_FILTERED_DATA = [];

let MAP_KOTA_PASAR = {};
let MAP_PASAR_KOTA = {};

let GLOBAL_MIN_DATE = null;
let GLOBAL_MAX_DATE = null;





/* ================= SPINNER ================= */
function showSourceSpinner() { $('#loadingSource').removeClass('d-none'); }
function hideSourceSpinner() { $('#loadingSource').addClass('d-none'); }
function showFilterSpinner() { $('#loadingFilter').removeClass('d-none'); }
function hideFilterSpinner() { $('#loadingFilter').addClass('d-none'); }

/* ================= APPLY SOURCE ================= */
$('#btnApplySource').on('click', function () {

    let file = $('#sumberData').val();
    if (!file) return alert('Pilih file terlebih dahulu');

    showSourceSpinner();

    $.get('/Home/ReadFile', { fileName: file }, function (lines) {

        GLOBAL_RAW_DATA = parseCSV(lines);
        GLOBAL_FILTERED_DATA = [...GLOBAL_RAW_DATA];

        buildRelations();

        fillSelect('#selCommodity', GLOBAL_RAW_DATA.map(x => x.komoditas).filter(Boolean));
        fillSelect('#selCity', GLOBAL_RAW_DATA.map(x => x.kota).filter(Boolean));
        fillSelect('#selMarket', GLOBAL_RAW_DATA.map(x => x.pasar).filter(Boolean));

        renderPreviewTable(GLOBAL_RAW_DATA);
        renderAgregasiTable([]);

        hideSourceSpinner();
        $(".d-analisis-filter").show();

        //INI PINDAH KE FILE
        btnApplyFilter();
        
    }).fail(function () {
        hideSourceSpinner();
    });

    
});

/* ================= RELATION & DATE ================= */
function buildRelations() {

    MAP_KOTA_PASAR = {};
    MAP_PASAR_KOTA = {};
    let dates = [];

    GLOBAL_RAW_DATA.forEach(d => {

        if (!d.kota || !d.pasar) return;

        MAP_KOTA_PASAR[d.kota] ??= new Set();
        MAP_KOTA_PASAR[d.kota].add(d.pasar);

        MAP_PASAR_KOTA[d.pasar] ??= new Set();
        MAP_PASAR_KOTA[d.pasar].add(d.kota);

        if (d.tanggal_obj instanceof Date && !isNaN(d.tanggal_obj)) {
            dates.push(d.tanggal_obj);
        }
    });

    if (dates.length) {
        GLOBAL_MIN_DATE = new Date(Math.min(...dates));
        GLOBAL_MAX_DATE = new Date(Math.max(...dates));

        // 🔥 FIX DATE INPUT (ANTI UTC BUG)
        $('#dateStart').val(formatDate(GLOBAL_MIN_DATE));
        $('#dateEnd').val(formatDate(GLOBAL_MAX_DATE));
    }
}

/* ================= DYNAMIC FILTER ================= */
$('#selCity').on('change', function () {

    let kota = $(this).val();
    let currentPasar = $('#selMarket').val();

    let pasarList = kota && MAP_KOTA_PASAR[kota]
        ? [...MAP_KOTA_PASAR[kota]]
        : [...new Set(GLOBAL_RAW_DATA.map(x => x.pasar).filter(Boolean))];

    $('#selMarket').html('<option value="">Semua</option>');
    pasarList.forEach(p => $('#selMarket').append(`<option value="${p}">${p}</option>`));

    if (currentPasar && pasarList.includes(currentPasar)) {
        $('#selMarket').val(currentPasar);
    }
});

function btnApplyFilter() {
    showFilterSpinner();

    // 🔥 PAKSA BROWSER RENDER SPINNER
    setTimeout(() => {

        let komoditas = $('#selCommodity').val();
        let kota = $('#selCity').val();
        let pasar = $('#selMarket').val();
        let periode = $('#selPeriode').val(); // daily | weekly | monthly

        let start = $('#dateStart').val()
            ? new Date($('#dateStart').val() + 'T00:00:00')
            : GLOBAL_MIN_DATE;

        let end = $('#dateEnd').val()
            ? new Date($('#dateEnd').val() + 'T23:59:59')
            : GLOBAL_MAX_DATE;

        GLOBAL_FILTERED_DATA = GLOBAL_RAW_DATA.filter(d =>
            (!komoditas || d.komoditas === komoditas) &&
            (!kota || d.kota === kota) &&
            (!pasar || d.pasar === pasar) &&
            d.tanggal_obj &&
            d.tanggal_obj >= start &&
            d.tanggal_obj <= end
        );

        renderPreviewTable(GLOBAL_FILTERED_DATA);

        let aggregated = aggregateByPeriod(GLOBAL_FILTERED_DATA, periode);
        renderAgregasiTable(aggregated);

        hideFilterSpinner();
        $(".d-analisis").show();

        console.log(GLOBAL_FILTERED_DATA);

        // INI  LANJUT KE FILE GRAFIK TREN
        renderTrendChartAnalisis(GLOBAL_FILTERED_DATA, periode);
        renderSeasonChartPrediksi(GLOBAL_FILTERED_DATA, $('#selPeriodePrediksi').val());
        renderVolatilityInfo(GLOBAL_FILTERED_DATA);
    }, 0);
}

/* ================= APPLY FILTER ================= */
//$('#btnApplyFilter').on('click', function () {

//    //showFilterSpinner();

//    //// 🔥 PAKSA BROWSER RENDER SPINNER
//    //setTimeout(() => {

//    //    let komoditas = $('#selCommodity').val();
//    //    let kota = $('#selCity').val();
//    //    let pasar = $('#selMarket').val();
//    //    let periode = $('#selPeriode').val(); // daily | weekly | monthly

//    //    let start = $('#dateStart').val()
//    //        ? new Date($('#dateStart').val() + 'T00:00:00')
//    //        : GLOBAL_MIN_DATE;

//    //    let end = $('#dateEnd').val()
//    //        ? new Date($('#dateEnd').val() + 'T23:59:59')
//    //        : GLOBAL_MAX_DATE;

//    //    GLOBAL_FILTERED_DATA = GLOBAL_RAW_DATA.filter(d =>
//    //        (!komoditas || d.komoditas === komoditas) &&
//    //        (!kota || d.kota === kota) &&
//    //        (!pasar || d.pasar === pasar) &&
//    //        d.tanggal_obj &&
//    //        d.tanggal_obj >= start &&
//    //        d.tanggal_obj <= end
//    //    );

//    //    renderPreviewTable(GLOBAL_FILTERED_DATA);

//    //    let aggregated = aggregateByPeriod(GLOBAL_FILTERED_DATA, periode);
//    //    renderAgregasiTable(aggregated);

//    //    hideFilterSpinner();

//    //    console.log(GLOBAL_FILTERED_DATA);

//    //    // INI  LANJUT KE FILE GRAFIK TREN
//    //    renderTrendChartAnalisis(GLOBAL_FILTERED_DATA, periode);

//    //}, 0);
//});

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

    $('#previewCount').text(data.length);
    let tbody = $('#tblAgregasiBody');
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
                <td>${i + 1}</td>
                <td>${d.periode}</td>
                <td class="text-end">${d.avg.toLocaleString('id-ID', { maximumFractionDigits: 0 })}</td>
                <td>${arah}</td>
            </tr>
        `);

        prev = d.avg;
    });
}

/* ================= PREVIEW TABLE ================= */
function renderPreviewTable(data) {    

    let tbody = $('#previewTableBody');
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
