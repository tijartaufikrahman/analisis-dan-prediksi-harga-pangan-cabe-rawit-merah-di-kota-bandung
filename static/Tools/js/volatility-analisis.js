/* =====================================================
   VOLATILITAS HARGA (Coefficient of Variation / CV)
   KATEGORI:
   < 10%        : Stabil
   10% – < 20%  : Moderat
   20% – < 30%  : Fluktuatif
   ≥ 30%        : Sangat Fluktuatif
===================================================== */

function renderVolatilityInfo(data) {

    // =============================
    // VALIDASI DATA
    // =============================
    if (!data || !data.length) {
        $('#cvValue').text('-');
        $('#cvBadge')
            .removeClass()
            .addClass('badge bg-secondary')
            .text('-');
        $('#cvInsight').text('Belum ada data volatilitas.');
        return;
    }

    // =============================
    // AMBIL DATA HARGA
    // =============================
    const prices = data
        .map(d => d.harga)
        .filter(v => !isNaN(v));

    if (prices.length < 2) {
        $('#cvValue').text('-');
        $('#cvBadge')
            .removeClass()
            .addClass('badge bg-secondary')
            .text('Data tidak cukup');
        $('#cvInsight').text('Data tidak mencukupi untuk menghitung volatilitas.');
        return;
    }

    // =============================
    // HITUNG CV
    // =============================
    const mean =
        prices.reduce((a, b) => a + b, 0) / prices.length;

    const variance =
        prices.reduce((s, v) => s + Math.pow(v - mean, 2), 0) / prices.length;

    const stdDev = Math.sqrt(variance);
    const cv = (stdDev / mean) * 100;

    // =============================
    // TENTUKAN KATEGORI
    // =============================
    let kategori = '';
    let badgeClass = '';
    let insight = '';

    if (cv < 10) {
        kategori = 'Stabil';
        badgeClass = 'bg-success';
        insight = `
            Harga relatif stabil dengan fluktuasi yang kecil
            dan berada dalam rentang yang terkendali.
        `;
    }
    else if (cv < 20) {
        kategori = 'Moderat';
        badgeClass = 'bg-info';
        insight = `
            Harga mengalami fluktuasi yang masih terkendali,
            dengan perubahan yang cukup terasa namun tidak ekstrem.
        `;
    }
    else if (cv < 30) {
        kategori = 'Fluktuatif';
        badgeClass = 'bg-warning text-dark';
        insight = `
            Harga menunjukkan fluktuasi yang cukup besar,
            menandakan tingkat ketidakstabilan harga yang meningkat.
        `;
    }
    else {
        kategori = 'Sangat Fluktuatif';
        badgeClass = 'bg-danger';
        insight = `
            Harga sangat tidak stabil dengan fluktuasi yang besar
            antarperiode dan berpotensi meningkatkan risiko harga.
        `;
    }

    // =============================
    // OUTPUT KE UI
    // =============================
    $('#cvValue').text(cv.toFixed(2) + '%');

    $('#cvBadge')
        .removeClass()
        .addClass(`badge ${badgeClass}`)
        .text(kategori);

    $('#cvInsight').html(insight);
}
