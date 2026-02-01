$('#btnRunSTLGlobal').on('click', async function () {

    console.log(btnRunSTLGlobal);

    if (!GLOBAL_AGGREGATED_DATA || GLOBAL_AGGREGATED_DATA.length < 3) {
        alert('Data agregasi belum tersedia');
        return;
    }

    // 🔥 ambil GLOBAL AVERAGE (avg)
    const series = GLOBAL_AGGREGATED_DATA.map(x => x.avg);

    $('#stlResultGlobal').html('⏳ Memanggil API STL...');

    try {
        const res = await fetch(
            'http://127.0.0.1:8000/test/seasonality-stl',
            {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({
                    data: series,
                    max_s: 12,
                    threshold: 0.2
                })
            }
        );

        if (!res.ok) throw new Error('API gagal');

        const json = await res.json();
        renderSTLResult(json);

    } catch (e) {
        $('#stlResultGlobal').html(
            `<span class="text-danger">${e.message}</span>`
        );
    }
});

function renderSTLResult(res) {

    let html = `
        <p><b>Musiman:</b>
        <span style="font-weight:bold;color:${res.has_seasonality ? 'green' : 'red'}">
            ${res.has_seasonality}
        </span></p>

        <p><b>Musiman Signifikan :</b> ${res.recommendation}</p>

        <table class="table table-bordered mt-2">
            <thead>
                <tr>
                    <th>Periode (s)</th>
                    <th>Seasonal Strength</th>
                </tr>
            </thead>
            <tbody>
    `;
    
    

    res.tested_periods.forEach(p => {
        html += `
            <tr>
                <td class="text-center">${p.s}</td>
                <td class="text-center">${p.seasonal_strength}</td>
            </tr>
        `;
    });

    html += `</tbody></table>`;

    $('#stlResultGlobal').html(html);
}
