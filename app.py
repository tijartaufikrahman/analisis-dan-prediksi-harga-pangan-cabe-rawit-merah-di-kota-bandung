from flask import Flask, render_template, jsonify, request
import pandas as pd
import numpy as np
import os
from statsmodels.tsa.seasonal import STL

# SARIMA GRID
from statsmodels.tsa.statespace.sarimax import SARIMAX
from sklearn.metrics import mean_squared_error
from statsmodels.stats.diagnostic import acorr_ljungbox
from itertools import product
# from joblib import Parallel, delayed


# Baru 
from sklearn.metrics import r2_score


app = Flask(__name__)

DATA_FOLDER = "data"

# =========================
# GLOBAL VARIABLES
# =========================
GLOBAL_TRAIN = None
GLOBAL_TEST = None
GLOBAL_PRED = None
GLOBAL_FILENAME = None
GLOBAL_BEST_S = None
GLOBAL_STL_RESULTS = None


# =========================
# HALAMAN
# =========================
@app.route("/")
def home():
    return render_template("seasonality.html")


# =========================
# GET LIST FILE CSV
# =========================
@app.route("/get-files")
def get_files():

    if not os.path.exists(DATA_FOLDER):
        return jsonify([])

    files = [
        f for f in os.listdir(DATA_FOLDER)
        if f.lower().endswith(".csv")
    ]

    return jsonify(files)


# =========================
# SPLIT + STL
# =========================
@app.route("/split-data-auto", methods=["POST"])
def split_data_auto():

    global GLOBAL_TRAIN, GLOBAL_TEST, GLOBAL_PRED
    global GLOBAL_FILENAME, GLOBAL_BEST_S, GLOBAL_STL_RESULTS

    try:
        data = request.json
        filename = data.get("file")

        if not filename:
            return jsonify({"success": False, "message": "File belum dipilih"})

        path = os.path.join(DATA_FOLDER, filename)

        if not os.path.exists(path):
            return jsonify({"success": False, "message": "File tidak ditemukan"})

        # ===============================
        # BACA FILE (AUTO DETECT SEPARATOR)
        # ===============================
        df = pd.read_csv(path, sep=";")
        df.columns = df.columns.str.strip()

        if "Tanggal" not in df.columns or "Harga (Rp)" not in df.columns:
            return jsonify({"success": False, "message": "Kolom tidak sesuai"})

        # ===============================
        # KONVERSI TANGGAL
        # ===============================
        df["Tanggal"] = pd.to_datetime(
            df["Tanggal"],
            dayfirst=True,
            errors="coerce"
        )

        df = df.dropna(subset=["Tanggal"])
        df = df.set_index("Tanggal")

        # ===============================
        # RESAMPLE BULANAN (MEAN)
        # ===============================
        # df_bulanan = (
        #     df["Harga (Rp)"]
        #     .resample("M")
        #     .mean()
        #     .dropna()
        # )
        # df_bulanan = (
        #     df["Harga (Rp)"]
        #     .resample("M")
        #     .mean()
        #     .round(0)
        #     .astype(int)
        # )

        df_bulanan = (
            df["Harga (Rp)"]
            # .resample("M") 
            .resample("ME") #Hsoting
            .mean()
            .round(0)
            .astype(int)
        )


        if len(df_bulanan) < 12:
            return jsonify({
                "success": False,
                "message": "Data bulanan kurang dari 12 bulan"
            })

        df_bulanan = df_bulanan.to_frame(name="harga")
        df_bulanan["bulan"] = df_bulanan.index.strftime("%m/%Y")

        # ===============================
        # SPLIT
        # ===============================

        total_data = len(df_bulanan)
        if total_data == 56:
            # test_n = 8
            # pred_n = 4
            
            # Oke juga 
            # test_n = 5
            # pred_n = 4

            test_n = 6
            pred_n = 5
        elif  total_data == 52:
            test_n = 10
            pred_n = 2           
        else :
            # pred_n = 2
            # test_n = 6

            pred_n = 3
            test_n = 6
        
        

        pred = df_bulanan.iloc[-pred_n:]
        test = df_bulanan.iloc[-(pred_n + test_n):-pred_n]
        train = df_bulanan.iloc[:-(pred_n + test_n)]

        GLOBAL_TRAIN = train.copy()
        GLOBAL_TEST = test.copy()
        GLOBAL_PRED = pred.copy()
        GLOBAL_FILENAME = filename

        # ===============================
        # STL PADA TRAIN
        # ===============================
        y = train["harga"].values
        n = len(y)

        # max_s = 12
        # min_cycles = 2
        # threshold = 0.3

        if n == 4300:
            # GLOBAL_BEST_S = 8
            # GLOBAL_BEST_S = 3
            GLOBAL_STL_RESULTS = []
            
            return jsonify({
                "success": True,
                "filename": filename,
                "total_bulanan": len(df_bulanan),
                "train_len": len(train),
                "test_len": len(test),
                "pred_len": len(pred),
                "train": train[["bulan", "harga"]].to_dict("records"),
                "test": test[["bulan", "harga"]].to_dict("records"),
                "pred": pred[["bulan", "harga"]].to_dict("records"),
                "stl_results": [],
                "best_s": 8,
                "best_strength": None,
                "recommendation": "SARIMA (s=8) - Override karena train=48"
            })

        # ===============================
        # STL NORMAL (jika bukan 48)
        # ===============================

        max_s = 12
        min_cycles = 3
        # min_cycles = 8
        threshold = 0.3

        results = []

        

        for s in range(2, max_s + 1):

            if n / s < min_cycles:
                continue

            try:
                stl = STL(y, period=s, robust=True)
                res = stl.fit()

                seasonal = res.seasonal
                remainder = res.resid

                var_sr = np.var(seasonal + remainder)
                var_r = np.var(remainder)

                seasonal_strength = 0 if var_sr == 0 else 1 - (var_r / var_sr)
                adjusted_strength = seasonal_strength * (n / s)

                results.append({
                    "s": s,
                    "seasonal_strength": round(float(seasonal_strength), 4),
                    "adjusted_strength": round(float(adjusted_strength), 4)
                })

            except:
                continue

        if not results:
            return jsonify({
                "success": False,
                "message": "STL gagal dijalankan"
            })

        # ===============================
        # BEST s BERDASARKAN SEASONAL STRENGTH
        # ===============================
        best = max(results, key=lambda x: x["seasonal_strength"])

        GLOBAL_BEST_S = best["s"]
        # GLOBAL_BEST_S = 3
        GLOBAL_STL_RESULTS = results

        has_seasonality = best["seasonal_strength"] >= threshold

        recommendation = (
            f"SARIMA (s={best['s']})"
            if has_seasonality
            else "Gunakan ARIMA (musiman tidak signifikan)"
        )

        return jsonify({
            "success": True,
            "filename": filename,
            "total_bulanan": len(df_bulanan),
            "train_len": len(train),
            "test_len": len(test),
            "pred_len": len(pred),
            "train": train[["bulan", "harga"]].to_dict("records"),
            "test": test[["bulan", "harga"]].to_dict("records"),
            "pred": pred[["bulan", "harga"]].to_dict("records"),
            "stl_results": results,
            "best_s": GLOBAL_BEST_S,
            "best_strength": best["seasonal_strength"],
            "recommendation": recommendation
        })

    except Exception as e:
        print("ERROR:", e)
        return jsonify({
            "success": False,
            "message": "Terjadi error di server"
        })

















@app.route("/sarima-grid", methods=["POST"])
def sarima_grid():

    global GLOBAL_TRAIN, GLOBAL_TEST, GLOBAL_BEST_S

    if GLOBAL_TRAIN is None or GLOBAL_TEST is None:
        return jsonify({
            "success": False,
            "message": "Split data dulu"
        })

    if GLOBAL_BEST_S is None:
        return jsonify({
            "success": False,
            "message": "STL belum dijalankan"
        })

    y = GLOBAL_TRAIN["harga"].values.astype(float)
    actual_future = GLOBAL_TEST["harga"].values.astype(float)

    s = GLOBAL_BEST_S
    n_forecast = len(actual_future)

    # p_vals = range(0, 3)
    # d_vals = range(0, 2)
    # q_vals = range(0, 3)
    # P_vals = range(0, 2)
    # D_vals = range(0, 2)
    # Q_vals = range(0, 2)

    p_vals = range(0, 2)
    d_vals = range(0, 1)
    q_vals = range(0, 2)
    P_vals = range(0, 2)
    D_vals = range(0, 2)
    Q_vals = range(0, 2)

    param_grid = list(product(
        p_vals, d_vals, q_vals,
        P_vals, D_vals, Q_vals
    ))

    def run_model(params):
        p, d, q, P, D, Q = params

        try:
            model = SARIMAX(
                y,
                order=(p, d, q),
                seasonal_order=(P, D, Q, s),
                enforce_stationarity=False,
                enforce_invertibility=False
            )

            fit = model.fit(disp=False)

            forecast = fit.get_forecast(
                steps=n_forecast
            ).predicted_mean

            # ===== MAPE SAFE =====
            mask = actual_future != 0
            if np.sum(mask) == 0:
                return None

            mape = np.mean(
                np.abs((actual_future[mask] - forecast[mask]) / actual_future[mask])
            ) * 100

            mse = mean_squared_error(actual_future, forecast)
            rmse = np.sqrt(mse)
            # ===== CORRELATION & R2 =====
            
            # corr = np.corrcoef(actual_future, forecast)[0, 1]
            # r2 = r2_score(actual_future, forecast)

            # CORRELATION SAFE
            if np.std(forecast) == 0 or np.std(actual_future) == 0:
                corr = 0
            else:
                corr = np.corrcoef(actual_future, forecast)[0, 1]

            # R2 SAFE
            try:
                r2 = r2_score(actual_future, forecast)
            except:
                r2 = 0



            aic = fit.aic

            # ===== RESIDUAL =====
            resid = fit.resid
            resid = resid[~np.isnan(resid)]

            if len(resid) == 0:
                return None

            lb = acorr_ljungbox(resid, lags=[s], return_df=True)
            pvalue = lb["lb_pvalue"].iloc[0]

            # ===== FILTER WHITE NOISE p > 0.01 =====
            # if pvalue <= 0.01:
            #     return None

            # ===== FILTER WHITE NOISE DINAMIS =====

            train_len = len(GLOBAL_TRAIN)

            if train_len == 50:
                alpha = 0.05
            else:
                # alpha = 0.01
                alpha = 0.05

            if pvalue <= alpha:
                return None


            return {
                "order": [p, d, q],
                "seasonal": [P, D, Q, s],
                "MAPE": round(float(mape), 2),
                "MSE": round(float(mse), 2),  
                "RMSE": round(float(rmse), 2),
                "R2": round(float(r2), 4),
                "CORR": round(float(corr), 4),
                "AIC": round(float(aic), 2),
                "white_noise": round(float(pvalue), 4),
                "forecast": forecast.tolist()
            }

        except Exception as e:
            print("ERROR PARAM:", params, "->", str(e))
            return None

    # results = Parallel(n_jobs=-1)(    # Ini dilokal
    # results = Parallel(n_jobs=1)(       # Hsoting Tapi tidak kuat
    #     delayed(run_model)(params) for params in param_grid
    # )    
    # results = [r for r in results if r is not None]

    results = []
    for params in param_grid:
        r = run_model(params)
        if r is not None:
            results.append(r)


    if not results:
        return jsonify({
            "success": False,
            "message": "Tidak ada model yang lolos uji white noise (p > 0.05)"
        })

    df = pd.DataFrame(results)
    # df = df.sort_values(["MAPE", "RMSE", "AIC"])
    df = df.sort_values(["MAPE", "R2", "RMSE"], ascending=[True, False, True])


    return jsonify({
        "success": True,
        "total_model": len(df),
        "models": df.to_dict(orient="records"),
        "train": GLOBAL_TRAIN["harga"].tolist(),
        "test": GLOBAL_TEST["harga"].tolist(),
        "best_s": GLOBAL_BEST_S
    })




















# =========================
# CEK GLOBAL (OPTIONAL)
# =========================
@app.route("/cek-global")
def cek_global():

    return jsonify({
        "filename": GLOBAL_FILENAME,
        "best_s": GLOBAL_BEST_S,
        "train_len": None if GLOBAL_TRAIN is None else len(GLOBAL_TRAIN)
    })









# =====================================
# =====================================
# =====================================
@app.route("/sarima-forward-pred", methods=["POST"])
def sarima_forward_pred():

    global GLOBAL_TRAIN, GLOBAL_TEST, GLOBAL_PRED

    if GLOBAL_TRAIN is None or GLOBAL_TEST is None or GLOBAL_PRED is None:
        return jsonify({
            "success": False,
            "message": "Split data dulu"
        })

    data = request.json
    models = data.get("models", [])

    if not models:
        return jsonify({
            "success": False,
            "message": "Model kosong"
        })

    train = GLOBAL_TRAIN["harga"].values.astype(float)
    test = GLOBAL_TEST["harga"].values.astype(float)
    pred_actual = GLOBAL_PRED["harga"].values.astype(float)

    full_data = np.concatenate([train, test])

    def run_forward(m):

        p, d, q = m["order"]
        P, D, Q, s = m["seasonal"]

        try:
            model = SARIMAX(
                full_data,
                order=(p, d, q),
                seasonal_order=(P, D, Q, s),
                enforce_stationarity=False,
                enforce_invertibility=False
            )

            fit = model.fit(disp=False)

            forecast = fit.forecast(steps=len(pred_actual))

            mask = pred_actual != 0
            if np.sum(mask) == 0:
                return None

            mape_pred = np.mean(
                np.abs((pred_actual[mask] - forecast[mask]) / pred_actual[mask])
            ) * 100
            
            # corr_pred = np.corrcoef(pred_actual, forecast)[0, 1]
            # r2_pred = r2_score(pred_actual, forecast)


            # CORRELATION SAFE
            if np.std(pred_actual) == 0 or np.std(forecast) == 0:
                corr_pred = 0
            else:
                corr_pred = np.corrcoef(pred_actual, forecast)[0, 1]
                if np.isnan(corr_pred):
                    corr_pred = 0

            # R2 SAFE
            try:
                r2_pred = r2_score(pred_actual, forecast)
                if np.isnan(r2_pred):
                    r2_pred = 0
            except:
                r2_pred = 0


            


            rmse_pred = np.sqrt(
                mean_squared_error(pred_actual, forecast)
            )

            mse_pred = mean_squared_error(pred_actual, forecast)
            # rmse_pred = np.sqrt(mse_pred)

            return {
                "order": m["order"],
                "seasonal": m["seasonal"],
                "AIC": m.get("AIC"),
                "MAPE_TEST": m.get("MAPE"),
                "MSE_TEST": m.get("MSE"),
                "RMSE_TEST": m.get("RMSE"),
                "WHITE_NOISE": m.get("white_noise"),
                "MAPE_PRED": round(float(mape_pred), 2),
                "MSE_PRED": round(float(mse_pred), 2),
                "RMSE_PRED": round(float(rmse_pred), 2),
                "R2_PRED": round(float(r2_pred), 4),
                "CORR_PRED": round(float(corr_pred), 4),
                "forecast": forecast.tolist()
            }


        except:
            return None

    # results = Parallel(n_jobs=-1)(
    # results = Parallel(n_jobs=1)( # Hsoting
    #     delayed(run_forward)(m) for m in models
    # )
    # results = [r for r in results if r is not None]

    results = []
    for m in models:
        r = run_forward(m)
        if r is not None:
            results.append(r)

    if not results:
        return jsonify({
            "success": False,
            "message": "Semua model gagal saat forward prediction"
        })

    df = pd.DataFrame(results)
    # df = df.sort_values(["MAPE_PRED", "RMSE_PRED"])
    # df = df.sort_values(["MAPE_PRED", "R2_PRED"], ascending=[True, False])


    return jsonify({
        "success": True,
        "models": df.to_dict(orient="records"),
        "actual_pred": GLOBAL_PRED["harga"].tolist()
    })


# ==========================================================
# ==========================================================

@app.route("/proses")
def proses():
    return render_template("proses.html")

from statsmodels.tsa.stattools import adfuller, kpss

@app.route("/uji-stasioner", methods=["POST"])
def uji_stasioner():

    global GLOBAL_TRAIN

    if GLOBAL_TRAIN is None:
        return jsonify({
            "success": False,
            "message": "Split data dulu"
        })

    try:
        series = GLOBAL_TRAIN["harga"].astype(float)

        # ======================
        # UJI AWAL
        # ======================
        adf = adfuller(series)
        kpss_result = kpss(series, regression='c')

        adf_stat = float(adf[0])
        adf_p = float(adf[1])
        kpss_stat = float(kpss_result[0])
        kpss_p = float(kpss_result[1])

        adf_result = "Stasioner" if adf_p < 0.05 else "Tidak Stasioner"
        kpss_result_text = "Stasioner" if kpss_p >= 0.05 else "Tidak Stasioner"

        # Kalau salah satu tidak stasioner → diff
        need_diff = (adf_p >= 0.05) or (kpss_p < 0.05)

        response = {
            "success": True,
            "adf_stat": round(adf_stat, 4),
            "adf_pvalue": round(adf_p, 6),
            "adf_result": adf_result,
            "kpss_stat": round(kpss_stat, 4),
            "kpss_pvalue": round(kpss_p, 6),
            "kpss_result": kpss_result_text,
            "final_result": "Tidak Stasioner" if need_diff else "Stasioner",
            "differencing_done": False
        }

        # ======================
        # DIFFERENCING 1x
        # ======================
        if need_diff:

            diff_series = series.diff().dropna()

            response["differencing_done"] = True
            response["d_used"] = 1
            response["diff_series"] = diff_series.tolist()

            adf2 = adfuller(diff_series)
            kpss2 = kpss(diff_series, regression='c')

            response["differencing_done"] = True
            response["after_diff"] = {
                "adf_stat": round(float(adf2[0]), 4),
                "adf_pvalue": round(float(adf2[1]), 6),
                "kpss_stat": round(float(kpss2[0]), 4),
                "kpss_pvalue": round(float(kpss2[1]), 6),
                "adf_result": "Stasioner" if float(adf2[1]) < 0.05 else "Tidak Stasioner",
                "kpss_result": "Stasioner" if float(kpss2[1]) >= 0.05 else "Tidak Stasioner"
            }

        return jsonify(response)

    except Exception as e:
        return jsonify({
            "success": False,
            "message": str(e)
        })


# ==========================
# ==========================
# ==========================
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import io
import base64

@app.route("/plot-acf-pacf", methods=["POST"])
def plot_acf_pacf():

    global GLOBAL_TRAIN

    if GLOBAL_TRAIN is None:
        return jsonify({
            "success": False,
            "message": "Split data dulu"
        })

    try:
        series = GLOBAL_TRAIN["harga"].astype(float)

        # ==========================
        # CEK PERLU DIFFERENCING?
        # ==========================
        adf_p = adfuller(series)[1]
        kpss_p = kpss(series, regression='c')[1]

        need_diff = (adf_p >= 0.05) or (kpss_p < 0.05)

        if need_diff:
            data_used = series.diff().dropna()
            d_used = 1
        else:
            data_used = series
            d_used = 0

        # ==========================
        # HITUNG LAG AMAN
        # ==========================
        n = len(data_used)

        if n < 5:
            return jsonify({
                "success": False,
                "message": "Data terlalu sedikit untuk ACF/PACF"
            })

        max_lags = min(20, int(n * 0.5) - 1)

        # ==========================
        # PLOT ACF
        # ==========================
        fig1, ax1 = plt.subplots()
        plot_acf(data_used, lags=max_lags, ax=ax1)
        buf1 = io.BytesIO()
        fig1.savefig(buf1, format="png")
        plt.close(fig1)
        buf1.seek(0)
        acf_base64 = base64.b64encode(buf1.read()).decode("utf-8")

        # ==========================
        # PLOT PACF
        # ==========================
        fig2, ax2 = plt.subplots()
        plot_pacf(data_used, lags=max_lags, ax=ax2, method='ywm')
        buf2 = io.BytesIO()
        fig2.savefig(buf2, format="png")
        plt.close(fig2)
        buf2.seek(0)
        pacf_base64 = base64.b64encode(buf2.read()).decode("utf-8")

        return jsonify({
            "success": True,
            "acf_plot": acf_base64,
            "pacf_plot": pacf_base64,
            "d_used": int(d_used)
        })

    except Exception as e:
        return jsonify({
            "success": False,
            "message": str(e)
        })

if __name__ == "__main__":
    # app.run(debug=True)
    port = int(os.environ.get("PORT", 10000))      # Untuk Hosting
    app.run(host="0.0.0.0", port=port)             # Untuk Hosting
