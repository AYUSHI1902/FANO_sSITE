import numpy as np
import streamlit as st
import matplotlib.pyplot as plt
import pandas as pd
from scipy.integrate import simpson
from scipy.optimize import curve_fit

st.set_page_config(page_title="Raman-Fano line-shape", layout="centered")

st.title("Raman-Fano line-shape plot, gamma")
st.write("Enter parameters and upload file")

# -------- PARAMETER ----------
st.subheader("Enter Parameters")

col1, col2, col3 = st.columns(3)

with col1:
    a = st.number_input("a ", value=0.5)

with col2:
    A = st.number_input("A value", value=171400.0)

with col3:
    B = st.number_input("B value", value=100000.0)

# -------- FILE UPLOAD ----------
st.subheader("Upload Raman File")

uploaded_file = st.file_uploader(
    "Upload CSV/TXT/XLSX file (two columns)",
    type=["csv", "txt", "xlsx"]
)

# -------- LOAD DATA ----------
with st.spinner("Fitting Raman spectrum... Please wait"):

    if uploaded_file is not None:

        if uploaded_file.name.endswith(".xlsx"):
            data = pd.read_excel(uploaded_file, header=None)
        else:
            data = pd.read_csv(uploaded_file, sep=None, engine="python", header=None)

        st.write("Uploaded file:", uploaded_file.name)

        omega_exp = data.iloc[:, 0].values
        I_exp = data.iloc[:, 1].values

        # fitting range
        mask = (omega_exp >= 440) & (omega_exp <= 560)
        omega_exp = omega_exp[mask]
        I_exp = I_exp[mask]

        # -------- SORT DATA ----------
        idx = np.argsort(omega_exp)
        omega_exp = omega_exp[idx]
        I_exp = I_exp[idx]

        omega_peak = omega_exp[np.argmax(I_exp)]
        st.write("peak", omega_peak)
        # k grid
        k = np.linspace(0, 1, 2000)

        # -------- omega(k) ----------
        omega_k_vals = np.sqrt(A + B*np.cos(np.pi*k/2))

        # -------- FANO MODEL ----------
        def fano_model(omega, q, L, Gamma, shift, C, m, c):

            omega2D = omega[:, None] + shift

            eps = (omega2D - omega_k_vals) / (Gamma/2)

            integrand = np.exp(-(k**2 * L**2)/(4*a**2)) * ((q+eps)**2/(1+eps**2))

            integrand *= (2*np.pi*k)

            I = np.trapezoid(integrand, k, axis=1)

            background = m*omega + c

            return C*I + background

        popt, _ = curve_fit(
            fano_model,
            omega_exp,
            I_exp,
            p0=[2,5,6,0,100,0,10],
            bounds=([-50,0,1,-10,0,-10,-500],
                    [50,50,30,10,1e6,10,500]),
            maxfev=40000
        )

        q, L, Gamma, shift, C, m, c = popt

        fit = fano_model(omega_exp, *popt)

        st.subheader("Final Fitted Values")
        st.write("q =", round(q,3))
        st.write("L =", round(L,3), "nm")
        st.write("Gamma =", round(Gamma,3))

        # plot
        fig, ax = plt.subplots(figsize=(6,5))
        half_max = np.max(fit)/2
        ax.hlines(half_max, peak_fit - Gamma/2, peak_fit + Gamma/2, color='black', linewidth=2)
        ax.plot(omega_exp, I_exp, 'r.', label="Experimental")
        ax.plot(omega_exp, fit, 'b-', label="Fitted")
        ax.legend()
        ax.grid()

        st.pyplot(fig)

    else:
        st.warning("Upload file first")
