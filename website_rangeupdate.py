import numpy as np
import streamlit as st
import matplotlib.pyplot as plt
import pandas as pd
from scipy.integrate import simpson
from scipy.optimize import curve_fit


st.set_page_config(page_title="Raman-Fano line-shape", layout="centered")

st.title("Raman-Fano line-shape plot")
st.write("Enter parameters and upload file")
# -------- PARAMETER ----------
st.subheader("Enter Parameters")

col1, col2, col3 = st.columns(3)

with col1:
    a = st.number_input("a (lattice constant)", value=0.5)

with col2:
    A = st.number_input("A value", value=171400.0)

with col3:
    B = st.number_input("B value", value=100000.0)

# -------- FILE UPLOAD ----------
st.subheader("Upload Raman File")

uploaded_file = st.file_uploader(
    "Upload CSV/TXT file (two columns)",
    type=["csv","txt"]
)


# -------- LOAD DATA ----------


if st.button("Plot Graph"):

    if uploaded_file is not None:

        data = pd.read_csv(uploaded_file, sep=None, engine="python", header=None)
        omega_exp = data.iloc[:,0].values
        I_exp = data.iloc[:,1].values

        mask = (omega_exp >= 480) & (omega_exp <= 560)
        omega_exp = omega_exp[mask]
        I_exp = I_exp[mask]
        k = np.linspace(0,1,250)
        omega_k_vals = np.sqrt(A + B*np.cos(np.pi*k/2))

        # -------- FAST MODEL ----------
        def fano_model(omega, q, L, Gamma, C, p,m, c, shift):

         # vectorized calculation
         omega2D = omega[:,None] + shift
         eps = (omega2D - omega_k_vals)/(Gamma/2)

         integrand = np.exp(-(k**2 * L**2)/(4*a**2)) * ((q+eps)**2/(1+eps**2))
         integrand = integrand*(2*np.pi*k)

         I = simpson(integrand, k, axis=1)

         background = p*omega**2 + m*omega + c
         return C*I + background


     # -------- FITTING ----------
        with st.spinner("Fitting running... please wait"):
         popt, _ = curve_fit(
         fano_model,
          omega_exp,
         I_exp,
         p0=[4,4,8,200,0,0,10,0],
         bounds=([0,0,1,0,-1,-50,-200,-20],[20,50,60,1e6,1,50,200,20]),
          maxfev=20000
         )


        q,L,Gamma,C,p,m,c,shift = popt

        st.subheader("Final Fitted Values")
        st.write("q =",round(q,3))
        st.write("L =",round(L,3),"nm")
        st.write("Gamma =",round(Gamma,3))
        #print("Shift =",round(shift,3))
        st.write("omega_k_val =",omega_k_vals)


# plot
        fig, ax = plt.subplots(figsize=(6,5))
        ax.plot(omega_exp,I_exp,'r.',label="Experimental")
        ax.plot(omega_exp,fano_model(omega_exp,*popt),'b-',label="Fitted")
        ax.legend()
        ax.grid()
        st.pyplot(fig)

    else:
        st.warning("Upload file first")
