import numpy as np
import streamlit as st
import matplotlib.pyplot as plt
import pandas as pd
from scipy.optimize import curve_fit

from io import BytesIO
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Image
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.lib.pagesizes import letter

st.set_page_config(page_title="Raman-Fano line-shape", layout="centered")

st.title("Raman-Fano line-shape plot")
st.write("Enter parameters and upload file")

# -------- PARAMETERS ----------
st.subheader("Enter Parameters")


col1, col2, col3 = st.columns(3)

with col1:
    a = st.number_input("a", value=0.5)

with col2:
    A = st.number_input("A value",value=171400.0)

with col3:
    B = st.number_input("B value", value=100000.0)

# -------- MODE ----------
st.subheader("Select Mode")

mode = st.selectbox(
    "Choose model type",
    ["Fano and Confinement", "Confinement", "Fano"]
)

# -------- FILE ----------
st.subheader("Upload Raman File")

uploaded_file = st.file_uploader(
    "Upload CSV//TXT/XLSX file (two columns)",
    type=["csv", "txt", "xlsx"]
)

def compute_r2(omega_exp, I_exp, fit, infodict):
       mask_2 = (omega_exp >= 510) & (omega_exp <= 530)

       I_exp_sub = I_exp[mask_2]
       fit_sub = fit[mask_2]

    
       ss_res = np.sum((I_exp_sub - fit_sub) ** 2)
       ss_tot = np.sum((I_exp_sub - np.mean(I_exp_sub)) ** 2)

       r2 = 1 - ss_res / ss_tot if ss_tot != 0 else np.nan


       nfev = infodict.get("nfev", "Not available for this mode")
       return r2, nfev
# -------- PROCESS ----------
with st.spinner("Fitting Raman spectrum... Please wait"):

    if uploaded_file is not None:

        # LOAD DATA
        if uploaded_file.name.endswith(".xlsx"):
            data = pd.read_excel(uploaded_file, header=None)
        else:
            data = pd.read_csv(uploaded_file, sep=None, engine="python", header=None)

        omega_exp = data.iloc[:, 0].values
        I_exp = data.iloc[:, 1].values

        # FILTER RANGE
        mask = (omega_exp >= 450) & (omega_exp <= 550)
        omega_exp = omega_exp[mask]
        I_exp = I_exp[mask]

        # SORT
        idx = np.argsort(omega_exp)
        omega_exp = omega_exp[idx]
        I_exp = I_exp[idx]

        st.write("Peak:", omega_exp[np.argmax(I_exp)])

        # k GRID
        k = np.linspace(0, 1, 2000)
        omega_k_vals = np.sqrt(A + B * np.cos(np.pi * k / 2))

        # -------- MODEL ----------
        def fano_model(omega, q, L, Gamma, shift, C, m, c):
            omega2D = omega[:, None] + shift
            eps = (omega2D - omega_k_vals) / (Gamma / 2)

            integrand = np.exp(-(k**2 * L**2) / (4 * a**2)) * ((q + eps)**2 / (1 + eps**2))
            integrand *= (2 * np.pi * k)

            I = np.trapezoid(integrand, k, axis=1)
            background = m * omega + c

            return C * I + background
        infodict = {"nfev": None}
        # -------- FITTING ----------
        if mode == "Fano and Confinement":

            popt, pcov, infodict, mesg, ier = curve_fit(
                fano_model,
                omega_exp,
                I_exp,
                p0=[0, 5, 6, 0, 100, 0, 10],
                bounds=([-10, 0, 1, -10, 0, -10, -500],
                        [10, 50, 30, 10, 1e6, 10, 500]),
                maxfev=4000,
                full_output=True 
            )

            q, L, Gamma, shift, C, m, c = popt
           # st.write("sample is having Fano & confinement effect")
        elif mode == "Confinement":

            def model_fixed_q(omega, L, Gamma, shift, C, m, c):
                return fano_model(omega, 1000, L, Gamma, shift, C, m, c)

            popt, _ = curve_fit(
                model_fixed_q,
                omega_exp,
                I_exp,
                p0=[1, 6, 0, 100, 0, 10],
                bounds=([0, 1, -10, 0, -10, -500],
                        [50, 30, 10, 1e6, 10, 500]),
                maxfev=4000
            )

            L, Gamma, shift, C, m, c = popt
            
            q = 1000
            #st.write("sample is having Confinement effect")
        elif mode == "Fano":

            def model_fixed_L(omega, q, Gamma, shift, C, m, c):
                return fano_model(omega, q, 1000, Gamma, shift, C, m, c)

            popt, _ = curve_fit(
                model_fixed_L,
                omega_exp,
                I_exp,
                p0=[0, 6, 0, 100, 0, 10],
                bounds=([-10, 1, -10, 0, -10, -500],
                        [10, 30, 10, 1e6, 10, 500]),
                maxfev=4000
            )

            q, Gamma, shift, C, m, c = popt
            L = 1000
            #st.write("sample is having Fano effect")
        # -------- FINAL FIT ----------
        fit = fano_model(omega_exp, q, L, Gamma, shift, C, m, c)

        r2, nfev = compute_r2(omega_exp, I_exp, fit, infodict)

     
       #print("R² (480–510):", r2)

        
        # Manual R²
        #ss_res = np.sum((I_exp - fit) ** 2)
       # ss_tot = np.sum((I_exp - np.mean(I_exp)) ** 2)
        #r2 = 1 - ss_res / ss_tot if ss_tot != 0 else np.nan
        #nfev = infodict.get("nfev", "Not available for this mode")

        # -------- OUTPUT ----------
        st.subheader("Final Fitted Values")
        st.write("Mode:", mode)
        #st.write("q =", round(q, 3))
       # st.write("L =", round(L, 3), "nm")
        #st.write("Gamma =", round(Gamma, 3))
        #st.write("Function evaluations:", nfev)
        #st.write("R² (480–510)=", round(r2, 5))
        # -------- RESULT ----------
                  # -------- PLOT ----------
        fig, ax = plt.subplots(figsize=(6, 5))
        
        ax.plot(omega_exp, I_exp, 'r.', label="Experimental")
        ax.plot(omega_exp, fit, 'b-', label="Fitted")
        
        ax.set_xlabel("Raman Shift (cm⁻¹)", fontsize=12)
        ax.set_ylabel("Intensity (a.u.)", fontsize=12)
        
        ax.legend()
        ax.grid()
        
        st.pyplot(fig)
        
        # -------- RESULT TEXT ----------
        
        if mode == "Fano":
        
            result_display = f"""
            Selected Mode : Fano
        
            This plot is showing Fano effect.
        
            q value = {round(q,3)}
        
           
            """
        
            pdf_result = f"""
            Sample is showing Fano Effect
            
            Selected Mode : Fano 
            
            q = {round(q,3)}
        
           
            """
        
        elif mode == "Confinement":
        
            result_display = f"""
            Selected Mode : Confinement
        
            This plot is showing Confinement effect.
        
            L value = {round(L,3)} nm
        
         
            """
        
            pdf_result = f"""
            Sample is showing Confinement Effect
            
            Selected Mode :  Confinement
            
            L = {round(L,3)} nm
         
            """
        
        elif mode == "Fano and Confinement":
        
            result_display = f"""
            Selected Mode : Fano and Confinement
        
            This plot is showing both Fano and Confinement effects.
        
            q value = {round(q,3)}
        
            L value = {round(L,3)} nm
        
            
            """
        
            pdf_result = f"""
            Sample is showing Fano and Confinement Effect
            
            Selected Mode : Fano and Confinement
            
            q = {round(q,3)}
        
            L = {round(L,3)} nm
        
          
            """
        
            # -------- CREATE PDF ----------
        
        img_buffer = BytesIO()
        
        fig.savefig(img_buffer, format="png", dpi=300, bbox_inches='tight')
        
        img_buffer.seek(0)
        
        pdf_buffer = BytesIO()
        
        doc = SimpleDocTemplate(pdf_buffer, pagesize=letter)
        
        styles = getSampleStyleSheet()
        
        elements = []
        
        # TITLE
        elements.append(
            Paragraph("<b>RAMAN ANALYSIS REPORT</b>", styles['Title'])
        )
        
        elements.append(Spacer(1, 12))
        
        # FILE NAME
        elements.append(
            Paragraph(
                f"<b>Uploaded File:</b> {uploaded_file.name}",
                styles['BodyText']
            )
        )
        
        elements.append(Spacer(1, 12))
        
        # RESULT
        elements.append(
            Paragraph(pdf_result.replace("\n", "<br/>"),
            styles['BodyText'])
        )
        
        elements.append(Spacer(1, 20))
        
        # IMAGE
        plot_image = Image(img_buffer, width=400, height=300)
        
        elements.append(plot_image)
        
        # BUILD PDF
        doc.build(elements)
        
        pdf_buffer.seek(0)
        
        # -------- RESULT BLOCK ----------
        
        st.subheader("Result")
        
        st.info(result_display)
        
        st.download_button(
            label="Download Report",
            data=pdf_buffer,
            file_name=f"{uploaded_file.name}_Raman_Report.pdf",
            mime="application/pdf"
        )


       

    else:
        st.warning("Upload file first")
