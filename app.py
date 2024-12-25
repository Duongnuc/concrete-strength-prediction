import pandas as pd
import streamlit as st
import joblib
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

# Đọc dữ liệu phát thải CO2 và giá thành
phatthai_df = pd.read_csv('phatthai.csv')  # Sửa đường dẫn
giathanh_df = pd.read_csv('giathanh.csv')  # Sửa đường dẫn

# Chuyển đổi dòng phát thải và giá thành thành dictionary
phatthai = phatthai_df.iloc[0].to_dict()  # Lấy dòng phát thải CO2
giathanh = giathanh_df.iloc[0].to_dict()  # Lấy dòng giá thành

# Đọc khối lượng riêng từ file CSV
klr_df = pd.read_csv('klr.csv')  # Sửa đường dẫn
klr = klr_df.iloc[0].to_dict()

# Load mô hình đã đào tạo
model = joblib.load('lightgbm_model.pkl')  # Sửa đường dẫn

# Giới hạn hàm lượng các loại vật liệu
LIMITS = {
    'cement': (129, 486),
    'slag': (0, 350),
    'ash': (0, 358),
    'water': (105, 240),
    'superplastic': (0, 23),
    'coarseagg': (708, 1232),
    'fineagg': (555, 971),
}

# Tên hiển thị tương ứng
DISPLAY_NAMES = {
    'cement': "Hàm lượng xi măng (kg/m³)",
    'slag': "Hàm lượng xỉ (kg/m³)",
    'ash': "Hàm lượng tro bay (kg/m³)",
    'water': "Hàm lượng nước (kg/m³)",
    'superplastic': "Hàm lượng siêu dẻo (kg/m³)",
    'coarseagg': "Hàm lượng cốt liệu lớn (kg/m³)",
    'fineagg': "Hàm lượng cốt liệu nhỏ (kg/m³)",
}

# Các hàm tính toán như trước
def kiem_tra_cap_phoi(quy_doi_materials):
    for mat, (min_val, max_val) in LIMITS.items():
        if not (min_val <= quy_doi_materials[mat] <= max_val):
            return False, f"{DISPLAY_NAMES[mat]} không nằm trong khoảng [{min_val}, {max_val}]"
    slag_ash_ratio = (quy_doi_materials['slag'] + quy_doi_materials['ash']) / quy_doi_materials['cement']
    if not (0.3 <= slag_ash_ratio <= 0.6):
        return False, f"Tỷ lệ (Xỉ + Tro bay) / Xi măng = {slag_ash_ratio:.2f} không nằm trong khoảng [0.3, 0.6]"
    return True, "Cấp phối phù hợp"

def quy_doi_ve_1m3(materials, klr):
    total_volume = sum(materials[mat] / klr[mat] for mat in materials.keys())
    he_so_quy_doi = 1000 / total_volume
    return {mat: materials[mat] * he_so_quy_doi for mat in materials.keys()}

def du_doan_cuong_do(quy_doi_materials, tuoi_list):
    predictions = []
    for tuoi in tuoi_list:
        inputs = [
            quy_doi_materials['cement'],
            quy_doi_materials['slag'],
            quy_doi_materials['ash'],
            quy_doi_materials['water'],
            quy_doi_materials['superplastic'],
            quy_doi_materials['coarseagg'],
            quy_doi_materials['fineagg'],
            tuoi
        ]
        prediction = model.predict([inputs])[0]
        predictions.append(prediction)
    return predictions

def tinh_gia_thanh_va_phat_thai(quy_doi_materials, giathanh, phatthai, predictions):
    tong_gia_thanh = 0
    tong_phat_thai = 0
    for mat in quy_doi_materials:
        tong_gia_thanh += quy_doi_materials[mat] * giathanh[mat]
        tong_phat_thai += quy_doi_materials[mat] * phatthai[mat]
    gia_thanh_mpa = tong_gia_thanh / predictions[-1]
    phat_thai_mpa = tong_phat_thai / predictions[-1]
    return tong_gia_thanh, tong_phat_thai, gia_thanh_mpa, phat_thai_mpa

def ve_duong_xu_huong(tuoi_list, predictions):
    def logistic_growth(x, a, b, c):
        return a / (1 + np.exp(-b * (x - c)))
    params, _ = curve_fit(logistic_growth, tuoi_list, predictions, maxfev=10000, bounds=(0, [np.inf, np.inf, np.inf]))
    tuoi_fit = np.linspace(min(tuoi_list), max(tuoi_list), 100)
    predictions_fit = logistic_growth(tuoi_fit, *params)
    plt.figure(figsize=(8, 5))
    plt.scatter(tuoi_list, predictions, color='blue', label="Cường độ thực tế")
    plt.plot(tuoi_fit, predictions_fit, color='red', linestyle='--', label="Đường xu hướng")
    plt.title("Cường độ bê tông theo ngày tuổi")
    plt.xlabel("Ngày tuổi")
    plt.ylabel("Cường độ (MPa)")
    plt.grid(True)
    plt.legend()
    st.pyplot(plt)

# Giao diện Streamlit
st.title("Dự đoán cường độ bê tông và tính toán kinh tế, phát thải")

# Nhập liệu hàm lượng vật liệu
st.header("Nhập liệu hàm lượng vật liệu (kg)")
materials = {
    'cement': st.number_input("Xi măng (kg):", value=250.0),
    'slag': st.number_input("Xỉ (kg):", value=100.0),
    'ash': st.number_input("Tro bay (kg):", value=150.0),
    'water': st.number_input("Nước (kg):", value=150.0),
    'superplastic': st.number_input("Siêu dẻo (kg):", value=5.0),
    'coarseagg': st.number_input("Cốt liệu lớn (kg):", value=900.0),
    'fineagg': st.number_input("Cốt liệu nhỏ (kg):", value=800.0),
}

if st.button("Quy đổi về 1m³, dự đoán và tính toán"):
    quy_doi_materials = quy_doi_ve_1m3(materials, klr)
    is_valid, message = kiem_tra_cap_phoi(quy_doi_materials)
    st.subheader("Kết quả kiểm tra cấp phối:")
    st.write(message)

    if is_valid:
        tuoi_list = [3, 7, 28, 91]
        predictions = du_doan_cuong_do(quy_doi_materials, tuoi_list)
        tong_gia_thanh, tong_phat_thai, gia_thanh_mpa, phat_thai_mpa = tinh_gia_thanh_va_phat_thai(
            quy_doi_materials, giathanh, phatthai, predictions)

        st.subheader("Cấp phối đã quy đổi về 1m³:")
        for mat, value in quy_doi_materials.items():
            st.write(f"{DISPLAY_NAMES[mat]}: {value:.2f} kg")

        st.subheader("Kết quả kinh tế và phát thải:")
        st.markdown(f'Tổng giá thành: <b style="color:red;">{tong_gia_thanh:.2f} VNĐ</b>', unsafe_allow_html=True)
        st.markdown(f'Lượng phát thải: <b style="color:red;">{tong_phat_thai:.2f} kg</b>', unsafe_allow_html=True)
        st.markdown(f'Giá thành/MPa: <b style="color:red;">{gia_thanh_mpa:.2f} VNĐ/MPa</b>', unsafe_allow_html=True)
        st.markdown(f'CO2/MPa: <b style="color:red;">{phat_thai_mpa:.2f} kg CO2/MPa</b>', unsafe_allow_html=True)

        ve_duong_xu_huong(tuoi_list, predictions)
    else:
        st.warning("Không thể tiến hành dự đoán do cấp phối không phù hợp.")
