import os
import joblib
import traceback
import tempfile
from datetime import datetime
import pandas as pd
import numpy as np
import gradio as gr
import matplotlib.pyplot as plt

# ----------------- 1. Cấu hình -----------------
RISK_MODEL_PATH = "credit_risk_model.pkl"
SCORE_MODEL_PATH = "credit_score_model.pkl"
FEATURE_NAMES_PATH = "feature_names.pkl"

# ----------------- 2. Load models -----------------
if not os.path.exists(RISK_MODEL_PATH):
    raise FileNotFoundError(f"Không tìm thấy model risk tại '{RISK_MODEL_PATH}'.")
risk_model = joblib.load(RISK_MODEL_PATH)

if not os.path.exists(SCORE_MODEL_PATH):
    raise FileNotFoundError(f"Không tìm thấy model score tại '{SCORE_MODEL_PATH}'.")
score_data = joblib.load(SCORE_MODEL_PATH)
score_model = score_data['model']
score_scaler = score_data['scaler']

feature_names_from_file = None
if os.path.exists(FEATURE_NAMES_PATH):
    try:
        feature_names_from_file = joblib.load(FEATURE_NAMES_PATH)
    except Exception as e:
        print("⚠️ Không thể load feature names từ file:", e)

base_cols = ['monthly_inc','age','debt_ratio','open_credit','real_estate',
             'late_30_59','late_60_89','late_90','dependents','rev_util']
eps = 1e-8

# ----------------- 3. Hàm extract expected features -----------------
def _extract_expected_features(model, sample_columns=None):
    if feature_names_from_file is not None:
        return list(feature_names_from_file)
    if hasattr(model, "feature_names_in_"):
        return list(model.feature_names_in_)
    return None

# ----------------- 4. Phân tích insight và khuyến nghị -----------------
def generate_insights(df_out):
    try:
        insights = []

        high_risk = df_out[df_out['prediction'] == 1]
        low_risk = df_out[df_out['prediction'] == 0]

        # ======= HIGH RISK GROUP =======
        if not high_risk.empty:
            avg_score = high_risk['credit_score'].mean()
            top_features = high_risk[['monthly_inc', 'debt_ratio', 'late_30_59', 'late_60_89', 'late_90']].mean().sort_values(ascending=False)

            insights.append("### Nhóm High Risk (Rủi ro cao):")
            insights.append(f"- Số lượng khách hàng: **{len(high_risk)}**")
            insights.append(f"- Trung bình điểm tín dụng: **{avg_score:.2f}**")

            insights.append("- 🔍 **Các yếu tố rủi ro chính:**")
            for feat, val in top_features.items():
                desc = {
                    'monthly_inc': "Thu nhập hàng tháng cao bất thường → có thể là dữ liệu sai lệch hoặc thu nhập không ổn định.",
                    'debt_ratio': "Tỷ lệ nợ cao → gánh nặng tài chính lớn.",
                    'late_30_59': "Trễ hạn 30-59 ngày → dấu hiệu mất khả năng thanh toán nhẹ.",
                    'late_60_89': "Trễ hạn 60-89 ngày → tình trạng nợ trung bình.",
                    'late_90': "Trễ hạn ≥90 ngày → rủi ro tín dụng cực cao."
                }
                explanation = desc.get(feat, "")
                insights.append(f"  - `{feat}` trung bình: **{val:.2f}** → {explanation}")

            insights.append("-  **Khuyến nghị hành động:**")
            insights.append("  - Liên hệ nhóm này để xác minh thu nhập và tình trạng công việc.")
            insights.append("  - Hướng dẫn lập kế hoạch trả nợ và theo dõi sát lịch sử thanh toán.")
            insights.append("  - Tạm ngừng cấp thêm tín dụng cho đến khi lịch sử tín dụng cải thiện.\n")

        # ======= LOW RISK GROUP =======
        if not low_risk.empty:
            avg_score = low_risk['credit_score'].mean()
            top_features = low_risk[['monthly_inc', 'debt_ratio', 'late_30_59', 'late_60_89', 'late_90']].mean().sort_values(ascending=False)

            insights.append("### 🟢 Nhóm Low Risk (Rủi ro thấp):")
            insights.append(f"- Số lượng khách hàng: **{len(low_risk)}**")
            insights.append(f"- Trung bình điểm tín dụng: **{avg_score:.2f}**")

            num_good_payers = (low_risk[['late_30_59', 'late_60_89', 'late_90']].sum(axis=1) == 0).sum()
            insights.append(f"- **{num_good_payers} khách hàng (chiếm {num_good_payers/len(low_risk)*100:.1f}%) không có lịch sử trễ hạn.**")

            insights.append("- 🌱 **Khuyến nghị phát triển:**")
            insights.append("  - Xem xét tăng hạn mức tín dụng hoặc giới thiệu sản phẩm tài chính mới.")
            insights.append("  - Tri ân nhóm khách hàng này bằng các chương trình thưởng/thẻ tín dụng ưu đãi.")
            insights.append("  - Có thể xây dựng mô hình khách hàng lý tưởng dựa trên đặc điểm nhóm này.\n")

        # Nếu không có dữ liệu hợp lệ
        if not insights:
            insights.append("⚠️ Không có dữ liệu phù hợp để tạo insight.")

        return "\n".join(insights)

    except Exception as e:
        return f"❌ Lỗi khi tạo insight: {str(e)}"

# ----------------- 5. Hàm xử lý file upload -----------------
def process_file(uploaded_file):
    try:
        if uploaded_file is None:
            return [None]*9 + ["❌ Vui lòng upload file CSV hoặc Excel."]

        fname = getattr(uploaded_file, "name", None) or str(uploaded_file)
        ext = fname.split(".")[-1].lower()
        if ext == "csv":
            df = pd.read_csv(uploaded_file.file if hasattr(uploaded_file, "file") else fname)
        elif ext in ("xls", "xlsx"):
            df = pd.read_excel(uploaded_file.file if hasattr(uploaded_file, "file") else fname)
        else:
            return [None]*9 + ["❌ Định dạng file không hỗ trợ."]

        if df.shape[0] == 0:
            return [None]*9 + ["❌ File rỗng."]

        original_df = df.copy()
        existing_output_cols = [c for c in ['prediction','prediction_label','prob_risk','credit_score'] if c in df.columns]
        overwritten = False
        if existing_output_cols:
            overwritten = True
            df = df.drop(columns=existing_output_cols, errors='ignore')

        for c in base_cols:
            if c not in df.columns:
                df[c] = 0
        for c in base_cols:
            df[c] = pd.to_numeric(df[c], errors='coerce').fillna(0)

        df['income_to_debt'] = df['monthly_inc'] / (df['debt_ratio'] + eps)
        df['total_late'] = df['late_30_59'] + df['late_60_89'] + df['late_90']
        df['age_per_credit'] = df['age'] / (df['open_credit'] + eps)
        df['real_estate_ratio'] = df['real_estate'] / (df['open_credit'] + eps)

        expected = _extract_expected_features(risk_model, sample_columns=df.columns.tolist())
        if expected:
            for c in expected:
                if c not in df.columns:
                    df[c] = 0
            df_model = df.reindex(columns=expected, fill_value=0)
        else:
            df_model = df.copy()

        risk_preds = risk_model.predict(df_model)
        try:
            proba = risk_model.predict_proba(df_model)
            prob_vals = proba[:,1] if proba.shape[1]>=2 else None
        except Exception:
            prob_vals = None

        X_scaled = score_scaler.transform(df[base_cols])
        score_preds = score_model.predict(X_scaled)

        df_out = original_df.copy()
        df_out['prediction'] = risk_preds
        df_out['prediction_label'] = df_out['prediction'].map({0:'Low risk',1:'High risk'})
        df_out['credit_score'] = score_preds.round(2)
        df_out['prob_risk'] = (prob_vals*100).round(2) if prob_vals is not None else np.nan

        # ---------- Bar Chart & Insight ----------
        fig_counts = plt.figure(figsize=(6,4))
        counts = df_out['prediction_label'].value_counts()
        ax = fig_counts.subplots()
        counts.plot.bar(ax=ax, color=['green','red'])
        ax.set_title("Số lượng theo loại rủi ro")
        ax.set_xlabel("Loại")
        ax.set_ylabel("Số bản ghi")
        ax.grid(axis='y', linestyle=':', linewidth=0.6)
        plt.tight_layout()
        high_count = counts.get('High risk', 0)
        low_count = counts.get('Low risk', 0)
        total_count = high_count + low_count
        high_ratio = (high_count / total_count * 100) if total_count > 0 else 0
        low_ratio = 100 - high_ratio

        counts_md = (
            f"📊 **Phân bố theo loại rủi ro:**\n"
            f"- 🔴 High risk: **{high_count}** bản ghi (**{high_ratio:.1f}%**)\n"
            f"- 🟢 Low risk: **{low_count}** bản ghi (**{low_ratio:.1f}%**)\n\n"
            f"🧠 **Phân tích:** Nhóm High risk chiếm tỷ lệ đáng kể trong tổng dữ liệu, cho thấy có nhiều khách hàng đang ở trạng thái tài chính không ổn định.\n"
            f"💡 **Khuyến nghị:**\n"
            f"- Tập trung rà soát và theo dõi kỹ lưỡng nhóm High risk.\n"
            f"- Áp dụng các chiến lược phòng ngừa như cảnh báo sớm, giới hạn tín dụng và tư vấn tài chính.\n"
            f"- Đối với nhóm Low risk, có thể xem xét nâng hạn mức, đề xuất sản phẩm tín dụng ưu đãi."
        )

        # ---------- Pie Chart & Insight ----------
        fig_pie = plt.figure(figsize=(5,5))
        ax2 = fig_pie.subplots()
        labels = counts.index.tolist()
        sizes = counts.values.tolist()
        if len(sizes)==1:
            labels += ['']; sizes += [0]
        explode = [0.02]*len(sizes)
        ax2.pie(sizes, labels=labels, autopct='%1.1f%%', startangle=90, explode=explode, colors=['green','red'])
        ax2.set_title("Tỷ lệ Low vs High risk")
        ax2.axis('equal')
        plt.tight_layout()
        high_pct = sizes[labels.index('High risk')]/sum(sizes)*100 if 'High risk' in labels else 0
        pie_md = (
            f"📊 Tỷ lệ High risk: **{high_pct:.1f}%** trong tổng số khách hàng.\n"
            f"💡 **Phân tích:** Tỷ lệ này phản ánh nhóm khách hàng có lịch sử tín dụng yếu, thường xuyên trễ hạn hoặc có tỷ lệ nợ cao.\n"
            f"🛠️ **Gợi ý biện pháp cải thiện:**\n"
            f"- Thiết lập hệ thống nhắc nợ tự động (qua email/SMS) trước hạn thanh toán.\n"
            f"- Cân nhắc cơ cấu lại khoản vay, giảm áp lực trả nợ hàng tháng.\n"
            f"- Tạm dừng cấp thêm tín dụng cho đến khi khách hàng cải thiện lịch sử trả nợ.\n"
            f"- Áp dụng chương trình ưu đãi 'Trả đúng – Tăng hạn mức' để khuyến khích hành vi tích cực."
        )
        # ---------- Histogram & Insight ----------
        fig_prob = None
        prob_md = "⚠️ Không có dữ liệu xác suất rủi ro."
        if prob_vals is not None:
            fig_prob = plt.figure(figsize=(8,4.5))
            ax3 = fig_prob.subplots()
            values = df_out['prob_risk'].dropna().values
            counts_hist, bins, _ = ax3.hist(values, bins=30, alpha=0.6, density=True, color='skyblue')
            bin_centers = 0.5*(bins[:-1]+bins[1:])
            smooth = np.convolve(counts_hist,np.ones(5)/5,mode='same')
            ax3.plot(bin_centers,smooth,linewidth=2,label='Smoothed density')
            mean_val = values.mean(); median_val = np.median(values)
            ax3.axvline(mean_val, linestyle='--', linewidth=1.5, color='red', label=f"Mean: {mean_val:.2f}%")
            ax3.axvline(median_val, linestyle=':', linewidth=1.5, color='blue', label=f"Median: {median_val:.2f}%")
            ax3.set_title("Phân bố xác suất rủi ro (%)")
            ax3.set_xlabel("Xác suất rủi ro (%)"); ax3.set_ylabel("Density")
            ax3.grid(True, linestyle=':', linewidth=0.6)
            ax3.legend()
            plt.tight_layout()
            prob_md = f"🔹 Trung bình xác suất rủi ro: **{mean_val:.2f}%**\n🔹 Trung vị: **{median_val:.2f}%**\n💡 Nhóm khách hàng có xác suất >75% cần kiểm tra kỹ và tư vấn giảm rủi ro."

        # ---------- Export CSV ----------
        tmpdir = tempfile.gettempdir()
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        out_path = os.path.join(tmpdir,f"predictions_output_{ts}.csv")
        df_out.to_csv(out_path,index=False,encoding="utf-8-sig")

        total = len(df_out)
        high_count = int((df_out['prediction']==1).sum())
        low_count = int((df_out['prediction']==0).sum())
        summary = f"**Tổng bản ghi:** {total}  \n**High risk:** {high_count}  \n**Low risk:** {low_count}"
        if existing_output_cols:
            summary += "  \n⚠️ File input có cột output trước đó đã bị ghi đè."

        insight_markdown = generate_insights(df_out)
        summary += "\n\n" + insight_markdown

        return df_out, fig_counts, counts_md, fig_pie, pie_md, fig_prob, prob_md, out_path, summary

    except Exception as e:
        traceback.print_exc()
        return [None]*9 + [f"❌ Lỗi xử lý file: {str(e)}"]

# ----------------- 6. Hàm dự đoán single input -----------------
def predict_single(monthly_inc, age, debt_ratio, open_credit, real_estate,
                   late_30_59, late_60_89, late_90, dependents, rev_util):
    try:
        data = {c:[monthly_inc, age, debt_ratio, open_credit, real_estate,
                   late_30_59, late_60_89, late_90, dependents, rev_util][i] for i,c in enumerate(base_cols)}
        df = pd.DataFrame(data,index=[0])

        df['income_to_debt'] = df['monthly_inc'] / (df['debt_ratio'] + eps)
        df['total_late'] = df['late_30_59'] + df['late_60_89'] + df['late_90']
        df['age_per_credit'] = df['age'] / (df['open_credit'] + eps)
        df['real_estate_ratio'] = df['real_estate'] / (df['open_credit'] + eps)

        expected = _extract_expected_features(risk_model, sample_columns=df.columns.tolist())
        if expected:
            for c in expected:
                if c not in df.columns: df[c]=0
            df_model = df.reindex(columns=expected,fill_value=0)
        else:
            df_model = df.copy()

        risk_pred = risk_model.predict(df_model)[0]
        try:
            prob_val = risk_model.predict_proba(df_model)[:,1][0]
        except Exception:
            prob_val = None

        X_scaled = score_scaler.transform(df[base_cols])
        score_pred = score_model.predict(X_scaled)[0]

        df_result = pd.DataFrame({
            "Loại kết quả": ["Prediction", "Risk label", "Probability (%)", "Credit score"],
            "Giá trị": [risk_pred, 'High risk' if risk_pred==1 else 'Low risk',
                        round(prob_val*100,2) if prob_val is not None else np.nan,
                        round(score_pred,2)]
        })
        return df_result
    except Exception as e:
        traceback.print_exc()
        return pd.DataFrame({"Loại kết quả":["Error"],"Giá trị":[str(e)]})

# ----------------- 7. Giao diện Gradio -----------------
with gr.Blocks(title="Credit Risk & Score Assessment", theme=gr.themes.Soft()) as demo:
    gr.Markdown("## 🗂️ Đánh giá rủi ro & điểm tín dụng\nBạn có thể upload file hoặc nhập dữ liệu trực tiếp.")

    with gr.Tab("Upload file batch"):
        file_input = gr.File(label="Upload CSV hoặc Excel", file_types=[".csv",".xlsx",".xls"])
        run_btn = gr.Button("🔍 Chạy phân loại")
        output_table = gr.Dataframe(headers=None, label="Bảng kết quả")
        
        fig_counts = gr.Plot(label="Biểu đồ số lượng")
        counts_md = gr.Markdown(label="Insight số lượng")
        
        fig_pie = gr.Plot(label="Biểu đồ tỷ lệ Low vs High")
        pie_md = gr.Markdown(label="Insight tỷ lệ")
        
        fig_prob = gr.Plot(label="Histogram xác suất")
        prob_md = gr.Markdown(label="Insight xác suất")

        download_btn = gr.File(label="Tải CSV kết quả")
        summary_md = gr.Markdown()

        run_btn.click(
            process_file,
            inputs=[file_input],
            outputs=[
                output_table, 
                fig_counts, counts_md, 
                fig_pie, pie_md, 
                fig_prob, prob_md, 
                download_btn, 
                summary_md
            ]
        )

    with gr.Tab("Nhập dữ liệu thủ công"):
        inputs = [gr.Number(label=c, value=0) for c in base_cols]
        predict_btn = gr.Button("🔍 Dự đoán")
        output_single = gr.Dataframe(headers=["Loại kết quả","Giá trị"])
        predict_btn.click(
            predict_single,
            inputs=inputs,
            outputs=output_single
        )

if __name__=="__main__":
    port = int(os.environ.get('PORT', 7860))
    demo.launch(
        server_name='127.0.0.1',
        server_port=port,
        share=False
    )
