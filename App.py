import pickle
import streamlit as st
import os
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

# ── Page config ──────────────────────────────────────────
st.set_page_config(page_title="Bank Churn Predictor", page_icon="🏦", layout="wide")

# ── Load model ───────────────────────────────────────────
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
model_path = os.path.join(BASE_DIR, "model_simple.pkl")

with open(model_path, "rb") as f:
    model = joblib.load(model_path)


# ── Sidebar sliders ──────────────────────────────────────
st.sidebar.header("🎛️ Customer Features")

amt = st.sidebar.slider(
    "💳 Total Transaction Amount", min_value=510, max_value=18484, value=5000, step=10
)

ct = st.sidebar.slider(
    "🔢 Total Transaction Count", min_value=10, max_value=139, value=60, step=1
)

bal = st.sidebar.slider(
    "💰 Total Revolving Balance", min_value=0, max_value=2517, value=1000, step=10
)

# ── Main page ─────────────────────────────────────────────
st.title("🏦 Bank Churn Predictor")


st.divider()

# ── Show selected values ──────────────────────────────────
col1, col2, col3 = st.columns(3)
col1.metric("💳 Trans Amount", f"{amt:,}")
col2.metric("🔢 Trans Count", ct)
col3.metric("💰 Revolving Bal", f"{bal:,}")

st.divider()

# ── Predict button ────────────────────────────────────────
if st.button("🔍 Predict", use_container_width=True):
    pred = model.predict([[amt, ct, bal]])
    if pred[0] == 1:
        st.error("⚠️ Attrited Customer — العميل هيمشي!")
    else:
        st.success("✅ Existing Customer — العميل هيفضل!")


# ── Data Preview ──────────────────────────────────────────
st.divider()
st.subheader("📊 Dataset Preview")

data_path = os.path.join(BASE_DIR, "BankChurners.csv")
df = pd.read_csv(data_path)

st.dataframe(df, use_container_width=True)
st.caption(f"📁 {df.shape[0]:,} rows × {df.shape[1]} columns")


# ── Dashboard ─────────────────────────────────────────────
st.divider()
st.subheader("📈 Dashboard")

tab1, tab2, tab3, tab4 = st.tabs(
    ["👥 Demographics", "💳 Cards & Income", "🔗 Correlations", "🎯 Churn Analysis"]
)

with tab1:
    col1, col2 = st.columns(2)

    with col1:
        st.markdown("**Gender Distribution**")
        fig1, ax1 = plt.subplots(figsize=(4, 4))
        Gender = df.groupby("Gender")["CLIENTNUM"].count()
        wedges, texts, autotexts = ax1.pie(
            Gender, autopct="%1.1f%%", colors=["#c2477a", "#378add"], startangle=90
        )
        ax1.set_title("Gender Distribution")
        ax1.legend(
            wedges, Gender.index, loc="lower center", bbox_to_anchor=(0.5, -0.1), ncol=2
        )
        st.pyplot(fig1, use_container_width=True)
        plt.close()

    with col2:
        st.markdown("**Attrited vs Existing**")
        fig2, ax2 = plt.subplots(figsize=(4, 4))
        Existant_Customer = df.groupby("Attrition_Flag")["CLIENTNUM"].count()
        wedges2, texts2, autotexts2 = ax2.pie(
            Existant_Customer, autopct="%1.1f%%", startangle=90
        )
        ax2.set_title("Attrited Vs Existing")
        ax2.legend(
            wedges2,
            Existant_Customer.index,
            loc="lower center",
            bbox_to_anchor=(0.5, -0.1),
            ncol=2,
        )
        st.pyplot(fig2, use_container_width=True)
        plt.close()

    st.markdown("**Education Level**")
    fig3, ax3 = plt.subplots()
    colors_edu = [
        "#0d4a2a",
        "#1a7a4a",
        "#2a9e60",
        "#53b87a",
        "#85d4a0",
        "#b3e8c4",
        "#daf5e4",
    ]
    Education_Level = df.groupby("Education_Level")["CLIENTNUM"].count()
    Education_Level.plot(kind="bar", color=colors_edu, ax=ax3)
    ax3.set_title("Education Level")
    plt.xticks(rotation=45, ha="right")
    for i, v in enumerate(Education_Level):
        ax3.text(i, v + 10, str(v), ha="center", fontsize=9)
    plt.tight_layout()
    st.pyplot(fig3)
    plt.close()

with tab2:
    col1, col2 = st.columns(2)

    with col1:
        st.markdown("**Marital Status**")
        fig4, ax4 = plt.subplots()
        colors_mar = ["#1a6fb5", "#2a8fd4", "#53a8e0", "#85c4ee"]
        Marital_Status = df.groupby("Marital_Status")["CLIENTNUM"].count()
        Marital_Status.plot(kind="barh", color=colors_mar, ax=ax4)
        for i, v in enumerate(Marital_Status):
            ax4.text(v + 10, i, str(v), va="center", fontsize=9)
        plt.tight_layout()
        st.pyplot(fig4)
        plt.close()

    with col2:
        st.markdown("**Income Category**")
        fig5, ax5 = plt.subplots()
        colors_inc = [
            "#2a0d6e",
            "#4a1a9e",
            "#6a2ac8",
            "#8f53e0",
            "#b985ee",
            "#d8b3f5",
            "#eedaff",
        ]
        Income_Category = df.groupby("Income_Category")["CLIENTNUM"].count()
        Income_Category.plot(kind="bar", color=colors_inc, ax=ax5)
        plt.xticks(rotation=15, ha="right")
        for i, v in enumerate(Income_Category):
            ax5.text(i, v + 10, str(v), ha="center", fontsize=9)
        plt.tight_layout()
        st.pyplot(fig5)
        plt.close()

    st.markdown("**Card Category**")
    fig6, ax6 = plt.subplots()
    colors_card = ["#1a6fb5", "#c8a000", "#b0b0b0", "#c0c0c0"]
    Card_Category = df.groupby("Card_Category")["CLIENTNUM"].count()
    Card_Category.plot(kind="bar", color=colors_card, ax=ax6)
    plt.xticks(rotation=15, ha="right")
    for i, v in enumerate(Card_Category):
        ax6.text(i, v + 10, str(v), ha="center", fontsize=9)
    plt.tight_layout()
    st.pyplot(fig6)
    plt.close()

    st.markdown("**Card Category vs Age**")
    fig7, ax7 = plt.subplots()
    sns.boxplot(data=df, x="Card_Category", y="Customer_Age", ax=ax7)
    ax7.set_title("Card Category vs Age")
    st.pyplot(fig7)
    plt.close()

    st.markdown("**Card Category vs Gender**")
    fig8, ax8 = plt.subplots()
    pd.crosstab(df["Card_Category"], df["Gender"]).plot(kind="bar", ax=ax8)
    ax8.set_title("Card Category vs Gender")
    plt.tight_layout()
    st.pyplot(fig8)
    plt.close()

with tab3:
    st.markdown("**Correlation Heatmap**")
    df_encoded = df.copy()
    df_encoded = df_encoded.iloc[:, :-2]
    from sklearn.preprocessing import LabelEncoder

    le = LabelEncoder()
    for col in df_encoded.select_dtypes(include="object").columns:
        df_encoded[col] = le.fit_transform(df_encoded[col])
    fig9, ax9 = plt.subplots(figsize=(16, 12))
    sns.heatmap(
        df_encoded.corr(),
        annot=True,
        fmt=".1f",
        cmap="Blues",
        annot_kws={"size": 7},
        linewidths=0.5,
        linecolor="white",
        ax=ax9,
    )
    ax9.set_title("Correlation Heatmap", fontsize=14)
    plt.xticks(rotation=45, ha="right", fontsize=8)
    plt.yticks(fontsize=8)
    plt.tight_layout()
    st.pyplot(fig9)
    plt.close()

    st.markdown("**Confusion Matrix - Simple Model**")
    from sklearn.preprocessing import LabelEncoder
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import confusion_matrix

    df_model = df.copy()
    df_model = df_model.iloc[:, :-2]
    le2 = LabelEncoder()
    for col in df_model.select_dtypes(include="object").columns:
        df_model[col] = le2.fit_transform(df_model[col])

    X_simple = df_model[["Total_Trans_Amt", "Total_Trans_Ct", "Total_Revolving_Bal"]]
    y = df_model["Attrition_Flag"]
    X_train_s, X_test_s, y_train_s, y_test_s = train_test_split(
        X_simple, y, test_size=0.2, random_state=42
    )
    y_pred_s = model.predict(X_test_s)

    cm = confusion_matrix(y_test_s, y_pred_s)
    fig10, ax10 = plt.subplots()
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=["Attrited", "Existing"],
        yticklabels=["Attrited", "Existing"],
        ax=ax10,
    )
    ax10.set_title("Confusion Matrix - Simple Model")
    ax10.set_ylabel("Actual")
    ax10.set_xlabel("Predicted")
    plt.tight_layout()
    st.pyplot(fig10)
    plt.close()

with tab4:
    st.markdown("**Churn Rate by Category**")
    cat_cols = [
        "Gender",
        "Education_Level",
        "Marital_Status",
        "Income_Category",
        "Card_Category",
    ]
    fig10, axes = plt.subplots(2, 3, figsize=(16, 10))
    for ax, col in zip(axes.flatten(), cat_cols):
        pd.crosstab(df[col], df["Attrition_Flag"], normalize="index").plot(
            kind="bar", ax=ax, color=["#1a6fb5", "#c2477a"]
        )
        ax.set_title(col)
        ax.set_xlabel("")
        ax.tick_params(axis="x", rotation=45)
    axes[-1, -1].set_visible(False)
    plt.tight_layout()
    st.pyplot(fig10)
    plt.close()
