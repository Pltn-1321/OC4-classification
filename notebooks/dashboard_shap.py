import streamlit as st
import shap
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import joblib

st.set_page_config(page_title="Attrition Explainer", layout="wide")
st.title("🎯 Dashboard d'Explicabilité - Attrition")



@st.cache_resource
def load_model_and_data():
    log_reg_model = joblib.load("log_reg_pipeline.joblib")
    X_test = joblib.load("X_test.joblib")
    X_train = joblib.load("X_train.joblib")   # <-- Ajout ici
    y_test = joblib.load("y_test.joblib")
    
    ct = log_reg_model.named_steps['columntransformer']
    lr = log_reg_model.named_steps['logisticregression']

    X_test_processed = ct.transform(X_test)
    feature_names = ct.get_feature_names_out()
    X_test_df = pd.DataFrame(X_test_processed, columns=feature_names)

    # SHAP Explainer : fit sur le train transformé
    explainer = shap.LinearExplainer(lr, ct.transform(X_train))
    shap_vals = explainer.shap_values(X_test_processed)
    if isinstance(shap_vals, list) and len(shap_vals) == 2:
        shap_values = shap_vals[1]
    else:
        shap_values = shap_vals

    return log_reg_model, X_test, X_test_df, shap_values, explainer, feature_names



model, X_test_orig, X_test_df, shap_values, explainer, feature_names = load_model_and_data()

st.sidebar.header("⚙️ Configuration")
employee_idx = st.sidebar.selectbox("Sélectionner un employé", 
                                    range(len(X_test_orig)),
                                    format_func=lambda x: f"Employé #{x}")
n_features = st.sidebar.slider("Nombre de features à afficher", 5, 20, 10)

tab1, tab2, tab3, tab4 = st.tabs([
    "📊 Vue d'ensemble", "👤 Prédiction individuelle", 
    "📈 Feature Importance", "🔍 Analyse de dépendance"
])

with tab1:
    st.header("Vue d'ensemble du modèle")
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Summary Plot (Beeswarm)")
        fig, ax = plt.subplots(figsize=(10, 8))
        shap.summary_plot(shap_values, X_test_df, max_display=n_features, show=False)
        st.pyplot(fig)
        plt.close(fig)
    with col2:
        st.subheader("Feature Importance")
        fig, ax = plt.subplots(figsize=(10, 8))
        shap.summary_plot(shap_values, X_test_df, plot_type="bar", max_display=n_features, show=False)
        st.pyplot(fig)
        plt.close(fig)

with tab2:
    st.header(f"Analyse de l'employé #{employee_idx}")
    pred = model.predict(X_test_orig.iloc[[employee_idx]])[0]
    proba = model.predict_proba(X_test_orig.iloc[[employee_idx]])[0][1]
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Prédiction", "Attrition" if pred == 1 else "Rétention")
    with col2:
        st.metric("Probabilité d'attrition", f"{proba:.1%}")
    with col3:
        st.metric("Risque", 
                  "🔴 Élevé" if proba > 0.7 else "🟡 Moyen" if proba > 0.4 else "🟢 Faible")
    st.divider()
    with st.expander("📋 Voir les données de l'employé"):
        st.dataframe(X_test_orig.iloc[[employee_idx]].T)
    st.subheader("🌊 Waterfall Plot - Contribution des features")
    fig, ax = plt.subplots(figsize=(10, 8))
    shap.plots.waterfall(
        shap.Explanation(
            values=shap_values[employee_idx],
            base_values=explainer.expected_value,
            data=X_test_df.iloc[employee_idx].values,
            feature_names=feature_names
        ),
        max_display=n_features,
        show=False
    )
    st.pyplot(fig)
    plt.close(fig)
    st.subheader("⚡ Force Plot")
    fig, ax = plt.subplots(figsize=(20, 3))
    shap.force_plot(
        explainer.expected_value,
        shap_values[employee_idx],
        X_test_df.iloc[employee_idx],
        matplotlib=True,
        show=False
    )
    st.pyplot(fig)
    plt.close(fig)
    st.subheader("🎯 Top 5 facteurs de cette prédiction")
    shap_contrib = pd.DataFrame({
        'Feature': feature_names,
        'SHAP Value': shap_values[employee_idx]
    })
    shap_contrib['Impact'] = shap_contrib['SHAP Value'].apply(
        lambda x: '⬆️ Augmente risque' if x > 0 else '⬇️ Diminue risque'
    )
    top_features = shap_contrib.reindex(
        shap_contrib['SHAP Value'].abs().sort_values(ascending=False).index
    ).head(5)
    st.dataframe(top_features, use_container_width=True)

with tab3:
    st.header("Importance des features")
    mean_abs_shap = np.abs(shap_values).mean(0)
    feature_importance_df = pd.DataFrame({
        'Feature': feature_names,
        'Importance': mean_abs_shap
    }).sort_values('Importance', ascending=False)
    fig, ax = plt.subplots(figsize=(10, 8))
    top_n = feature_importance_df.head(n_features)
    ax.barh(range(len(top_n)), top_n['Importance'])
    ax.set_yticks(range(len(top_n)))
    ax.set_yticklabels(top_n['Feature'])
    ax.invert_yaxis()
    ax.set_xlabel('Mean |SHAP value|')
    ax.set_title(f'Top {n_features} Features')
    st.pyplot(fig)
    plt.close(fig)
    st.subheader("📊 Tableau complet")
    st.dataframe(feature_importance_df, use_container_width=True)

with tab4:
    st.header("Analyse de dépendance des features")
    selected_feature = st.selectbox("Choisir une feature à analyser", feature_names)
    feature_idx = list(feature_names).index(selected_feature)
    st.subheader(f"Dépendance : {selected_feature}")
    fig, ax = plt.subplots(figsize=(10, 6))
    shap.dependence_plot(feature_idx, shap_values, X_test_df, show=False)
    st.pyplot(fig)
    plt.close(fig)
    st.info("""
    📖 **Interprétation** :
    - L'axe X montre les valeurs de la feature
    - L'axe Y montre l'impact SHAP (contribution à la prédiction)
    - La couleur représente une autre feature (interaction)
    """)

st.sidebar.divider()
st.sidebar.info("""
📊 **Dashboard SHAP Interactif**
- Analysez les prédictions individuelles
- Explorez les features importantes
- Comprenez les décisions du modèle
""")
