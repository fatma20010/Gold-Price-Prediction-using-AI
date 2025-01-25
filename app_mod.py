# app_fr_final.py
# app_fr_final.py
import streamlit as st
import joblib
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

def charger_artefacts():
    try:
        modele = joblib.load('SVR.joblib')
        scaler = joblib.load('scaler.joblib')
        return modele, scaler
    except Exception as e:
        st.error(f"Erreur de chargement : {e}")
        return None, None

def main():
    st.title("📈 Système de Prédiction Financière")
    st.write("Application de prédiction avancée avec visualisations interactives")
    
    if 'historique' not in st.session_state:
        st.session_state.historique = []
    
    modele, scaler = charger_artefacts()
    
    if modele and scaler:
        st.sidebar.header("Paramètres d'Entrée")
        
        entrees = {
            'PrixArgent': st.sidebar.number_input("PrixArgent", value=18.5),
            'Day': st.sidebar.number_input("Day", value=15, min_value=1, max_value=31),
            'month': st.sidebar.number_input("month", value=6, min_value=1, max_value=12),
            'year': st.sidebar.number_input("year", value=2023, min_value=2000, max_value=2030),
            'Réserve extérieur': st.sidebar.number_input("Réserve extérieur", value=120.5),
            'Export': st.sidebar.number_input("Export", value=250.0),
            'Import': st.sidebar.number_input("Import", value=230.0),
            'Indice des prix à la consommation': st.sidebar.number_input("Indice des prix à la consommation", value=105.2),
            'revenu personnel': st.sidebar.number_input("revenu personnel", value=45000.0),
            'Dépenses de consommation personnelle': st.sidebar.number_input("Dépenses de consommation personnelle", value=42000.0)
        }
        
        if st.sidebar.button("Prédire"):
            input_array = np.array([list(entrees.values())])
            entrees_scalees = scaler.transform(input_array)
            prediction = modele.predict(entrees_scalees)[0]
            
            st.session_state.historique.append({
                **entrees,
                'Prédiction': prediction
            })
            
            st.subheader("Tableau de Bord Analytique")
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("### Prédiction Actuelle")
                st.markdown(f"<h1 style='text-align: center; color: #1f77b4;'>{prediction:.2f}</h1>", 
                          unsafe_allow_html=True)
                
                fig = px.bar(x=list(entrees.keys()), 
                           y=list(entrees.values()),
                           labels={'x': 'Caractéristiques', 'y': 'Valeurs'},
                           title="Valeurs des Paramètres")
                st.plotly_chart(fig)

            with col2:
                st.markdown("### Répartition des Valeurs")
                fig = px.pie(values=list(entrees.values()), 
                           names=list(entrees.keys()),
                           hole=0.3)
                st.plotly_chart(fig, use_container_width=True)
            
            st.markdown("---")
            st.markdown("### Analyse Historique")
            
            if len(st.session_state.historique) > 0:
                df_historique = pd.DataFrame(st.session_state.historique)
                
                tab1, tab2, tab3 = st.tabs([
                    "Tendance", "3D", "Statistiques"
                ])
                
                with tab1:
                    st.markdown("#### Évolution des Prédictions")
                    fig = px.line(df_historique, y='Prédiction', markers=True)
                    st.plotly_chart(fig)
                
                with tab2:
                    st.markdown("#### Visualisation 3D")
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        x_axe = st.selectbox("Axe X", entrees.keys(), index=0)
                    with col2:
                        y_axe = st.selectbox("Axe Y", entrees.keys(), index=1)
                    with col3:
                        z_axe = st.selectbox("Axe Z", entrees.keys(), index=2)
                    
                    fig = px.scatter_3d(df_historique,
                                      x=x_axe,
                                      y=y_axe,
                                      z=z_axe,
                                      color='Prédiction')
                    st.plotly_chart(fig)
                
                with tab3:
                    st.markdown("#### Analyse Statistique")
                    col1, col2 = st.columns(2)
                    with col1:
                        fig = px.histogram(df_historique, x='Prédiction', nbins=10)
                        st.plotly_chart(fig)
                    with col2:
                        fig = px.box(df_historique, y='Prédiction')
                        st.plotly_chart(fig)
                
                with st.expander("Voir Historique Complet"):
                    st.dataframe(
                        df_historique.style.format({
                            'Prédiction': '{:.2f}',
                            **{k: '{:.2f}' for k in entrees.keys() if isinstance(entrees[k], float)}
                        })
                    )
            else:
                st.write("Aucune donnée historique disponible")

if __name__ == "__main__":
    main()