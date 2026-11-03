import os
import uproot
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px

# variable pour stocker le nom du dossier dans lequel je vais sauvegarder les differents histograms
# "histograms_plots" est un chemin relatif, python va ce debrouiller
dossier_sortie = "histograms_plots"

def charge_donnees(chemin_fichier, branches=None):
    # Je recupere la ou les branches du fichier root au format pandas data frame
    return uproot.open(f"{chemin_fichier}:DecayTree").arrays(branches, library="pd")

# f"{} -> f-string (format string) sert a insere directement la valeur dune variable dans du texte cool

# tracage dun ou des histogramms
def dessine_branche(df, noms_branches, nbins=169, dossier="histograms_plots"):

    # boucle qui parcourt noms_branches element par element
    for nom in noms_branches:
        # figure interactive de plotly
        fig = px.histogram(
            df, # table de donnees
            x=nom, # definition de la colonne du tableau df a mettre sur laxe x
            nbins=nbins, # nombre de barres
            title=f"Distribution de {nom}", # titre de lhistogram
            template="plotly_dark", # Theme sombre de plotly
            opacity=0.85 # opacite de lhistogram
        )
        fig.update_layout(
            xaxis_title=nom, # titre de laxe x
            yaxis_title="Nombre d'évènements", # titre de laxe y
            bargap=0.03, # distances entre les barres
            title_x=0.5 # position du titre de laxe x au centre du graphique
        )
        # sauvegarde de lhistogram en fichier html, automatique gere par plotly cool
        fig.write_html(f"{dossier}/hist_{nom}.html")

# tracage de plusieurs histograms sur un graphique
def dessine_branche_multiple(dict_dfs, noms_branches, nbins=169, normaliser=True, dossier="histograms_plots"):
    # dict_dfs - dictionnaire python qui contien les criteres applique sur les data frames
    # normaliser -> booleen, True quand les histograms sont normaliser pour que laire totale soit egale a 1
    
    # si normaliser est True alors norme recoit du texte sinon elle est none
    if normaliser:
        norme = "probability density"
    else:
        norme = None

    # adaptation du texte sur laxe des y en fonction de la variable normaliser
    if normaliser:
        unite_y = "Densité de probabilité"
    else:
        unite_y = "Nombre d'évènements"

    # boucle qui parcours les differents data frames
    for nom in noms_branches:

        # initialisation dune figure vide 
        fig = go.Figure()

        # boucle qui poarcours les clefs et les valeurs du dictionnaire
        for label, df in dict_dfs.items():
            # ajoute un graphique sur la figure en cours
            fig.add_trace(go.Histogram(
                x=df[nom].dropna(), # extraction de la colonne pour la metre sur laxe des x
                name=label,
                nbinsx=nbins,
                histnorm=norme, # application de la norme
                opacity=0.65
            ))

        fig.update_layout(
            title=f"Comparaison : {nom}", # titre du graphique
            title_x=0.5, 
            xaxis_title=nom, 
            yaxis_title=unite_y,
            barmode="overlay", # superposition des histogrammes par plotly
            template="plotly_dark",
            bargap=0.02
        )

        fig.write_html(f"{dossier}/superposition_{nom}.html") # enregistremeent entant que fichier html



df_total = charge_donnees("data/MasterclassData_2012_all.root")
# visualisation des donnees
print("Branches disponibles :", df_total.columns.tolist())

# creation des differents histogrammes individuellement
dessine_branche(df_total, df_total.columns.tolist())

# .tolist() -> recupere tout les entetes des branches D0_MM ect.

# selection simple sur lidentification des particules 
masque_sel = (df_total["Kplus_ProbNNk"] > 0.95) & (df_total["piminus_ProbNNpi"] > 0.8)
# application du masque sur les tableau 
df_selectionne = df_total[masque_sel]

# nouvelle figure
fig_sel = go.Figure()

fig_sel.add_trace(go.Histogram(
    x=df_total["D0_MM"], # le masque deja applique a partire des probabilite
    name="Avant sélection",
    nbinsx=120,
    histnorm="probability density",
    opacity=0.6
))
fig_sel.add_trace(go.Histogram(
    x=df_selectionne["D0_MM"],
    name="Après coupures ProbNN",
    nbinsx=120,
    histnorm="probability density",
    opacity=0.6
))
fig_sel.update_layout(
    title="Effet des coupures ProbNN sur le pic de masse du D⁰",
    title_x=0.5,
    xaxis_title="Masse D0",
    xaxis_range=[1800, 1930],
    yaxis_title="Densité de probabilité",
    barmode="overlay",
    template="plotly_dark",
    bargap=0.02
)
fig_sel.write_html(f"{dossier_sortie}/comparaison_selection.html")

# on prend que les evenements qui contienne la mesure de la masse de D
df_signal_region = df_total[(df_total["D0_MM"] >= 1850) & (df_total["D0_MM"] <= 1882)]
# on predns les mesures qui ce situe en dehors de la mesure de la masse de D
df_bruit_region = df_total[((df_total["D0_MM"] >= 1814) & (df_total["D0_MM"] <= 1849)) | 
                         ((df_total["D0_MM"] >= 1883) & (df_total["D0_MM"] <= 1914))]

# Superposition Signal vs Bruit
dessine_branche_multiple(
    {"Région Pic (Signal dominant)": df_signal_region, "Bandes Latérales (Bruit)": df_bruit_region},
    ["D0_DIRA_OWNPV", "D0_MINIPCHI2", "Kplus_ProbNNk", "piminus_ProbNNpi"]
    # on lui donne un dictionaire 
)

print(f"Tracés Plotly terminés et enregistrés dans : {dossier_sortie}")

# 3.1 Preambule

# We can see that the Measured Mass of the D0 is contained in the branch D0_MM
# the lifetime of the D0 is contained in the branch D0_TAU
# and the transverse momentum of each particle is contained in the branches D0_PT, Kplus_PT, piminus_PT
# the probability of a kaon candidate being identified as pion or a kaon is stored in the branches Kplus_ProbNNpi and Kplus_ProbNNk
# the probability of a pion candidate being identified as pion or a kaon is stored in the branches piminus_ProbNNpi and piminus_ProbNNk

# 3.2 Selection

# The easiest data from which we can 'extract' the signal by a simple selection with as little noise as possible is the probability histograms of the kaons and the pions because of the probability peaks that are easy to spot.
# I think the best easy way to seperate the data with the noise and the important data is to check the full histograms and find a value on the Y axis where i will 'cut' the histogram.